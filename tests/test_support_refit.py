"""Independent objective, mask, separation and checkpoint tests for support refits."""
import copy,pickle,unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from tao.hard import HardTAOClassifier,fit_leaf_parameters,ce_value_gradient
from tao.support import freeze_support,continue_full_refit,separation_check
from tao.epoch import begin_epoch,step_epoch
from tao.convergence import leaf_kkt_residual


class SupportRefitTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(602)
        self.x=self.rng.normal(size=(180,3));self.y=self.rng.integers(0,3,180)

    def parent(self,mode='linear',classes=3,depth=2):
        model=HardTAOClassifier(depth=depth,trees=2,leaf_mode=mode,indexed='dense',leaf_max_iter=500,split_max_iter=500).initialize(3,classes)
        for tree in model.trees:
            tree.split_w[:,1]=0;tree.split_b[::2]=0
            tree.leaf_parameters[:]=self.rng.normal(size=tree.leaf_parameters.shape)*.1
            if mode=='linear':tree.leaf_parameters[:,1]=0
            tree.leaf_parameters[0,-1,0]=0
        return model

    def test_freezing_preserves_predictions_parent_and_exact_support(self):
        parent=self.parent();parent.trees[0].split_w[0,0]=1e-25
        before=pickle.dumps(parent);frozen=freeze_support(parent)
        np.testing.assert_array_equal(parent.decision_function(self.x),frozen.decision_function(self.x))
        self.assertEqual(pickle.dumps(parent),before)
        self.assertTrue(frozen.support_masks[0]['split_w'][0,0])
        self.assertEqual(frozen.split_alpha,0);self.assertEqual(frozen.leaf_alpha,0)
        with self.assertRaises(ValueError):freeze_support(frozen)
        for corruption in ['mask','coefficient','fixed','penalty','architecture']:
            broken=copy.deepcopy(frozen)
            if corruption=='mask':broken.support_masks[0]['leaf'].flat[0]^=True
            if corruption=='coefficient':broken.trees[0].leaf_parameters[0,-1,0]=1
            if corruption=='fixed':broken.trees[0].split_b[0]=.1
            if corruption=='penalty':broken.leaf_alpha=1e-8
            if corruption=='architecture':broken.ntrees+=1
            with self.assertRaises(ValueError):broken.predict_proba(self.x)

    def test_masked_multiclass_fit_matches_independent_reduced_optimizer(self):
        design=np.column_stack((self.x,np.ones(len(self.x))))
        initial=self.rng.normal(size=(4,2))*.1;mask=np.array([[1,0],[0,1],[1,1],[0,1]],dtype=bool);initial[~mask]=0
        offset=self.rng.normal(size=(len(self.x),3))*.2;weight=self.rng.uniform(.2,2,len(self.x));normalizer=400.
        fitted,stats=fit_leaf_parameters(design,self.y,offset,weight,normalizer,initial,[1,1,1,0],0,1000,
            ftol=1e-14,gtol=1e-10,trainable_mask=mask)
        def objective(v):
            theta=np.zeros_like(initial);theta[mask]=v
            logits=np.column_stack((design@theta,np.zeros(len(design))))+offset
            return float(weight@(logsumexp(logits,axis=1)-logits[np.arange(len(self.y)),self.y])/normalizer)
        independent=minimize(objective,initial[mask],method='BFGS',options=dict(gtol=1e-7))
        self.assertLessEqual(abs(objective(fitted[mask])-independent.fun),1e-9)
        np.testing.assert_array_equal(fitted[~mask],0)
        self.assertAlmostEqual(stats['final_objective'],objective(fitted[mask]),places=12)
        # No sign constraint: a surviving coefficient initialized positive can turn negative.
        x=np.linspace(-2,2,300)[:,None];y=(x[:,0]>0).astype(int)
        out,_=fit_leaf_parameters(x,y,None,np.ones(300),300,np.ones((1,1)),[1],0,50,trainable_mask=np.ones((1,1),bool))
        self.assertLess(out[0,0],0)

    def test_empty_mask_is_valid_and_nonzero_frozen_input_rejected(self):
        design=np.ones((10,1));initial=np.zeros((1,1));y=np.tile([0,1],5)
        out,stats=fit_leaf_parameters(design,y,None,np.ones(10),10,initial,[False],0,50,trainable_mask=np.zeros_like(initial,dtype=bool))
        np.testing.assert_array_equal(out,initial);self.assertTrue(stats['converged']);self.assertEqual(stats['iterations'],0)
        with self.assertRaises(ValueError):fit_leaf_parameters(design,y,None,np.ones(10),10,initial+1,[False],0,50,trainable_mask=np.zeros_like(initial,dtype=bool))

    def test_leaf_only_joint_and_local_refits_preserve_all_splits(self):
        for mode in ['linear','constant']:
            parent=self.parent(mode);model=freeze_support(parent);initial=model.objective(self.x,self.y)
            routes=[t.leaf_indices(self.x).copy() for t in model.trees]
            for _ in range(3):
                profile=model.fit_epoch(self.x,self.y)
                self.assertLessEqual(profile['objective_after'],profile['objective_before']+1e-12)
                model.assert_support()
            self.assertLess(model.objective(self.x,self.y),initial)
            for t,p,r in zip(model.trees,parent.trees,routes):
                np.testing.assert_array_equal(t.split_w,p.split_w);np.testing.assert_array_equal(t.split_b,p.split_b)
                np.testing.assert_array_equal(t.leaf_indices(self.x),r)
            self.assertEqual(model.support_counts()['allowed_coefficients'],freeze_support(parent).support_counts()['allowed_coefficients'])

    def test_kkt_ignores_frozen_gradients_but_checks_free_coordinates(self):
        x=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.array([0,0,1,1])
        parent=HardTAOClassifier(depth=0).initialize(1,2);parent.trees[0].leaf_parameters[:]=0
        model=freeze_support(parent)
        self.assertEqual(leaf_kkt_residual(model,x,y),0)
        parent.trees[0].leaf_parameters[0,0,0]=.1;model=freeze_support(parent)
        self.assertGreater(leaf_kkt_residual(model,x,y),.5)
        weights=np.array([1.,0,2,4]);theta=model.trees[0].leaf_parameters[0]
        _,gradient=ce_value_gradient(theta,np.column_stack((x,np.ones(4))),y,None,weights,weights.sum())
        self.assertAlmostEqual(leaf_kkt_residual(model,x,y,weights),abs(gradient[0,0]))

    def test_full_split_update_cannot_revive_weight_or_zero_bias(self):
        x=self.rng.normal(size=(300,3));y=(x[:,0]>.7).astype(int)
        parent=HardTAOClassifier(depth=1,leaf_mode='constant').initialize(3,2)
        tree=parent.trees[0];tree.split_w[:]=[.2,0,.1];tree.split_b[:]=0
        tree.leaf_parameters[:,0,0]=[2,-2]
        model=freeze_support(parent,'full');tree=model.trees[0]
        before=model.objective(x,y)
        model._split_update(tree,0,x,y,np.ones(len(y)),np.zeros((len(y),2)),len(y))
        self.assertEqual(tree.split_b[0],0);self.assertEqual(tree.split_w[0,1],0)
        self.assertLessEqual(model.objective(x,y),before)
        self.assertTrue(np.isfinite(tree.split_w).all())

    def test_constant_routes_keep_tiny_positive_bias_in_both_stages(self):
        parent=self.parent(depth=1)
        parent.trees[0].split_w[:]=0;parent.trees[0].split_b[:]=1e-25
        for stage in ['leaf_only','full']:
            model=freeze_support(parent,stage)
            for _ in range(2):model.fit_epoch(self.x,self.y)
            self.assertEqual(model.trees[0].split_b[0],1e-25)
            self.assertTrue(np.all(model.trees[0].leaf_indices(self.x)==1))
            self.assertTrue(model.support_masks[0]['leaf'][0].any())  # no data-based dead-branch deletion

    def test_separable_router_is_not_mistaken_for_unbounded_leaf_problem(self):
        x=np.array([[-2.],[-1.],[1.],[2.]]);y=np.array([0,0,1,1])
        parent=HardTAOClassifier(depth=1,leaf_mode='constant').initialize(1,2)
        tree=parent.trees[0];tree.split_w[:]=1;tree.split_b[:]=0
        tree.leaf_parameters[:,0,0]=[2,-2]
        model=freeze_support(parent,'full')
        with patch('tao.support.fit_leaf_parameters',side_effect=AssertionError('Hard router already optimal')):
            stats=model._split_update(model.trees[0],0,x,y,np.ones(4),np.zeros((4,2)),4)
        self.assertTrue(stats['converged']);self.assertEqual(stats['split_certificate'],'pointwise_lower_bound')

    def test_bias_can_move_only_if_parent_bias_survived_selection(self):
        x=np.array([[-2.,0.],[-1.,0.],[.2,0.],[.8,0.],[1.5,0.]])
        y=np.array([0,0,0,0,1])
        parent=HardTAOClassifier(depth=1,leaf_mode='constant').initialize(2,2)
        tree=parent.trees[0];tree.split_w[:]=[1.,0.];tree.split_b[:]=.1
        tree.leaf_parameters[:,0,0]=[2,-2]
        model=freeze_support(parent,'full');before=model.objective(x,y)
        model._split_update(model.trees[0],0,x,y,np.ones(5),np.zeros((5,2)),5)
        self.assertLess(model.objective(x,y),before)
        self.assertNotEqual(model.trees[0].split_b[0],.1);self.assertEqual(model.trees[0].split_w[0,1],0)

    def test_stage_transition_never_refreezes_a_free_coefficient_at_zero(self):
        model=freeze_support(self.parent());mask=model.support_masks[0]['leaf'];index=tuple(np.argwhere(mask)[0])
        model.trees[0].leaf_parameters[index]=0
        full=continue_full_refit(model)
        self.assertTrue(full.support_masks[0]['leaf'][index]);self.assertEqual(full.support_sha256,model.support_sha256)

    def test_pickle_resume_each_node_matches_complete_full_epoch(self):
        for mode in ['linear','constant']:
            for stage in ['leaf_only','full']:
                for classes in [2,3]:
                    y=self.y%classes;model=freeze_support(self.parent(mode,classes),stage);other=copy.deepcopy(model)
                    model.fit_epoch(self.x,y,global_refit=False)
                    work=begin_epoch(other,self.x,y)
                    while not step_epoch(other,work,self.x,y):other,work=pickle.loads(pickle.dumps((other,work)))
                    other.assert_support();np.testing.assert_array_equal(model.decision_function(self.x),other.decision_function(self.x))

    def test_separation_complete_quasi_overlap_and_fixed_zero_support(self):
        parent=HardTAOClassifier(depth=0).initialize(1,2);parent.trees[0].leaf_parameters[:]=.1;model=freeze_support(parent)
        for x,y,status in [([-2,-1,1,2],[0,0,1,1],'separated'),
                           ([-1,0,0,1],[0,0,1,1],'separated'),
                           ([-1,-1,1,1],[0,1,0,1],'no_recession_direction')]:
            report=separation_check(model,np.array(x)[:,None],np.array(y));self.assertEqual(report['status'],status)
        with patch('tao.support.linprog',return_value=type('Failed',(),dict(success=False,message='time limit'))()):
            self.assertEqual(separation_check(model,np.array([[-1.],[1.]]),np.array([0,1]))['status'],'inconclusive')
        parent.trees[0].leaf_parameters[:]=0;zero=freeze_support(parent)
        self.assertEqual(separation_check(zero,np.array([[-1.],[1.]]),np.array([0,1]))['status'],'no_recession_direction')

    def test_multiclass_separation_and_zero_weight_rows(self):
        parent=HardTAOClassifier(depth=0).initialize(1,3);parent.trees[0].leaf_parameters[:]=.1;model=freeze_support(parent)
        x=np.repeat([[-1.],[0.],[1.]],3,axis=0);y=np.tile(np.arange(3),3)
        self.assertEqual(separation_check(model,x,y)['status'],'no_recession_direction')
        x=np.array([[-2.],[-1.],[1.],[2.]]);y=np.array([0,0,2,2])
        self.assertEqual(separation_check(model,x,y)['status'],'separated')
        # Removing contradictory zero-weight data restores separation.
        binary=HardTAOClassifier(depth=0).initialize(1,2);binary.trees[0].leaf_parameters[:]=.1
        self.assertEqual(separation_check(freeze_support(binary),np.array([[-1.],[1.],[1.]]),np.array([0,1,0]),[1,1,0])['status'],'separated')
