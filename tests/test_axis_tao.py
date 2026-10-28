"""Axis proposals must minimize the current child losses and preserve architecture."""
import copy
import pickle
import unittest
import numpy as np
from tao.axis import AxisTAOClassifier,assert_axis,axis_parameter_counts
from tao.hard import cross_entropy_rows
from tao.initialization import initialize_cart
from tao.epoch import begin_epoch,step_epoch
from tao.support import freeze_support


class AxisTAOTests(unittest.TestCase):
    def test_reject_split_penalty_and_oblique_resume(self):
        with self.assertRaises(ValueError):AxisTAOClassifier(split_alpha=1e-5).initialize(2,2)
        m=AxisTAOClassifier(depth=1,split_alpha=0).initialize(2,2)
        m.trees[0].split_w[0]=[1.,1.]
        with self.assertRaises(ValueError):assert_axis(m)

    def test_exact_search_matches_exhaustive_hard_loss_binary_and_multiclass(self):
        rng=np.random.default_rng(4)
        for classes in [2,3]:
            for _ in range(5):
                x=rng.integers(-2,3,size=(31,3)).astype(float);y=rng.integers(classes,size=len(x))
                weights=rng.integers(0,4,size=len(x)).astype(float);offset=rng.normal(size=(len(x),classes))
                m=AxisTAOClassifier(depth=1,split_alpha=0,leaf_alpha=1e-5).initialize(3,classes)
                t=m.trees[0];t.leaf_parameters=rng.normal(size=t.leaf_parameters.shape)
                losses=[cross_entropy_rows(offset+t.predict_subtree(x,c,'dense'),y) for c in [1,2]]
                values=[]
                for feature in range(3):
                    for sign in [1.,-1.]:
                        scores=sign*x[:,feature];u=np.unique(scores)
                        for threshold in np.r_[np.nextafter(u[0],-np.inf),u]:
                            values.append(float(weights@np.where(scores>threshold,losses[1],losses[0])/weights.sum()))
                result=m._split_update(t,0,x,y,weights,offset,weights.sum())
                self.assertAlmostEqual(result['objective_after'],min(values),places=12)
                self.assertLessEqual(result['objective_after'],result['objective_before'])
                assert_axis(m)

    def test_threshold_direction_and_equality(self):
        x=np.array([[-2.],[-1.],[0.],[1.],[2.]])
        m=AxisTAOClassifier(depth=1,split_alpha=0).initialize(1,2);t=m.trees[0]
        t.leaf_parameters[:]=0;t.leaf_parameters[0,-1,0]=-3;t.leaf_parameters[1,-1,0]=3
        # Signal class 1 goes to the first leaf, requiring a negative direction.
        y=np.array([0,0,0,1,1]);t.split_w[0]=1;t.split_b[0]=0
        result=m._split_update(t,0,x,y,np.ones(len(y)),np.zeros((len(y),2)),len(y))
        self.assertTrue(result['accepted']);self.assertEqual(t.split_w[0,0],-1.)
        threshold=-t.split_b[0];scores=x[:,0]*t.split_w[0,0]
        np.testing.assert_array_equal(t.leaf_indices(x,'dense')==0,scores<=threshold)

    def test_tied_subtrees_leave_cut_unchanged(self):
        m=AxisTAOClassifier(depth=1,split_alpha=0).initialize(2,2);t=m.trees[0];t.leaf_parameters[:]=0
        before=(t.split_w.copy(),t.split_b.copy())
        result=m._split_update(t,0,np.zeros((5,2)),np.array([0,1,0,1,0]),np.ones(5),np.zeros((5,2)),5)
        self.assertFalse(result['accepted']);self.assertEqual(result['active_events'],0)
        np.testing.assert_array_equal(t.split_w,before[0]);np.testing.assert_array_equal(t.split_b,before[1])

    def test_resumable_epoch_matches_direct_and_keeps_axis_invariant(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(120,3));y=rng.integers(2,size=len(x))
        m=AxisTAOClassifier(depth=2,trees=2,split_alpha=0,leaf_alpha=.01,leaf_max_iter=500,indexed='dense').initialize(3,2)
        initialize_cart(m,x,y);other=copy.deepcopy(m)
        m.fit_epoch(x,y,global_refit=False)
        work=begin_epoch(other,x,y)
        for _ in range(100):
            other,work=pickle.loads(pickle.dumps((other,work)))
            if step_epoch(other,work,x,y):break
        else:self.fail('Epoch did not finish')
        np.testing.assert_array_equal(m.decision_function(x),other.decision_function(x));assert_axis(other)

    def test_leaf_only_refit_freezes_cuts_and_selected_zeros(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(150,2));y=rng.integers(2,size=len(x))
        m=AxisTAOClassifier(depth=1,split_alpha=0,leaf_alpha=.01,leaf_max_iter=500).initialize(2,2)
        initialize_cart(m,x,y);m.trees[0].leaf_parameters[:,0]=0
        frozen=freeze_support(m);before=copy.deepcopy(frozen.trees[0]);frozen.global_refit(x,y)
        np.testing.assert_array_equal(frozen.trees[0].split_w,before.split_w)
        np.testing.assert_array_equal(frozen.trees[0].split_b,before.split_b)
        np.testing.assert_array_equal(frozen.trees[0].leaf_parameters[:,0],0)
        assert_axis(frozen)

    def test_counts_separate_continuous_parameters_from_feature_choices(self):
        m=AxisTAOClassifier(depth=4,trees=3,split_alpha=0).initialize(16,2)
        counts=axis_parameter_counts(m)
        self.assertEqual(counts['continuous_slots'],861)
        self.assertEqual(counts['feature_index_slots'],45)
        self.assertEqual(counts['direction_bit_slots'],45)
        self.assertEqual(counts['routing_multiplications'],0)

if __name__=='__main__':unittest.main()
