import unittest
import numpy as np
from unittest.mock import patch
from tao.hard import HardTAOClassifier
from tao.split_search import best_threshold


class HardThresholdTests(unittest.TestCase):
    def test_exact_threshold_matches_exhaustive_routings_with_ties_and_penalty(self):
        rng=np.random.default_rng(17)
        for _ in range(30):
            scores=rng.integers(-4,5,size=60).astype(float)
            left,right=rng.uniform(0,5,size=(2,60));weights=rng.uniform(0,2,size=60)
            alpha=.2;norm=100
            bias,_=best_threshold(scores,left,right,weights,norm,alpha)
            def objective(b):return np.dot(weights,np.where(scores+b>0,right,left))/norm+alpha*abs(b)
            # Include every representable region endpoint and the L1 kink.
            candidates=[0.]
            for score in np.unique(scores):candidates.extend([-score,-np.nextafter(score,-np.inf)])
            self.assertAlmostEqual(objective(bias),min(map(objective,candidates)),places=12)

    def test_single_direction_recovers_split_rejected_by_raw_logistic_proposal(self):
        x=np.arange(-3,4,dtype=float)[:,None];y=(x[:,0]>1).astype(int)
        model=HardTAOClassifier(depth=1,leaf_mode='constant').initialize(1,2)
        model.split_search='hard_threshold';t=model.trees[0]
        t.split_w[:]=1;t.split_b[:]=0;t.leaf_parameters[:,0,0]=[4,-4]
        before=model.objective(x,y)
        with patch('tao.hard.fit_l1_split',return_value=(np.array([-1.]),0.,dict(converged=True,iterations=1,seconds=0.))):
            stats=model._split_update(t,0,x,y,np.ones(len(y)),np.zeros((len(y),2)),len(y))
        self.assertTrue(stats['accepted']);self.assertLess(model.objective(x,y),before)
        np.testing.assert_array_equal(t.leaf_indices(x),y)

    def test_one_sided_targets_can_route_every_event_to_better_child(self):
        x=np.arange(-2,3,dtype=float)[:,None];y=np.zeros(len(x),dtype=int)
        model=HardTAOClassifier(depth=1,leaf_mode='constant').initialize(1,2)
        model.split_search='hard_threshold';t=model.trees[0]
        t.split_w[:]=1;t.split_b[:]=0;t.leaf_parameters[:,0,0]=[3,-3]
        before=model.objective(x,y)
        stats=model._split_update(t,0,x,y,np.ones(len(y)),np.zeros((len(y),2)),len(y))
        self.assertTrue(stats['accepted']);self.assertFalse(stats['fitted'])
        self.assertLess(model.objective(x,y),before)
        np.testing.assert_array_equal(t.leaf_indices(x),0)

    def test_monotonic_full_epoch_for_linear_constant_and_multiclass(self):
        rng=np.random.default_rng(18);x=rng.normal(size=(150,3))
        for mode in ['linear','constant']:
            for classes in [2,3]:
                y=rng.integers(classes,size=len(x))
                m=HardTAOClassifier(depth=2,trees=2,leaf_mode=mode,leaf_max_iter=200,split_max_iter=300).initialize(3,classes)
                m.split_search='hard_threshold';m.global_refit(x,y);before=m.objective(x,y)
                m.fit_epoch(x,y);self.assertLessEqual(m.objective(x,y),before+1e-10)
