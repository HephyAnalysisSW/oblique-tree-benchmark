import unittest
import numpy as np
from sklearn.metrics import roc_curve
from tao.uncertainty import WeightedWorkingPoint,paired_intervals


class WorkingPointUncertaintyTests(unittest.TestCase):
    def test_weighted_grouping_matches_sklearn_with_ties_and_zero_weights(self):
        rng=np.random.default_rng(88);y=np.tile([0,1],100)
        for tied in [False,True]:
            scores=rng.random(200)
            if tied:scores=np.round(scores,1)
            calculator=WeightedWorkingPoint(y,scores)
            for _ in range(20):
                counts=rng.integers(0,5,size=200)
                fpr,tpr,_=roc_curve(y,scores,sample_weight=counts)
                self.assertAlmostEqual(calculator.evaluate(counts),np.interp(.01,fpr,tpr),places=14)

    def test_identical_models_have_exactly_zero_paired_difference_and_reproducible_intervals(self):
        rng=np.random.default_rng(7);y=np.tile([0,1],50);scores=rng.random(100)
        a=paired_intervals(y,dict(XGBoost=scores,identical=scores.copy()),repeats=30,comparisons=dict(same=('identical','XGBoost')))
        b=paired_intervals(y,dict(XGBoost=scores,identical=scores.copy()),repeats=30,comparisons=dict(same=('identical','XGBoost')))
        self.assertEqual(a,b);self.assertEqual(a['models']['identical']['difference_interval95'],[0,0])
        self.assertEqual(a['models']['identical']['interval95'],a['models']['XGBoost']['interval95'])
        self.assertEqual(a['comparisons']['same']['interval95'],[0,0])

    def test_invalid_inputs_and_empty_class_mass_are_rejected(self):
        with self.assertRaises(ValueError):WeightedWorkingPoint([0,0],[.1,.2])
        calculator=WeightedWorkingPoint([0,1],[.1,.9])
        for counts in [[0,1],[-1,1],[1,np.nan]]:
            with self.assertRaises(ValueError):calculator.evaluate(counts)
        with self.assertRaises(ValueError):paired_intervals([0,1],dict(other=[.1,.9]),repeats=30)
