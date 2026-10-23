import unittest
import numpy as np
from tao.hard import HardTAOClassifier
from tao.portable import export_model,predict_margin,deterministic_working_point,operation_counts


class PortableTests(unittest.TestCase):
    def test_reference_class_sign_boundary_and_constant_leaves(self):
        m=HardTAOClassifier(depth=1,leaf_mode='constant').initialize(1,2)
        t=m.trees[0];t.split_w[:]=1;t.split_b[:]=0;t.leaf_parameters[:,0,0]=[2,-3]
        spec=export_model(m,['x'],['b','s'],[0],[1]);margin,routes,_=predict_margin(spec,[[-1],[-0.],[0.],[1]])
        np.testing.assert_array_equal(routes[:,0],[0,0,0,1]);np.testing.assert_array_equal(margin,[-2,-2,-2,3])
        np.testing.assert_allclose(m.predict_proba([[-1],[1]])[:,1],1/(1+np.exp(-margin[[0,3]])),atol=1e-7)

    def test_raw_folding_and_independent_inference_match_original_forest(self):
        rng=np.random.default_rng(17);x=rng.normal(size=(100,3));mean=np.array([2.,-1.,.5]);scale=np.array([.5,2.,4.])
        for mode in ['linear','constant']:
            m=HardTAOClassifier(depth=3,trees=3,leaf_mode=mode).initialize(3,2)
            for tree in m.trees:tree.leaf_parameters[:]=rng.normal(size=tree.leaf_parameters.shape)
            standard=export_model(m,['a','b','c'],['b','s'],mean,scale)
            raw=export_model(m,['a','b','c'],['b','s'],mean,scale,raw=True)
            a,ar,_=predict_margin(standard,x);b,br,_=predict_margin(raw,x*scale+mean)
            np.testing.assert_allclose(a,-m.decision_function(x)[:,0],atol=1e-12)
            np.testing.assert_allclose(a,b,atol=1e-12);np.testing.assert_array_equal(ar,br)
            self.assertEqual(operation_counts(standard)['stored_scalar_parameters'],sum(t.split_w.size+t.split_b.size+t.leaf_parameters.size for t in m.trees))

    def test_working_point_respects_background_budget_including_ties(self):
        m=np.array([9,8,8,7,6,9,8,7]);y=np.array([0,0,0,0,0,1,1,1])
        point=deterministic_working_point(m,y,.4)
        self.assertLessEqual(point['false_positives'],2)
        self.assertEqual(point['signal_margin_threshold'],8.)
        self.assertEqual(point['false_positives'],1)
