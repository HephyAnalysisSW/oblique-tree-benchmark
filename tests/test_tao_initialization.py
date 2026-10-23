import unittest
import numpy as np
from tao.hard import HardTAOClassifier
from tao.initialization import initialize_cart,append_zero_tree


class InitializationTests(unittest.TestCase):
    def test_cart_initialization_uses_class_order_and_pads_early_leaves(self):
        x=np.linspace(-2,2,120)[:,None];y=np.where(x[:,0]>0,2,0)
        for mode in ['linear','constant']:
            m=HardTAOClassifier(depth=3,leaf_mode=mode).initialize(1,3)
            records=initialize_cart(m,x,y)
            self.assertEqual(records[0]['cart_nodes'],3)
            np.testing.assert_array_equal(m.predict(x),y)
            self.assertTrue(np.isfinite(m.predict_proba(x)).all())
            self.assertEqual(len(np.unique(m.trees[0].leaf_indices(x))),2)

    def test_cart_forest_initialization_is_reproducible(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(300,3));y=(x[:,0]+x[:,1]>0).astype(int)
        a=HardTAOClassifier(depth=2,trees=3).initialize(3,2)
        b=HardTAOClassifier(depth=2,trees=3).initialize(3,2)
        initialize_cart(a,x,y);initialize_cart(b,x,y)
        np.testing.assert_array_equal(a.decision_function(x),b.decision_function(x))
        self.assertLess(a.objective(x,y),.7)

    def test_sequential_growth_preserves_logits_and_parameter_types(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(60,3))
        for mode in ['linear','constant']:
            for classes in [2,4]:
                m=HardTAOClassifier(depth=2,leaf_mode=mode).initialize(3,classes)
                m.trees[0].leaf_parameters[:]=rng.normal(size=m.trees[0].leaf_parameters.shape)
                before=m.decision_function(x).copy();append_zero_tree(m,2,42)
                np.testing.assert_array_equal(m.decision_function(x),before)
                self.assertEqual(m.ntrees,2);self.assertEqual(m.trees[-1].leaf_mode,mode)
                with self.assertRaises(ValueError):append_zero_tree(m,4,42)
