import unittest
import numpy as np
from tao.hard import HardTAOClassifier
from tao.portable import export_model,restore_model
from tao.sparsity import sparsity_profile


class SparsityTests(unittest.TestCase):
    def test_portable_warm_start_preserves_exact_coefficients_and_predictions(self):
        rng=np.random.default_rng(81);x=rng.normal(size=(80,3))
        for mode in ['linear','constant']:
            source=HardTAOClassifier(depth=2,trees=3,leaf_mode=mode).initialize(3,2)
            for tree in source.trees:tree.leaf_parameters[:]=rng.normal(size=tree.leaf_parameters.shape)
            spec=export_model(source,['a','b','c'],['b','s'],np.zeros(3),np.ones(3))
            restored=restore_model(spec,split_alpha=1e-3,leaf_alpha=3e-4)
            np.testing.assert_array_equal(source.decision_function(x),restored.decision_function(x))
            self.assertEqual(restored.split_alpha,1e-3);self.assertEqual(restored.leaf_alpha,3e-4)
            raw=export_model(source,['a','b','c'],['b','s'],np.zeros(3),np.ones(3),raw=True)
            with self.assertRaises(ValueError):restore_model(raw)

    def test_constant_routes_ignore_unreachable_branches_and_keep_tiny_nonzeros(self):
        model=HardTAOClassifier(depth=2,trees=1).initialize(2,2);tree=model.trees[0]
        tree.split_w[:]=0;tree.split_b[:]=0;tree.leaf_parameters[:]=0
        tree.split_b[0]=1  # always route right; left subtree cannot be reached
        tree.split_w[1]=[3,4];tree.leaf_parameters[:2]=7
        tree.leaf_parameters[2,0,0]=1e-20;tree.leaf_parameters[2,-1,0]=2
        profile=sparsity_profile(model)
        self.assertEqual(profile['dense_parameter_slots'],21)
        self.assertEqual(profile['nonzero_coefficients'],11)
        self.assertEqual(profile['reachable_nonzero_coefficients'],3)
        self.assertEqual(profile['structurally_reachable_leaves'],1)
        self.assertEqual(profile['selected_path_multiplications_upper_bound'],1)
        self.assertEqual(profile['selected_path_comparisons_upper_bound'],0)
