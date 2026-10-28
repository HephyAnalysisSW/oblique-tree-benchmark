"""Export invariants for constant/linear leaf competition; no training changes."""
import copy
import math
import unittest
import numpy as np
from fpga.leaf_competition.models import arithmetic, normalized, conifer_constant
from tao.portable import predict_margin


def fixture(mode='constant'):
    return dict(schema='tree-benchmark.binary-hard-oblique.v1',feature_count=2,features=['a','b'],
        tree_count=1,depth=1,internal_count=1,leaf_count=2,leaf_mode=mode,
        trees=[dict(split_weights=[[2.**-40,0.]],split_bias=[-2.**-40],
            leaf_parameters=[[1.],[3.]] if mode=='constant' else [[.5,0.,1.],[0.,2.,3.]])])


class LeafCompetitionTests(unittest.TestCase):
    def test_positive_scaling_preserves_both_sides_and_equality_without_mutation(self):
        spec=fixture();before=copy.deepcopy(spec);scaled=normalized(spec)
        x=np.array([[math.nextafter(1.,0.),2.],[1.,2.],[math.nextafter(1.,2.),2.]])
        a=predict_margin(spec,x);b=predict_margin(scaled,x)
        self.assertEqual(spec,before)
        np.testing.assert_array_equal(a[0],[-1.,-1.,-3.])
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        self.assertGreater(max(abs(v) for v in scaled['trees'][0]['split_weights'][0]),.1)

    def test_subnormal_bias_is_never_flushed_by_scale_transform(self):
        spec=fixture();spec['trees'][0].update(split_weights=[[4.,0.]],split_bias=[-math.ulp(0.)])
        scaled=normalized(spec)
        self.assertEqual(scaled['trees'][0]['split_bias'][0],-math.ulp(0.))
        np.testing.assert_array_equal(predict_margin(spec,[[0.,0.]])[0],predict_margin(scaled,[[0.,0.]])[0])

    def test_constant_bias_only_cut_preserves_its_sign(self):
        spec=fixture();spec['trees'][0].update(split_weights=[[0.,0.]],split_bias=[math.ulp(0.)])
        self.assertEqual(predict_margin(normalized(spec),[[0.,0.]])[0][0],-3.)
        counts=arithmetic(spec)
        self.assertEqual(counts['sparse_path_comparisons_bound'],0)
        self.assertEqual(counts['reachable_nonzero_coefficients'],2)

    def test_conifer_adapter_uses_left_equality_and_negative_leaf_contributions(self):
        converted=conifer_constant(fixture());tree=converted['trees'][0][0]
        self.assertEqual(converted['splitting_convention'],'<=')
        self.assertEqual(tree['value'],[0.,-1.,-3.])
        self.assertEqual(tree['weight'][0][0],tree['threshold'][0])
        self.assertEqual(tree['children_left'],[1,-1,-1])
        with self.assertRaises(ValueError):conifer_constant(fixture('linear'))

    def test_sparse_arithmetic_has_distinct_path_and_all_node_counts(self):
        constant=arithmetic(fixture());linear=arithmetic(fixture('linear'))
        self.assertEqual(constant['dense_slots'],5)
        self.assertEqual(constant['nonzero_coefficients'],4)
        self.assertEqual(constant['sparse_path_multiplications_bound'],1)
        self.assertEqual(linear['dense_slots'],9)
        self.assertEqual(linear['nonzero_coefficients'],6)
        self.assertEqual(linear['sparse_path_multiplications_bound'],2)
        self.assertEqual(linear['reachable_all_node_multiplications'],3)
        self.assertEqual(linear['coefficient_value_bits_binary64'],384)

if __name__=='__main__':unittest.main()
