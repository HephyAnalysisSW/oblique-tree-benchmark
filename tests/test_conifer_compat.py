"""Conversion invariants that can fail silently across XGBoost/conifer versions."""
import copy
import math
import unittest
from fpga.conifer.compat import float32, reconcile


def fixture():
    native = {'learner': {'objective': {'name': 'binary:logistic'},
        'learner_model_param': {'num_class': '0', 'base_score': '[0.8]'},
        'gradient_booster': {'name': 'gbtree', 'model': {'tree_info':[0], 'trees':[
            {'left_children':[1,-1,-1], 'right_children':[2,-1,-1],
             'split_type':[0,0,0], 'split_indices':[0,0,0],
             'split_conditions':[1.0000001192092896, -.123456789, .345678912]}
        ]}}}}
    exported = {'max_depth':6, 'n_trees':1, 'n_classes':2, 'n_features':1,
        'norm':1, 'init_predict':[0.], 'splitting_convention':'<=', 'trees':[[{
            'feature':[0,-2,-2], 'children_left':[1,-1,-1], 'children_right':[2,-1,-1],
            'threshold':[1.0000001,0,0], 'value':[99,-.12345679,.34567891],
            'weight':[[1],[0],[0]]}]]}
    return native, exported


class ConiferCompatibilityTest(unittest.TestCase):
    def test_native_values_intercept_topology_and_no_mutation(self):
        native, upstream = fixture(); before = copy.deepcopy((native,upstream))
        out, report = reconcile(native, upstream)
        self.assertEqual((native,upstream), before)
        self.assertEqual(out['max_depth'],1)
        self.assertEqual(out['splitting_convention'],'<')
        self.assertAlmostEqual(out['init_predict'][0],math.log(float32(.8)/(1-float32(.8))))
        self.assertEqual(out['trees'][0][0]['threshold'][0],float32(1.0000001192092896))
        self.assertEqual(out['trees'][0][0]['value'][1],float32(-.123456789))
        self.assertEqual(report['thresholds'],1)
        self.assertEqual(report['leaf_values'],2)

    def test_equality_and_binary32_input_contract(self):
        native, upstream = fixture(); out,_ = reconcile(native,upstream)
        threshold = out['trees'][0][0]['threshold'][0]
        self.assertFalse(float32(threshold)<threshold)  # equality must go right
        self.assertTrue(float32(1.)<threshold)
        # A double input slightly below the threshold rounds to equality in XGBoost.
        self.assertFalse(float32(threshold-1e-10)<threshold)
        self.assertTrue(threshold-1e-10<threshold)

    def test_legacy_scalar_base_score(self):
        native, upstream = fixture()
        native['learner']['learner_model_param']['base_score']='0.5'
        self.assertEqual(reconcile(native,upstream)[0]['init_predict'],[0.])

    def test_unsupported_categorical_and_objective_are_rejected(self):
        native, upstream = fixture()
        native['learner']['gradient_booster']['model']['trees'][0]['split_type'][0]=1
        with self.assertRaises(AssertionError): reconcile(native,upstream)
        native, upstream = fixture();native['learner']['objective']['name']='multi:softprob'
        with self.assertRaises(AssertionError): reconcile(native,upstream)

    def test_oblique_weights_are_rejected_by_legacy_bridge(self):
        native, upstream = fixture();upstream['trees'][0][0]['weight'][0]=[.5]
        with self.assertRaises(AssertionError):reconcile(native,upstream)

    def test_changed_topology_is_rejected(self):
        native, upstream = fixture();upstream['trees'][0][0]['children_right'][0]=1
        with self.assertRaises(AssertionError):reconcile(native,upstream)


if __name__ == '__main__':unittest.main()
