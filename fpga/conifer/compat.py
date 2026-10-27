"""Reconcile an upstream XGBoost conversion with its immutable native JSON.

Only binary logistic, numeric, finite-input gbtree inference is supported here.
The upstream conversion is preserved for diagnostics. This adapter never edits
an XGBoost model or any frozen handover file.
"""
import copy
import json
import math
import struct


def float32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def reconcile(native, upstream):
    learner = native['learner']
    assert learner['objective']['name'] == 'binary:logistic', 'Binary logistic only'
    assert int(learner['learner_model_param']['num_class']) == 0
    booster = learner['gradient_booster']
    assert booster['name'] == 'gbtree', 'DART and other boosters are not supported'
    model = booster['model']
    assert not any(model['tree_info']), 'Expected one binary margin'
    result = copy.deepcopy(upstream)
    assert len(result['trees']) == len(model['trees'])
    assert result['n_features'] == int(learner['learner_model_param'].get('num_feature', result['n_features']))
    corrections = {'thresholds': 0, 'leaf_values': 0, 'base_score_before': result['init_predict'],
                   'max_depth_before': result['max_depth']}
    depths = []
    for original, converted in zip(model['trees'], result['trees']):
        assert len(converted) == 1
        tree = converted[0]
        assert not any(original['split_type']), 'Categorical splits are not supported'
        n = len(original['left_children'])
        assert len(tree['feature']) == n
        if 'weight' in tree:
            for feature, weights in zip(tree['feature'], tree['weight']):
                expected = [int(j == feature) for j in range(result['n_features'])]
                assert weights == expected, 'Legacy CMSSW emulator requires axis-aligned one-hot weights'
        depth = []
        stack = [(0, 0)]
        while stack:
            node, level = stack.pop()
            left, right = original['left_children'][node], original['right_children'][node]
            assert tree['children_left'][node] == left and tree['children_right'][node] == right
            if left == -1:
                assert right == -1 and tree['feature'][node] == -2
                value = float32(original['split_conditions'][node])
                corrections['leaf_values'] += tree['value'][node] != value
                tree['value'][node] = value
                tree['threshold'][node] = 0.
                depth.append(level)
            else:
                assert tree['feature'][node] == original['split_indices'][node]
                threshold = float32(original['split_conditions'][node])
                corrections['thresholds'] += tree['threshold'][node] != threshold
                tree['threshold'][node] = threshold
                stack.extend([(left, level+1), (right, level+1)])
        depths.append(max(depth))
    base = learner['learner_model_param']['base_score']
    if isinstance(base, str):
        base = json.loads(base)
    if isinstance(base, list):
        assert len(base) == 1
        base = base[0]
    base = float32(base)
    assert 0 < base < 1
    # XGBoost's binary logistic intercept is logit(base_score), not a probability.
    result['init_predict'] = [math.log(base)-math.log1p(-base)]
    result['max_depth'] = max(depths)
    result['splitting_convention'] = '<'  # native XGBoost sends equality right.
    corrections.update(base_score_after=result['init_predict'], max_depth_after=result['max_depth'],
                       source_trees=len(depths), input_contract='finite inputs rounded to binary32, matching XGBoost DMatrix')
    return result, corrections
