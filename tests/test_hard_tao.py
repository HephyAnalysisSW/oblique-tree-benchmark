"""Correctness checks for the hard-tree fixes, independent of jet test data."""
import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import softmax

from tao.hard import (HardTAOClassifier, InputTransform, ce_value_gradient,
                      cross_entropy_rows, fit_leaf_parameters, fit_l1_split)


class HardTAOTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_variance_to_scale_and_zero_variance(self):
        transform = InputTransform.from_variances([2, 5], [4, 0])
        np.testing.assert_array_equal(transform.apply([[4, 7]]), [[1, 2]])
        with self.assertRaises(ValueError):
            InputTransform.from_variances([0], [-1])

    def test_normalization_applied_exactly_once(self):
        raw = self.rng.normal(size=(100, 3)) * [2, 3, 4] + [4, 5, 6]
        transform = InputTransform.from_variances([4, 5, 6], [4, 9, 16])
        normalized = transform.apply(raw)
        model = HardTAOClassifier(depth=2).initialize(3, 3)
        model.trees[0].leaf_parameters[:] = self.rng.normal(size=model.trees[0].leaf_parameters.shape)
        raw_model = copy.deepcopy(model)
        raw_model.transform = transform
        np.testing.assert_allclose(model.decision_function(normalized), raw_model.decision_function(raw), atol=1e-13)

    def test_full_training_global_refit_recovers_all_class_frequencies(self):
        x = np.zeros((1000, 1))
        y = np.repeat([0, 1, 2], [700, 200, 100])
        full = HardTAOClassifier(depth=0, leaf_mode='constant', leaf_alpha=0,
                                 leaf_max_iter=200).initialize(1, 3)
        stats = full.global_refit(x, y)
        np.testing.assert_allclose(full.predict_proba(x[:1])[0], [.7, .2, .1], atol=1e-5)
        self.assertEqual(stats['events'], 1000)
        last = copy.deepcopy(full)
        last.global_refit(x[-100:], y[-100:])
        full_loss = cross_entropy_rows(full.decision_function(x), y).mean()
        last_loss = cross_entropy_rows(last.decision_function(x), y).mean()
        self.assertGreater(last_loss, full_loss + 1)

    def test_joint_design_preserves_linear_and_constant_leaf_types(self):
        x = self.rng.normal(size=(100, 4))
        y = self.rng.integers(0, 3, size=100)
        for mode, width in [('constant', 1), ('linear', 5)]:
            model = HardTAOClassifier(depth=2, trees=2, leaf_mode=mode).initialize(4, 3)
            stats = model.global_refit(x, y)
            self.assertEqual(stats['design_columns'], 2 * 4 * width)
            self.assertEqual(stats['design_nonzeros'], 100 * 2 * width)
            design = model._joint_design(x)
            theta = np.concatenate([t.leaf_parameters.reshape(-1, 2) for t in model.trees])
            np.testing.assert_allclose((design @ theta), model.decision_function(x)[:, :-1], atol=1e-12)

    def test_multiclass_cross_entropy_gradient_with_other_tree_offset(self):
        x = np.column_stack([self.rng.normal(size=(25, 3)), np.ones(25)])
        y = self.rng.integers(0, 5, size=25)
        theta = self.rng.normal(size=(4, 4)) * .2
        offset = self.rng.normal(size=(25, 5))
        weights = self.rng.uniform(.1, 2, size=25)
        value, analytic = ce_value_gradient(theta, x, y, offset, weights, 100)
        numerical = np.zeros_like(theta)
        for coordinate in np.ndindex(theta.shape):
            left, right = theta.copy(), theta.copy()
            left[coordinate] -= 1e-6
            right[coordinate] += 1e-6
            numerical[coordinate] = (ce_value_gradient(right, x, y, offset, weights, 100)[0] -
                                     ce_value_gradient(left, x, y, offset, weights, 100)[0]) / 2e-6
        np.testing.assert_allclose(analytic, numerical, atol=1e-9, rtol=1e-6)
        self.assertTrue(np.isfinite(value))

    def test_tied_routing_losses_do_not_fit_a_third_class(self):
        x = self.rng.normal(size=(80, 3))
        y = self.rng.integers(0, 3, size=80)
        model = HardTAOClassifier(depth=1).initialize(3, 3)
        with patch('tao.hard.LogisticRegression.fit', side_effect=AssertionError('Ties must not be fitted')):
            stats = model._split_update(model.trees[0], 0, x, y, np.ones(80), np.zeros((80, 3)), 80)
        self.assertEqual(stats['active_events'], 0)
        self.assertEqual(stats['tied_events'], 80)

    def test_dense_and_indexed_routing_agree_including_boundary(self):
        x = self.rng.normal(size=(500, 5))
        x[0] = 0
        model = HardTAOClassifier(depth=4).initialize(5, 3)
        tree = model.trees[0]
        tree.split_b[0] = 0
        tree.leaf_parameters[:] = self.rng.normal(size=tree.leaf_parameters.shape)
        np.testing.assert_array_equal(tree.leaf_indices(x, True), tree.leaf_indices(x, False))
        np.testing.assert_array_equal(tree.leaf_indices(x, True), tree.leaf_indices(x, 'batched'))
        np.testing.assert_allclose(tree.predict_subtree(x, indexed=True),
                                   tree.predict_subtree(x, indexed='batched'), atol=1e-12)
        # Counterfactual subtree predictions used during split fitting also agree.
        for child in (1, 2):
            np.testing.assert_allclose(tree.predict_subtree(x, child, True),
                                       tree.predict_subtree(x, child, 'batched'), atol=1e-12)
        self.assertIn(0, tree.routes(x)[1])  # zero decision goes left
        np.testing.assert_allclose(tree.predict_subtree(x, indexed=True),
                                   tree.predict_subtree(x, indexed=False), atol=1e-12)

    def test_mixed_ties_are_removed_before_binary_routing_fit(self):
        x = np.zeros((80, 2)); x[:40, 1] = -1; x[40:, 1] = 1
        y = np.tile([0, 1], 40)
        model = HardTAOClassifier(depth=2, leaf_mode='constant').initialize(2, 2)
        tree = model.trees[0]
        tree.split_w[1:] = [0, 1]; tree.split_b[1:] = 0
        tree.leaf_parameters[:, 0, 0] = [0, 2, 0, -2]

        def capture(features, targets, weights, normalizer, alpha, w, b, *args):
            self.assertEqual(len(targets), 40)
            np.testing.assert_array_equal(np.unique(targets), [0, 1])
            self.assertTrue(np.all(weights > 0))
            return w.copy(), b, dict(converged=True, iterations=0, seconds=0)

        with patch('tao.hard.fit_l1_split', side_effect=capture) as fit:
            stats = model._split_update(tree, 0, x, y, np.ones(80), np.zeros((80, 2)), 80)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(stats['tied_events'], 40)

    def test_split_candidate_that_increases_true_loss_is_rejected(self):
        x = np.r_[np.linspace(-2, -.1, 40), np.linspace(.1, 2, 40)][:, None]
        y = (x[:, 0] > 0).astype(int)
        model = HardTAOClassifier(depth=1, leaf_mode='constant').initialize(1, 2)
        tree = model.trees[0]
        tree.split_w[0] = 1; tree.split_b[0] = 0
        tree.leaf_parameters[:, 0, 0] = [3, -3]
        candidate = (np.array([-1.]), 0., dict(converged=True, iterations=1, seconds=0))
        with patch('tao.hard.fit_l1_split', return_value=candidate):
            stats = model._split_update(tree, 0, x, y, np.ones(80), np.zeros((80, 2)), 80)
        self.assertFalse(stats['accepted'])
        np.testing.assert_array_equal(tree.split_w[0], [1])

    def test_epoch_decreases_same_objective_binary_and_multiclass(self):
        x = self.rng.normal(size=(500, 3))
        for classes in [2, 3, 5]:
            logits = np.column_stack((x[:, :min(classes - 1, 3)], np.zeros(500)))
            # Generate a separate well-defined K-class problem for each check.
            w = self.rng.normal(size=(3, classes))
            y = (x @ w + self.rng.normal(size=(500, classes)) * .3).argmax(axis=1)
            model = HardTAOClassifier(depth=1, trees=2, leaf_max_iter=80).initialize(3, classes)
            before = model.objective(x, y)
            profile = model.fit_epoch(x, y)
            after = model.objective(x, y)
            self.assertLess(after, before - .1)
            self.assertLessEqual(profile['objective_after'], profile['objective_before'] + 1e-10)
            self.assertEqual(profile['global_refit']['events'], len(y))
            probabilities = model.predict_proba(x)
            self.assertEqual(probabilities.shape, (500, classes))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
            np.testing.assert_array_equal(probabilities, pickle.loads(pickle.dumps(model)).predict_proba(x))

    def test_limited_leaf_fit_reports_actual_stored_coefficient_objective(self):
        x = np.column_stack((self.rng.normal(size=(200, 3)), np.ones(200)))
        y = self.rng.integers(0, 3, size=200)
        offset = self.rng.normal(size=(200, 3))
        initial = self.rng.normal(size=(4, 2))
        fitted, stats = fit_leaf_parameters(x, y, offset, np.ones(200), 200,
                                           initial, [True, True, True, False], .03, 2)
        actual, _ = ce_value_gradient(fitted, x, y, offset, np.ones(200), 200)
        actual += .03 * np.abs(fitted[:-1]).sum()
        self.assertAlmostEqual(stats['final_objective'], actual, places=12)

    def test_split_optimizers_minimize_same_l1_objective(self):
        x = self.rng.normal(size=(1000, 4))
        probability = 1 / (1 + np.exp(-(x[:, 0] - .5 * x[:, 1] + .2)))
        y = (self.rng.uniform(size=1000) < probability).astype(int)
        weight = self.rng.uniform(.1, 2, size=1000)
        outputs = []
        for solver in ['liblinear', 'lbfgsb']:
            w, b, stats = fit_l1_split(x, y, weight, 1500, .002,
                np.zeros(4), 0, max_iter=1000, solver=solver)
            margin = x @ w + b
            actual = np.dot(weight, np.logaddexp(0, margin) - y * margin) / 1500
            actual += .002 * (np.abs(w).sum() + abs(b))
            self.assertAlmostEqual(actual, stats['surrogate_objective'], places=12)
            self.assertTrue(stats['converged'])
            outputs.append(stats['surrogate_objective'])
        self.assertLess(abs(outputs[0] - outputs[1]), 1e-6)

    def test_global_refit_uses_same_objective_with_l1(self):
        x = self.rng.normal(size=(300, 3))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        model = HardTAOClassifier(depth=2, trees=2, leaf_alpha=.001).initialize(3, 2)
        before = model.objective(x, y)
        stats = model.global_refit(x, y)
        self.assertLess(model.objective(x, y), before)
        self.assertLessEqual(stats['final_objective'], stats['initial_objective'])


if __name__ == '__main__':
    unittest.main()
