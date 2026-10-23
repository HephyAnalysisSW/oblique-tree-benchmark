# Legacy scalar prototype retained for historical comparison.
# Current training uses tao.hard.HardTAOClassifier.
"""Benchmark adapter for GOLLUM's hard TAO additive forest."""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from .tree import Tree
from .forest import Forest


class TAOClassifier:
    def __init__(self, depth=3, trees=1, seed=42, alpha=0.01, split_solver="liblinear_l1"):
        self.depth, self.ntrees, self.seed, self.alpha = depth, trees, seed, alpha
        self.split_solver = split_solver
        self.forest = None
        self.calibration_scale = 1.0
        self.calibration_offset = 0.0

    def initialize(self, features):
        config = dict(max_depth=self.depth, input_dim=features, ntrees=self.ntrees,
                      mode='regression', order='reverse_bfs', quantization=[],
                      rng_seed=self.seed, rng=np.random.default_rng(self.seed),
                      split_solver=self.split_solver)
        self.forest = Forest([Tree(config) for _ in range(self.ntrees)], config)
        # The exported data is already standardized with the fixed training scaler.
        # Both legacy layers must act as identities to avoid double standardization.
        self.forest.set_standardization(np.zeros(features), np.ones(features))
        for tree in self.forest.trees:
            tree.set_standardization(np.zeros(features), np.ones(features))
        return self

    def fit_epoch(self, x, y):
        self.forest.train_step(x, y, np.ones(len(y)),
                               {'alpha': self.alpha, 'alpha_leaf': 0.0})

    def decision_function(self, x):
        return self.forest.predict(x)

    def fit_calibration(self, x, y):
        # Training-only monotone sigmoid calibration; no validation/test labels fit it.
        scores = self.decision_function(x).reshape(-1, 1)
        calibrator = LogisticRegression(C=1e6, max_iter=200, random_state=self.seed)
        calibrator.fit(scores, y)
        self.calibration_scale = float(calibrator.coef_[0, 0])
        self.calibration_offset = float(calibrator.intercept_[0])
        if self.calibration_scale <= 0:
            raise RuntimeError('Unexpected reversed score orientation')
        return self

    def predict_proba(self, x):
        signal = expit(self.calibration_scale * self.decision_function(x) + self.calibration_offset)
        return np.column_stack([1 - signal, signal]).astype(np.float32)

    def complexity(self):
        internal, leaves, nonzero, maximum = 0, 0, 0, 0
        path_terms = []
        for tree in self.forest.trees:
            def visit(node, terms):
                nonlocal internal, leaves, nonzero, maximum
                if node.is_leaf:
                    leaves += 1
                    maximum = max(maximum, node.depth)
                    path_terms.append(terms)
                else:
                    internal += 1
                    count = int(np.count_nonzero(node.W))
                    nonzero += count
                    visit(node.left, terms + count)
                    visit(node.right, terms + count)
            visit(tree.root, 0)
        return dict(trees=self.ntrees, maximum_depth=maximum, internal_nodes=internal,
                    scalar_leaves=leaves, split_nonzero_coefficients=nonzero,
                    stored_scalars=nonzero + internal + leaves,
                    max_nonzero_terms_on_any_tree_path=max(path_terms),
                    calibration_scalars=2, quantized=False)
