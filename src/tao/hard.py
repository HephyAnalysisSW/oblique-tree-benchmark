"""Hard TAO with linear/constant multiclass leaves and one forest-wide objective.

The final class has zero logit, so each leaf stores K-1 affine functions. All
losses use the full training-weight sum as denominator. L1 split penalties
include the bias, matching liblinear's penalized synthetic intercept; leaf
intercepts are unpenalized. No soft routing or quantization is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from .split_search import best_threshold
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


@dataclass
class InputTransform:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def identity(cls, features):
        return cls(np.zeros(features), np.ones(features))

    @classmethod
    def from_variances(cls, mean, variances):
        variances = np.asarray(variances, dtype=float)
        if np.any(variances < 0):
            raise ValueError('Feature variances must be nonnegative')
        scale = np.sqrt(variances)
        return cls(np.asarray(mean, dtype=float), np.where(scale == 0, 1.0, scale))

    def apply(self, x):
        x = np.asarray(x, dtype=np.float64)
        # Avoid allocating another full feature array for standardized inputs.
        if np.all(self.mean == 0) and np.all(self.scale == 1):
            return x
        return (x - self.mean) / self.scale


def cross_entropy_rows(logits, y):
    return logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]


def ce_value_gradient(theta, design, y, offset, weights, normalizer):
    """Smooth CE and analytic gradient, including the other trees' fixed logits."""
    free_logits = np.asarray(design @ theta)
    logits = np.column_stack((free_logits, np.zeros(len(y))))
    if offset is not None:
        logits += offset
    value = float(np.dot(weights, cross_entropy_rows(logits, y)) / normalizer)
    residual = softmax(logits, axis=1)
    residual[np.arange(len(y)), y] -= 1
    residual *= (weights / normalizer)[:, None]
    gradient = np.asarray(design.T @ residual[:, :-1])
    return value, gradient


def fit_leaf_parameters(design, y, offset, weights, normalizer, initial,
                        penalty_rows, alpha, max_iter, ftol=1e-10, gtol=1e-7):
    """Exact L1 parameterization via nonnegative positive/negative coefficient parts.

    L-BFGS-B sees a smooth constrained objective, not an L2 approximation to L1.
    The returned candidate is accepted only if its actual objective improves.
    """
    shape = initial.shape
    flat = initial.ravel()
    mask = np.repeat(np.asarray(penalty_rows, dtype=bool), shape[1])
    if alpha == 0:
        mask[:] = False
    penalized, free = np.flatnonzero(mask), np.flatnonzero(~mask)
    size = len(penalized)
    packed = np.concatenate((np.maximum(flat[penalized], 0),
                             np.maximum(-flat[penalized], 0), flat[free]))

    def unpack(v):
        result = np.zeros_like(flat)
        result[penalized] = v[:size] - v[size:2 * size]
        result[free] = v[2 * size:]
        return result.reshape(shape)

    def objective(v):
        theta = unpack(v)
        value, gradient = ce_value_gradient(theta, design, y, offset, weights, normalizer)
        gradient = gradient.ravel()
        value += alpha * float(v[:2 * size].sum())
        jac = np.concatenate((gradient[penalized] + alpha,
                              -gradient[penalized] + alpha, gradient[free]))
        return value, jac

    started = time.perf_counter()
    start_value = objective(packed)[0]
    result = minimize(objective, packed, method='L-BFGS-B', jac=True,
                      bounds=[(0, None)] * (2 * size) + [(None, None)] * len(free),
                      options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                               'maxls': 30})
    candidate = unpack(result.x)
    upper_value = objective(result.x)[0]
    # At a finite iteration limit, positive and negative parts can both be
    # nonzero. Canonicalizing their difference reduces L1 without changing logits.
    # Report and accept using the actual stored coefficients' objective.
    excess_l1 = alpha * float(result.x[:2 * size].sum() -
                             np.abs(candidate.ravel()[penalized]).sum())
    value = upper_value - excess_l1
    accepted = bool(np.isfinite(value) and value <= start_value)
    return (candidate if accepted else initial.copy()), {
        'initial_objective': start_value, 'final_objective': value if accepted else start_value,
        'iterations': int(result.nit), 'converged': bool(result.success),
        'optimizer_parameterized_objective': float(upper_value),
        'l1_parameterization_gap': float(excess_l1),
        'accepted': accepted, 'message': str(result.message),
        'function_evaluations': int(result.nfev), 'seconds': time.perf_counter() - started,
    }


def fit_l1_split(x, targets, effective_weights, normalizer, alpha, initial_w,
                 initial_b, seed=42, max_iter=100, solver='liblinear'):
    """Two optimizers for the same weighted binary logistic + L1 split surrogate."""
    started = time.perf_counter()
    if solver == 'liblinear':
        classifier = LogisticRegression(penalty='l1', solver='liblinear',
            C=1 / (alpha * normalizer), max_iter=max_iter, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            classifier.fit(x, targets, sample_weight=effective_weights)
        w, b = classifier.coef_[0], float(classifier.intercept_[0])
        converged = not any(issubclass(v.category, ConvergenceWarning) for v in caught)
        stats = dict(converged=converged, iterations=int(classifier.n_iter_.max()))
    elif solver == 'lbfgsb':
        design = np.column_stack((x, np.ones(len(x))))
        # Reference class 1 has zero logit; class-0 logit is minus the split margin.
        initial = -np.r_[initial_w, initial_b][:, None]
        fitted, stats = fit_leaf_parameters(design, targets, None, effective_weights,
            normalizer, initial, np.ones(design.shape[1], dtype=bool), alpha, max_iter)
        w, b = -fitted[:-1, 0], float(-fitted[-1, 0])
    else:
        raise ValueError(f'Unknown L1 split solver: {solver}')
    margin = x @ w + b
    stats.update(seconds=time.perf_counter() - started, solver=solver,
        surrogate_objective=float(np.dot(effective_weights,
            np.logaddexp(0, margin) - targets * margin) / normalizer +
            alpha * (np.abs(w).sum() + abs(b))))
    return w, b, stats


class HardTree:
    def __init__(self, depth, features, classes, leaf_mode, rng):
        self.depth, self.features, self.classes = depth, features, classes
        self.leaf_mode = leaf_mode
        self.internal = 2 ** depth - 1
        self.leaves = 2 ** depth
        self.split_w = rng.normal(size=(self.internal, features))
        self.split_b = rng.normal(size=self.internal)
        self.leaf_parameters = np.zeros((self.leaves, self.leaf_width, classes - 1))

    @property
    def leaf_width(self):
        return self.features + 1 if self.leaf_mode == 'linear' else 1

    @property
    def penalty_rows(self):
        return np.r_[np.ones(self.leaf_width - 1, dtype=bool), False]

    def design(self, x):
        if self.leaf_mode == 'linear':
            return np.column_stack((x, np.ones(len(x))))
        return np.ones((len(x), 1))

    def routes(self, x, indexed=True):
        """Indices reaching every node. Dense mode is a measured reference traversal."""
        mode = ('indexed' if indexed is True else 'dense' if indexed is False else indexed)
        if mode not in ('indexed', 'dense', 'batched'):
            raise ValueError(f'Unknown routing implementation: {mode}')
        decisions = x @ self.split_w.T + self.split_b if mode == 'batched' else None
        result = [None] * (self.internal + self.leaves)
        result[0] = np.arange(len(x))
        for node in range(self.internal):
            indices = result[node]
            if len(indices) == 0:
                result[2 * node + 1] = indices
                result[2 * node + 2] = indices
                continue
            if mode == 'indexed':
                right = x[indices] @ self.split_w[node] + self.split_b[node] > 0
            elif mode == 'batched':
                right = decisions[indices, node] > 0
            else:
                right = (x @ self.split_w[node] + self.split_b[node])[indices] > 0
            result[2 * node + 1] = indices[~right]
            result[2 * node + 2] = indices[right]
        return result

    def leaf_indices(self, x, indexed=True):
        assignment = np.empty(len(x), dtype=np.int32)
        for leaf, indices in enumerate(self.routes(x, indexed)[self.internal:]):
            assignment[indices] = leaf
        return assignment

    def predict_subtree(self, x, root=0, indexed=True):
        mode = ('indexed' if indexed is True else 'dense' if indexed is False else indexed)
        if mode not in ('indexed', 'dense', 'batched'):
            raise ValueError(f'Unknown routing implementation: {mode}')
        scores = np.zeros((len(x), self.classes))
        decisions = columns = None
        if mode == 'batched':
            internal_nodes = []
            def collect(node):
                if node < self.internal:
                    internal_nodes.append(node)
                    collect(2 * node + 1); collect(2 * node + 2)
            collect(root)
            columns = {node: column for column, node in enumerate(internal_nodes)}
            if internal_nodes:
                decisions = x @ self.split_w[internal_nodes].T + self.split_b[internal_nodes]

        def visit(node, indices):
            if len(indices) == 0:
                return
            if node >= self.internal:
                scores[indices, :-1] = self.design(x[indices]) @ self.leaf_parameters[node - self.internal]
                return
            if mode == 'indexed':
                right = x[indices] @ self.split_w[node] + self.split_b[node] > 0
            elif mode == 'batched':
                right = decisions[indices, columns[node]] > 0
            else:
                right = (x @ self.split_w[node] + self.split_b[node])[indices] > 0
            visit(2 * node + 1, indices[~right])
            visit(2 * node + 2, indices[right])

        visit(root, np.arange(len(x)))
        return scores


class HardTAOClassifier:
    def __init__(self, depth=3, trees=1, leaf_mode='linear', seed=42,
                 split_alpha=1e-5, leaf_alpha=1e-5, leaf_max_iter=100,
                 split_max_iter=100, indexed=False, split_solver='liblinear'):
        if depth < 0 or trees < 1 or leaf_mode not in ('linear', 'constant'):
            raise ValueError('Invalid forest architecture')
        if split_alpha <= 0 or leaf_alpha < 0:
            raise ValueError('Require split_alpha > 0 and leaf_alpha >= 0')
        self.depth, self.ntrees, self.leaf_mode, self.seed = depth, trees, leaf_mode, seed
        self.split_alpha, self.leaf_alpha = split_alpha, leaf_alpha
        self.leaf_max_iter, self.split_max_iter = leaf_max_iter, split_max_iter
        self.indexed = indexed
        self.split_solver = split_solver
        self.transform = None
        self.trees = []

    def initialize(self, features, classes, transform=None):
        if classes < 2:
            raise ValueError('At least two classes are required')
        self.n_features_in_, self.n_classes_ = features, classes
        self.classes_ = np.arange(classes)
        self.transform = transform or InputTransform.identity(features)
        rng = np.random.default_rng(self.seed)
        self.trees = [HardTree(self.depth, features, classes, self.leaf_mode, rng)
                      for _ in range(self.ntrees)]
        return self

    def _inputs(self, x, y=None, sample_weight=None):
        if not self.trees:
            raise RuntimeError('Initialize the classifier first')
        x = self.transform.apply(x)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError('Expected finite features with the initialized width')
        if y is None:
            return x
        y = np.asarray(y)
        if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError('Targets must be integer class indices')
        if np.any(y < 0) or np.any(y >= self.n_classes_):
            raise ValueError('Class index outside initialized range')
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('Expected finite nonnegative weights with positive sum')
        return x, y, weights

    def decision_function(self, x):
        x = self._inputs(x)
        return sum((tree.predict_subtree(x, indexed=self.indexed) for tree in self.trees),
                   np.zeros((len(x), self.n_classes_)))

    def predict_proba(self, x):
        return softmax(self.decision_function(x), axis=1).astype(np.float32)

    def predict(self, x):
        return self.decision_function(x).argmax(axis=1)

    def _penalty(self):
        split = sum(np.abs(t.split_w).sum() + np.abs(t.split_b).sum() for t in self.trees)
        leaf = sum(np.abs(t.leaf_parameters[:, :-1]).sum() for t in self.trees)
        return float(self.split_alpha * split + self.leaf_alpha * leaf)

    def objective(self, x, y, sample_weight=None):
        x, y, weights = self._inputs(x, y, sample_weight)
        logits = sum((t.predict_subtree(x, indexed=self.indexed) for t in self.trees),
                     np.zeros((len(y), self.n_classes_)))
        return float(np.dot(weights, cross_entropy_rows(logits, y)) / weights.sum() + self._penalty())

    def _joint_design(self, x):
        # Construct CSR arrays directly; no Python object per nonzero coefficient.
        blocks = []
        for tree in self.trees:
            assignment = tree.leaf_indices(x, self.indexed)
            width = tree.leaf_width
            data = tree.design(x)
            indices = assignment[:, None] * width + np.arange(width, dtype=np.int32)
            indptr = np.arange(0, (len(x) + 1) * width, width, dtype=np.int64)
            block = sparse.csr_matrix((data.ravel(), indices.ravel(), indptr),
                                      shape=(len(x), tree.leaves * width))
            blocks.append(block)
        return sparse.hstack(blocks, format='csr')

    def _global_refit(self, x, y, weights):
        start = time.perf_counter()
        design = self._joint_design(x)
        initial = np.concatenate([t.leaf_parameters.reshape(-1, self.n_classes_ - 1)
                                  for t in self.trees])
        penalty = np.concatenate([np.tile(t.penalty_rows, t.leaves) for t in self.trees])
        fitted, stats = fit_leaf_parameters(design, y, None, weights, weights.sum(), initial,
                                            penalty, self.leaf_alpha, self.leaf_max_iter,
                                            ftol=getattr(self,'leaf_ftol',1e-10),
                                            gtol=getattr(self,'leaf_gtol',1e-7))
        cursor = 0
        for tree in self.trees:
            rows = tree.leaves * tree.leaf_width
            tree.leaf_parameters = fitted[cursor:cursor + rows].reshape(tree.leaf_parameters.shape)
            cursor += rows
        stats.update(events=len(y), design_rows=design.shape[0], design_columns=design.shape[1],
                     design_nonzeros=int(design.nnz), seconds=time.perf_counter() - start)
        return stats

    def global_refit(self, x, y, sample_weight=None):
        # Public API receives the complete training partition explicitly.
        x, y, weights = self._inputs(x, y, sample_weight)
        return self._global_refit(x, y, weights)

    def _split_update(self, tree, node, x, y, weights, offset, normalizer):
        started = time.perf_counter()
        losses = [cross_entropy_rows(offset + tree.predict_subtree(x, child, self.indexed), y)
                  for child in (2 * node + 1, 2 * node + 2)]
        delta = losses[0] - losses[1]
        effective = weights * np.abs(delta)
        active = effective > 0
        targets = (delta[active] > 0).astype(int)
        stats = dict(events=len(y), active_events=int(active.sum()), tied_events=int((delta == 0).sum()),
                     fitted=False, accepted=False, converged=True,
                     losses_seconds=time.perf_counter() - started, solver_seconds=0.0)
        enhanced=getattr(self,'split_search','surrogate')=='hard_threshold'
        if enhanced:
            return self._hard_split_update(tree,node,x,y,weights,normalizer,losses,active,targets,stats)
        if len(np.unique(targets)) < 2:
            return stats
        candidate_w, candidate_b, fit_stats = fit_l1_split(
            x[active], targets, effective[active], normalizer, self.split_alpha,
            tree.split_w[node], tree.split_b[node], self.seed, self.split_max_iter,
            getattr(self, 'split_solver', 'liblinear'))
        def value(w, b):
            right = x @ w + b > 0
            ce = np.dot(weights, np.where(right, losses[1], losses[0])) / normalizer
            return float(ce + self.split_alpha * (np.abs(w).sum() + abs(b)))
        before = value(tree.split_w[node], tree.split_b[node])
        after = value(candidate_w, candidate_b)
        accepted = bool(np.isfinite(after) and after <= before)
        if accepted:
            tree.split_w[node], tree.split_b[node] = candidate_w, candidate_b
        stats.update(fitted=True, accepted=accepted, objective_before=before,
                     objective_after=after if accepted else before,
                     converged=fit_stats['converged'], iterations=fit_stats['iterations'],
                     solver_seconds=fit_stats['seconds'], surrogate=fit_stats)
        return stats

    def _hard_split_update(self,tree,node,x,y,weights,normalizer,losses,active,targets,stats):
        started=time.perf_counter()
        old_w,old_b=tree.split_w[node].copy(),float(tree.split_b[node])
        def value(w,b):
            right=x@w+b>0
            return float(np.dot(weights,np.where(right,losses[1],losses[0]))/normalizer+
                         self.split_alpha*(np.abs(w).sum()+abs(b)))
        before=value(old_w,old_b);best_value=before;best_w,best_b=old_w,old_b
        directions=[('current',old_w)]
        fit_stats=None
        if len(np.unique(targets))==2:
            effective=weights[active]*np.abs(losses[0][active]-losses[1][active])
            candidate_w,candidate_b,fit_stats=fit_l1_split(x[active],targets,effective,normalizer,
                self.split_alpha,old_w,old_b,self.seed,self.split_max_iter,
                getattr(self,'split_solver','liblinear'))
            candidate_value=value(candidate_w,candidate_b)
            if candidate_value<best_value:best_value,best_w,best_b=candidate_value,candidate_w,candidate_b
            directions.append(('logistic',candidate_w))
            for fraction in (.25,.5,.75):
                directions.append((f'blend_{fraction:g}',(1-fraction)*old_w+fraction*candidate_w))
        # Even one-sided or tied routing targets can benefit from a constant route.
        directions.append(('constant',np.zeros_like(old_w)))
        proposals=[];selected='logistic_raw' if best_value<before else 'unchanged'
        for name,w in directions:
            bias,search=best_threshold(x@w,losses[0],losses[1],weights,normalizer,self.split_alpha)
            after=value(w,bias)  # Recompute using actual hard routing before acceptance.
            proposals.append(dict(direction=name,objective=after,**search))
            if after<best_value:best_value,best_w,best_b,selected=after,w,bias,name
        accepted=bool(best_value<before)
        if accepted:tree.split_w[node],tree.split_b[node]=best_w,best_b
        stats.update(fitted=fit_stats is not None,accepted=accepted,objective_before=before,
            objective_after=best_value,converged=True if fit_stats is None else fit_stats['converged'],
            iterations=0 if fit_stats is None else fit_stats['iterations'],
            solver_seconds=0. if fit_stats is None else fit_stats['seconds'],
            surrogate=fit_stats,search='hard_threshold',selected_direction=selected,
            proposals=proposals,search_seconds=time.perf_counter()-started)
        return stats

    def fit_epoch(self, x, y, sample_weight=None, global_refit=True):
        x, y, weights = self._inputs(x, y, sample_weight)
        normalizer = weights.sum()
        profile = dict(leaf_seconds=0.0, routing_seconds=0.0, split_seconds=0.0,
                       prediction_seconds=0.0, leaf_fits=[], split_fits=[])
        start = time.perf_counter()
        cached = [t.predict_subtree(x, indexed=self.indexed) for t in self.trees]
        total = sum(cached, np.zeros((len(y), self.n_classes_)))
        profile['prediction_seconds'] += time.perf_counter() - start
        profile['objective_before'] = float(np.dot(weights, cross_entropy_rows(total, y)) / normalizer + self._penalty())
        for index, tree in enumerate(self.trees):
            offset = total - cached[index]
            tick = time.perf_counter()
            routes = tree.routes(x, self.indexed)
            profile['routing_seconds'] += time.perf_counter() - tick
            tick = time.perf_counter()
            for leaf, rows in enumerate(routes[tree.internal:]):
                if len(rows) == 0 or weights[rows].sum() == 0:
                    tree.leaf_parameters[leaf] = 0
                    continue
                fitted, stats = fit_leaf_parameters(tree.design(x[rows]), y[rows], offset[rows],
                    weights[rows], normalizer, tree.leaf_parameters[leaf], tree.penalty_rows,
                    self.leaf_alpha, self.leaf_max_iter,
                    ftol=getattr(self,'leaf_ftol',1e-10),gtol=getattr(self,'leaf_gtol',1e-7))
                tree.leaf_parameters[leaf] = fitted
                profile['leaf_fits'].append(stats)
            profile['leaf_seconds'] += time.perf_counter() - tick
            tick = time.perf_counter()
            # Cached reaching-event sets remain valid for bottom-up updates:
            # descendants change before ancestors; no changed ancestor is reused.
            for node in range(tree.internal - 1, -1, -1):
                rows = routes[node]
                if len(rows):
                    profile['split_fits'].append(self._split_update(tree, node, x[rows], y[rows],
                                                weights[rows], offset[rows], normalizer))
            profile['split_seconds'] += time.perf_counter() - tick
            tick = time.perf_counter()
            cached[index] = tree.predict_subtree(x, indexed=self.indexed)
            total = offset + cached[index]
            profile['prediction_seconds'] += time.perf_counter() - tick
        if global_refit:
            profile['global_refit'] = self._global_refit(x, y, weights)
        profile['objective_after'] = self._objective_prepared(x, y, weights)
        profile['seconds'] = time.perf_counter() - start
        return profile

    def _objective_prepared(self, x, y, weights):
        logits = sum((t.predict_subtree(x, indexed=self.indexed) for t in self.trees),
                     np.zeros((len(y), self.n_classes_)))
        return float(np.dot(weights, cross_entropy_rows(logits, y)) / weights.sum() + self._penalty())

    def complexity(self):
        return dict(trees=self.ntrees, maximum_depth=self.depth, leaf_mode=self.leaf_mode,
                    classes=self.n_classes_, logits_per_leaf=self.n_classes_ - 1,
                    internal_nodes=sum(t.internal for t in self.trees),
                    leaves=sum(t.leaves for t in self.trees),
                    split_nonzeros=sum(int(np.count_nonzero(t.split_w)) for t in self.trees),
                    leaf_nonzeros=sum(int(np.count_nonzero(t.leaf_parameters[:, :-1])) for t in self.trees),
                    split_biases=sum(t.internal for t in self.trees),
                    leaf_biases=sum(t.leaves * (self.n_classes_ - 1) for t in self.trees),
                    hard_routing=True, quantized=False)
