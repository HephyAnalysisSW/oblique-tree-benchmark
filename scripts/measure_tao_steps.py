"""Ordered correctness demonstrations and same-model CPU microbenchmarks."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
import numpy as np
import scipy
import sklearn
from scipy import sparse
from threadpoolctl import threadpool_limits, threadpool_info
from data_utils import resolve, write_json
from tao.hard import HardTAOClassifier, InputTransform, ce_value_gradient, cross_entropy_rows

OUT = 'results/tao_steps'
report = dict(seed=42, host=platform.node(), platform=platform.platform(),
              numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
              source_sha256=hashlib.sha256(resolve('src/tao/hard.py').read_bytes()).hexdigest(),
              note='Steps 1-5 are isolated synthetic correctness checks. Only step 6 compares identical mathematical computations for speed.',
              steps={})
rng = np.random.default_rng(42)


def record(name, values):
    report['steps'][name] = values
    write_json(f'{OUT}/measurements.json', report)
    print(json.dumps({name: values}), flush=True)


def timed(function, repeats=5):
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter(); function(); samples.append(time.perf_counter() - start)
    return dict(median_seconds=statistics.median(samples), samples_seconds=samples)


def peak_memory(function):
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


with threadpool_limits(1):
    report['threadpools'] = threadpool_info()
    x = rng.normal(size=(12000, 8))
    weights = rng.normal(size=(8, 5))
    y = (x @ weights + rng.normal(size=(len(x), 5)) * .8).argmax(axis=1)
    result = {}
    for mode in ['constant', 'linear']:
        model = HardTAOClassifier(depth=0, leaf_mode=mode, leaf_max_iter=150).initialize(8, 5)
        stats = model.global_refit(x, y)
        result[mode] = dict(cross_entropy=float(cross_entropy_rows(model.decision_function(x), y).mean()),
                            accuracy=float((model.predict(x) == y).mean()), refit=stats,
                            complexity=model.complexity())
    assert result['linear']['cross_entropy'] < result['constant']['cross_entropy']
    record('1_linear_leaves_are_preserved', result)

    xx = np.zeros((1200, 1)); yy = np.repeat([0, 1, 2], [840, 240, 120])
    full = HardTAOClassifier(depth=0, leaf_mode='constant', leaf_alpha=0, leaf_max_iter=200).initialize(1, 3)
    last = HardTAOClassifier(depth=0, leaf_mode='constant', leaf_alpha=0, leaf_max_iter=200).initialize(1, 3)
    full_stats = full.global_refit(xx, yy)
    last_stats = last.global_refit(xx[-120:], yy[-120:])
    record('2_full_partition_refit', dict(
        full_events=full_stats['events'], last_batch_events=last_stats['events'],
        target_class_fractions=[.7, .2, .1], full_probabilities=full.predict_proba(xx[:1])[0].tolist(),
        last_batch_probabilities=last.predict_proba(xx[:1])[0].tolist(),
        full_cross_entropy=float(cross_entropy_rows(full.decision_function(xx), yy).mean()),
        last_batch_cross_entropy=float(cross_entropy_rows(last.decision_function(xx), yy).mean()),
        full_seconds=full_stats['seconds'], last_batch_seconds=last_stats['seconds']))

    design = np.column_stack((rng.normal(size=(40, 4)), np.ones(40)))
    theta = rng.normal(size=(5, 4)) * .2
    labels = rng.integers(0, 5, 40); offset = rng.normal(size=(40, 5)); ww = rng.uniform(.5, 2, 40)
    _, analytical = ce_value_gradient(theta, design, labels, offset, ww, 200)
    numerical = np.zeros_like(theta)
    for index in np.ndindex(theta.shape):
        plus, minus = theta.copy(), theta.copy(); plus[index] += 1e-6; minus[index] -= 1e-6
        numerical[index] = (ce_value_gradient(plus, design, labels, offset, ww, 200)[0] -
                            ce_value_gradient(minus, design, labels, offset, ww, 200)[0]) / 2e-6
    model = HardTAOClassifier(depth=2, trees=2).initialize(8, 5)
    profile = model.fit_epoch(x[:3000], y[:3000])
    assert profile['objective_after'] <= profile['objective_before'] + 1e-10
    record('3_multiclass_forest_objective', dict(classes=5, trees=2, depth=2, training_events=3000,
        finite_difference_max_gradient_error=float(np.abs(analytical - numerical).max()),
        probability_row_sum_error=float(np.abs(model.predict_proba(x[:3000]).sum(axis=1) - 1).max()),
        objective_before=profile['objective_before'], objective_after=profile['objective_after'],
        profile=profile))

    model = HardTAOClassifier(depth=1).initialize(8, 5)
    ties = model._split_update(model.trees[0], 0, x[:1200], y[:1200], np.ones(1200), np.zeros((1200, 5)), 1200)
    assert not ties['fitted'] and ties['tied_events'] == 1200
    record('4_uninformative_routing_ties', ties)

    raw = x * np.arange(2, 10) + np.arange(8)
    mean, variance = raw.mean(axis=0), raw.var(axis=0)
    expected = (raw - mean) / np.sqrt(variance)
    wrong = (raw - mean) / variance
    corrected = InputTransform.from_variances(mean, variance).apply(raw)
    record('5_normalization', dict(
        old_max_unit_std_error=float(np.abs(wrong.std(axis=0) - 1).max()),
        fixed_max_unit_std_error=float(np.abs(corrected.std(axis=0) - 1).max()),
        fixed_max_reference_difference=float(np.abs(expected - corrected).max())))

    xx = rng.normal(size=(40000, 16))
    timings = []
    for depth in [2, 3, 4, 5]:
        tree = HardTAOClassifier(depth=depth).initialize(16, 5).trees[0]
        tree.leaf_parameters[:] = rng.normal(size=tree.leaf_parameters.shape)
        dense = tree.predict_subtree(xx, indexed=False)
        indexed = tree.predict_subtree(xx, indexed=True)
        error = float(np.abs(dense - indexed).max())
        assert error < 1e-10
        dense_time = timed(lambda: tree.predict_subtree(xx, indexed=False))
        indexed_time = timed(lambda: tree.predict_subtree(xx, indexed=True))
        timings.append(dict(depth=depth, events=len(xx), dense=dense_time, indexed=indexed_time,
                            max_logit_difference=error,
                            speedup=dense_time['median_seconds'] / indexed_time['median_seconds']))
    model = HardTAOClassifier(depth=3, trees=4).initialize(16, 5)
    small = xx[:10000]
    assignments = [t.leaf_indices(small) for t in model.trees]
    # Old-style Python triplet lists, with column order matched to the new design.
    def python_triplets():
        rows, columns, values = [], [], []
        offset_col = 0
        for tree, assignment in zip(model.trees, assignments):
            data = tree.design(small)
            for leaf in range(tree.leaves):
                indices = np.flatnonzero(assignment == leaf)
                for feature in range(tree.leaf_width):
                    rows.extend(indices.tolist())
                    columns.extend([offset_col + leaf * tree.leaf_width + feature] * len(indices))
                    values.extend(data[indices, feature].tolist())
            offset_col += tree.leaves * tree.leaf_width
        return sparse.csr_matrix((values, (rows, columns)), shape=(len(small), offset_col))
    # Include routing in both timings: the old construction above reuses a frozen
    # assignment only for validation, so benchmark its complete wrapper separately.
    def old_complete():
        for i, tree in enumerate(model.trees):
            assignments[i] = tree.leaf_indices(small)
        return python_triplets()
    reference = old_complete(); candidate = model._joint_design(small)
    assert (reference != candidate).nnz == 0
    old_time = timed(old_complete, repeats=3)
    new_time = timed(lambda: model._joint_design(small), repeats=3)
    construction = dict(events=len(small), trees=4, depth=3, features=16,
        shape=list(candidate.shape), nonzeros=int(candidate.nnz), matrix_difference_nonzeros=0,
        python_triplets=old_time, direct_csr=new_time,
        speedup=old_time['median_seconds'] / new_time['median_seconds'],
        python_triplets_peak_traced_bytes=peak_memory(old_complete),
        direct_csr_peak_traced_bytes=peak_memory(lambda: model._joint_design(small)),
        memory_note='tracemalloc peak within each construction, not process RSS')
    record('6_same_model_performance', dict(routing=timings, joint_design=construction))

# Import only now so timed sections contain no publishing or plotting overhead.
import common.syncer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
steps = report['steps']
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
linear = steps['1_linear_leaves_are_preserved']
axes[0].bar(['Constant', 'Linear'], [linear[m]['cross_entropy'] for m in ['constant', 'linear']])
axes[0].set_title('1: leaf model (synthetic, 5 classes)'); axes[0].set_ylabel('Training cross-entropy')
refit = steps['2_full_partition_refit']
axes[1].bar(['Last batch', 'Full training'], [refit['last_batch_cross_entropy'], refit['full_cross_entropy']])
axes[1].set_title('2: last-batch bug'); axes[1].set_ylabel('Cross-entropy on full training mixture')
scale = steps['5_normalization']
axes[2].bar(['Divide by variance', 'Divide by std'], [scale['old_max_unit_std_error'], scale['fixed_max_unit_std_error']])
axes[2].set_title('5: normalization bug'); axes[2].set_ylabel('Maximum deviation from unit std')
for ax in axes: ax.grid(axis='y', alpha=.2)
fig.suptitle('Hard TAO correctness checks | controlled synthetic examples')
fig.tight_layout(); fig.savefig(resolve(f'{OUT}/correctness.png'), dpi=170); plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
routing = steps['6_same_model_performance']['routing']
for mode, label in [('dense', 'Full-array traversal'), ('indexed', 'Reaching-events traversal')]:
    axes[0].plot([r['depth'] for r in routing], [r[mode]['median_seconds'] * 1000 for r in routing], 'o-', label=label)
axes[0].set_xlabel('Tree depth'); axes[0].set_ylabel('Prediction time [ms]'); axes[0].set_title('40,000 events, 16 features'); axes[0].legend(fontsize=8)
build = steps['6_same_model_performance']['joint_design']
axes[1].bar(['Python lists', 'Direct CSR'], [build[k]['median_seconds'] * 1000 for k in ['python_triplets', 'direct_csr']])
axes[1].set_ylabel('Construction time [ms]'); axes[1].set_title('Same joint leaf design')
axes[2].bar(['Python lists', 'Direct CSR'], [build[k] / 2**20 for k in ['python_triplets_peak_traced_bytes', 'direct_csr_peak_traced_bytes']])
axes[2].set_ylabel('Peak traced allocation [MiB]'); axes[2].set_title('Construction allocation')
for ax in axes: ax.grid(alpha=.2)
fig.suptitle(f'Identical-model measurements | one CPU thread | {report["host"]}')
fig.tight_layout(); fig.savefig(resolve(f'{OUT}/performance.png'), dpi=170); plt.close(fig)
print('MEASUREMENTS COMPLETE', flush=True)
