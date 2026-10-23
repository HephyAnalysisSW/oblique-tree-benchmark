"""Measure all routing implementations on the same fixed model/data and one CPU."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import json, time, statistics, platform, hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from data_utils import resolve, write_json
from tao.hard import HardTAOClassifier
report=dict(host=platform.node(), seed=42, events=40000, features=16, classes=5,
            source_sha256=hashlib.sha256(resolve('src/tao/hard.py').read_bytes()).hexdigest(), measurements=[])
rng=np.random.default_rng(42)
x=rng.normal(size=(40000,16))
with threadpool_limits(1):
    for depth in [2,3,4,5]:
        tree=HardTAOClassifier(depth=depth).initialize(16,5).trees[0]
        tree.leaf_parameters[:]=rng.normal(size=tree.leaf_parameters.shape)
        reference=tree.predict_subtree(x,indexed=False)
        result=dict(depth=depth,timings={},max_logit_difference={})
        for mode in ['dense','indexed','batched']:
            candidate=tree.predict_subtree(x,indexed=mode)
            error=float(np.abs(candidate-reference).max())
            assert error<1e-10
            assert np.array_equal(candidate.argmax(axis=1),reference.argmax(axis=1))
            samples=[]
            for _ in range(7):
                tick=time.perf_counter();tree.predict_subtree(x,indexed=mode);samples.append(time.perf_counter()-tick)
            result['timings'][mode]=dict(median_seconds=statistics.median(samples),samples_seconds=samples)
            result['max_logit_difference'][mode]=error
        result['batched_speedup_over_dense']=result['timings']['dense']['median_seconds']/result['timings']['batched']['median_seconds']
        report['measurements'].append(result)
        print(json.dumps(result),flush=True)
write_json('results/tao_steps/batched_routing.json',report)
import common.syncer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,4.5))
for mode,label in [('dense','Full-array node loops'),('indexed','Reaching-event node loops'),('batched','Batched hyperplanes')]:
    ax.plot([r['depth'] for r in report['measurements']], [r['timings'][mode]['median_seconds']*1000 for r in report['measurements']], 'o-',label=label)
ax.set_xlabel('Tree depth');ax.set_ylabel('Prediction time [ms]');ax.grid(alpha=.2);ax.legend()
ax.set_title('Same hard tree and data | 40,000 events, 16 features, one thread')
fig.tight_layout();fig.savefig(resolve('results/tao_steps/batched_routing.png'),dpi=170)
plt.close(fig)
