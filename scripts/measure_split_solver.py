"""Compare identical L1 split objectives on a frozen real-training root problem."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import hashlib, json, pickle, platform, time
import numpy as np
from threadpoolctl import threadpool_limits
from data_utils import resolve, write_json
from tao.hard import cross_entropy_rows, fit_l1_split
run='qg_vs_wzt_hardtao_linear_d2_n1_sa1e-05_la1e-05_seed42_correctness_pilot'
with open(resolve('models/'+run+'.pkl'),'rb') as handle:model=pickle.load(handle)['model']
with np.load(resolve('data/qg_vs_wzt_standardized.npz')) as data:
    x,y=data['X_train'].astype(np.float64),data['y_train']
tree=model.trees[0]
with threadpool_limits(1):
    left=cross_entropy_rows(tree.predict_subtree(x,1,False),y)
    right=cross_entropy_rows(tree.predict_subtree(x,2,False),y)
    delta=left-right;active=np.abs(delta)>0
    xx=x[active];target=(delta[active]>0).astype(int);weight=np.abs(delta[active])
    report=dict(host=platform.node(),source_sha256=hashlib.sha256(resolve('src/tao/hard.py').read_bytes()).hexdigest(),
        training_events=len(y),active_events=int(active.sum()),features=x.shape[1],
        normalizer=len(y),split_alpha=model.split_alpha,max_iter=100,
        frozen_model=run,objective='weighted binary logistic / N_train + alpha*(L1(W)+abs(b))',
        measurements=[])
    for name,solver,w,b in [('liblinear_zero','liblinear',np.zeros(16),0),
                           ('lbfgsb_zero','lbfgsb',np.zeros(16),0),
                           ('lbfgsb_warm','lbfgsb',tree.split_w[0],tree.split_b[0])]:
        fitted_w,fitted_b,stats=fit_l1_split(xx,target,weight,len(y),model.split_alpha,w,b,
                                           max_iter=100,solver=solver)
        routing=x@fitted_w+fitted_b>0
        actual=float(np.where(routing,right,left).mean()+model.split_alpha*(np.abs(fitted_w).sum()+abs(fitted_b)))
        stats.update(name=name,actual_routing_objective=actual,nonzero_coefficients=int(np.count_nonzero(fitted_w)))
        report['measurements'].append(stats)
        write_json('results/tao_steps/split_solver.json',report)
        print(json.dumps(stats),flush=True)
import common.syncer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4))
labels=['Liblinear\nzero start','L-BFGS-B\nzero start','L-BFGS-B\nwarm start']
axes[0].bar(labels,[r['seconds'] for r in report['measurements']]);axes[0].set_ylabel('CPU wall time [s]')
axes[1].bar(labels,[r['surrogate_objective'] for r in report['measurements']]);axes[1].set_ylabel('Same weighted logistic + L1 objective')
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('Frozen root split on 531,200 training events | 100-iteration limits')
fig.tight_layout();fig.savefig(resolve('results/tao_steps/split_solver.png'),dpi=170);plt.close(fig)
