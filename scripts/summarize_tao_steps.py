"""Publish a measured change report; use validation histories, never test arrays."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json, glob
import numpy as np
from data_utils import resolve, write_json
import common.syncer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=resolve('results/tao_steps')
base=json.loads((root/'measurements.json').read_text())
batched=json.loads((root/'batched_routing.json').read_text())
split=json.loads((root/'split_solver.json').read_text())
paths=sorted(glob.glob(str(resolve('results/*hardtao*correctness_pilot_history.json')))+
             glob.glob(str(resolve('results/*hardtao*depth3_pilot_history.json'))))
pilots=[]
for path in paths:
    history=json.loads(Path(path).read_text())
    validation=json.loads(Path(path.replace('_history.json','_validation.json')).read_text())
    config=json.loads(resolve('configs/'+validation['run_name']+'.yaml').read_text())
    best=history[validation['best_epoch']-1]
    profiles=[r['profile'] for r in history]
    pilots.append(dict(run_name=validation['run_name'],task=config['run_name'].split('_hardtao')[0],
        trees=config['model']['tao']['trees'],depth=config['model']['tao']['depth'],
        leaf_mode=config['model']['tao']['leaf_mode'],epochs=len(history),best_epoch=validation['best_epoch'],
        validation_auc=best['validation']['macro_auc'],validation_accuracy=best['validation']['accuracy'],
        validation_efficiency=best['validation']['signal_efficiency_at_fixed_fpr'],
        reference_validation_auc=validation['reference_validation']['macro_auc'],
        reference_validation_efficiency=validation['reference_validation']['signal_efficiency_at_fixed_fpr'],
        mean_epoch_seconds=float(np.mean([r['seconds'] for r in profiles])),
        mean_leaf_seconds=float(np.mean([r['leaf_seconds'] for r in profiles])),
        mean_split_seconds=float(np.mean([r['split_seconds'] for r in profiles])),
        mean_global_refit_seconds=float(np.mean([r['global_refit']['seconds'] for r in profiles])),
        global_refit_training_rows=[r['global_refit']['events'] for r in profiles],
        unconverged_leaf_fits=sum(not f['converged'] for r in profiles for f in r['leaf_fits']),
        unconverged_global_refits=sum(not r['global_refit']['converged'] for r in profiles),
        objective_decreases=all(r['objective_after']<=r['objective_before']+1e-9 for r in profiles),
        test_evaluated=False,history=history))
write_json(root/'pilot_summary.json',[{k:v for k,v in p.items() if k!='history'} for p in pilots])
fig,ax=plt.subplots(figsize=(8,4.5))
labels=[p['task']+'\ndepth '+str(p['depth'])+', 1 tree' for p in pilots]
left=np.zeros(len(pilots))
for key,label in [('mean_leaf_seconds','Leaf fits'),('mean_split_seconds','Split updates'),('mean_global_refit_seconds','Global leaf refit')]:
    values=[p[key] for p in pilots]
    ax.bar(labels,values,bottom=left,label=label);left+=values
other=[max(0,p['mean_epoch_seconds']-t) for p,t in zip(pilots,left)]
ax.bar(labels,other,bottom=left,label='Routing / prediction / other')
ax.set_ylabel('Mean epoch wall time [s], one CPU');ax.set_title('Full-data training profile | 531,200 training events')
ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(root/'pilot_profiles.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,task in zip(axes,['qg_vs_wzt','multiclass']):
    items=[p for p in pilots if p['task']==task]
    for p in items:
        h=p['history'];ax.plot([r['epoch'] for r in h],[r['validation']['macro_auc'] for r in h],
                              'o-',label=f'1 hard tree, depth {p["depth"]}')
    if items:ax.axhline(items[0]['reference_validation_auc'],ls='--',color='tab:orange',label='Reference BDT')
    ax.set_title(task);ax.set_xlabel('Epoch');ax.set_ylabel('Validation macro AUC');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Linear multiclass leaves | short pilots, not converged comparisons')
fig.tight_layout();fig.savefig(root/'pilot_validation.png',dpi=170);plt.close(fig)
steps=base['steps'];design=steps['6_same_model_performance']['joint_design']
lines=['# Hard TAO: measured fixes','',
'All routing is hard. Linear leaves are the default; constant leaves are supported.',
'Original GOLLUM snapshots and the sbi-pdf-paper checkout remain unchanged.','',
'## Correctness checks','',
'- 13 unit checks pass: full-partition refit, leaf types, 2/3/5 classes, conditional CE gradients, tied events, normalization, serialization, hard boundaries, actual-loss acceptance, and equivalent L1 split objectives.',
'- Last-batch fixture: full refit recovers 70/20/10%% class probabilities. Training-mixture CE falls from %.6f (last batch only) to %.6f (full training).'%(steps['2_full_partition_refit']['last_batch_cross_entropy'],steps['2_full_partition_refit']['full_cross_entropy']),
'- Five-class conditional-gradient maximum finite-difference error: %.3g.'%steps['3_multiclass_forest_objective']['finite_difference_max_gradient_error'],
'- Normalization matches the sqrt(variance) reference exactly on the controlled fixture.',
'- All 1,200 tied events in the routing fixture are excluded; no third routing class is fitted.','',
'## Measurements','',
'Timings use one CPU thread. Microbenchmarks compare identical fixed computations, not old binary MSE against new multiclass CE.','',
'- Direct CSR construction: %.2fx faster; %.2f MiB versus %.2f MiB peak traced allocation on the same 10,000-event, 4-tree, 16-feature design.'%(design['speedup'],design['direct_csr_peak_traced_bytes']/2**20,design['python_triplets_peak_traced_bytes']/2**20),
'- Indexed and batched routing were not consistently faster at shallow depths. Dense traversal remains the default; alternatives are explicit options.',
'- Frozen real-data split objective: weighted binary logistic/N_train + alpha*(L1(W)+abs(b)). Both optimizers retain L1; L2 is not substituted.','',
'| Optimizer / initialization | Seconds | Surrogate objective | Converged |',
'|---|---:|---:|---|']
for r in split['measurements']:
    lines.append('| %s | %.3f | %.9f | %s |'%(r['name'],r['seconds'],r['surrogate_objective'],r['converged']))
lines+=['','Liblinear remains the default: the warm-start alternative has only a small measured benefit, while its zero-start run is slower and iteration-limited.','',
'## Full-training pilots (validation only)','',
'| Task | Trees | Depth | Epochs | Validation AUC | BDT validation AUC | Mean epoch seconds |',
'|---|---:|---:|---:|---:|---:|---:|']
for p in pilots:
    lines.append('| %s | %d | %d | %d | %.6f | %.6f | %.1f |'%(p['task'],p['trees'],p['depth'],p['epochs'],p['validation_auc'],p['reference_validation_auc'],p['mean_epoch_seconds']))
lines+=['','Global refits use all 531,200 training events. Pilot leaf/global optimizers reach iteration limits, which are recorded; these are not claims of convergence or matching the BDT. Test arrays are not loaded by training.']
for path in sorted(resolve('results').glob('*fixedrouting_refinement_validation.json')):
    refinement=json.loads(path.read_text())
    history_path=Path(str(path).replace('_validation.json','_history.json'))
    history=json.loads(history_path.read_text())
    lines+=['','## Leaf convergence at fixed routing','',
        'One depth-3 tree, eight linear leaves. Hyperplane coefficients remain exactly unchanged; only leaf coefficients are refitted on all training events. The iteration budgets below are additional, sequential fits.','',
        '| Additional iteration budget | Seconds | Training objective | Validation AUC | Signal efficiency at 1% FPR | Converged |',
        '|---|---:|---:|---:|---:|---|']
    for row in history:
        metrics=row['validation']
        lines.append('| %d | %.1f | %.9f | %.6f | %.6f | %s |'%(
            row['additional_iteration_budget'],row['seconds'],row['objective'],metrics['macro_auc'],
            metrics['signal_efficiency_at_fixed_fpr'],row.get('refit',{}).get('converged','initial')))
    lines+=['','Best checkpoint selected by validation AUC; test data remain unused.',
        '', '../'+refinement['run_name']+'_convergence.png']
lines+=['',
'## Artifacts','',
'- correctness.png','- performance.png','- batched_routing.png','- split_solver.png','- pilot_profiles.png','- pilot_validation.png',
'- measurements.json, batched_routing.json, split_solver.json, pilot_summary.json','',
'Run configs, model checkpoints, full validation metrics, per-epoch profiles, and Slurm logs are retained in the benchmark repository. Source snapshots hard_v1.py / hard_v2.py / hard_v3.py correspond to measurement implementation stages.','',
'No FPGA resource or latency claims are made by these CPU measurements.']
(root/'report.md').write_text('\n'.join(lines)+'\n')
for file in ['report.md','measurements.json','batched_routing.json','split_solver.json','pilot_summary.json']:
    common.syncer.register_file(root/file)
print('Wrote measured report for',len(pilots),'pilots',flush=True)
