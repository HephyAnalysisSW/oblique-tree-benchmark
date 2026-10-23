"""Measure optimizer convergence at fixed hard routing, without adding trees."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,copy,json,pickle,time
import numpy as np
from data_utils import load_config,load_metadata,resolve,write_json
from evaluate import compute_metrics
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',required=True)
p.add_argument('--budgets',type=int,nargs='+',default=[100,300,1000])
a=p.parse_args()
config=load_config(a.config)
base=config['run_name'];run=base+'_fixedrouting_refinement'
out=resolve('models/'+run+'.pkl')
if out.exists():raise FileExistsError(out)
with open(resolve(config['model']['output_model_path']),'rb') as f:payload=pickle.load(f)
model=copy.deepcopy(payload['model'])
with np.load(resolve(config['data']['dataset_path'])) as d:
 x,y=d['X_train'].astype(float),d['y_train'];xv,yv=d['X_val'].astype(float),d['y_val']
metadata=load_metadata(config['data']['metadata_path'])
reference,_=compute_metrics(yv,model.predict_proba(xv),metadata['class_names'])
initial_route=[(t.split_w.copy(),t.split_b.copy()) for t in model.trees]
history=[dict(additional_iteration_budget=0,seconds=0,validation=reference,objective=model.objective(x,y))]
best_auc=reference['macro_auc'];best_model=copy.deepcopy(model);best_stage=0
new_config=copy.deepcopy(config);new_config.pop('_config_path',None)
new_config['run_name']=run
new_config['model']['output_model_path']='models/'+run+'.pkl'
new_config['evaluation']['output_prefix']='results/'+run
new_config['model']['fixed_routing_refinement']=dict(base_run=base,budgets=a.budgets,test_evaluated=False)
for stage,budget in enumerate(a.budgets,1):
 model.leaf_max_iter=budget
 tick=time.perf_counter();stats=model.global_refit(x,y);elapsed=time.perf_counter()-tick
 for tree,(w,b) in zip(model.trees,initial_route):
  assert np.array_equal(tree.split_w,w) and np.array_equal(tree.split_b,b)
 metrics,_=compute_metrics(yv,model.predict_proba(xv),metadata['class_names'])
 row=dict(additional_iteration_budget=budget,seconds=elapsed,validation=metrics,
          objective=model.objective(x,y),refit=stats)
 history.append(row)
 assert row['objective']<=history[-2]['objective']+1e-9
 if metrics['macro_auc']>best_auc:
  best_auc=metrics['macro_auc'];best_model=copy.deepcopy(model);best_stage=stage
 write_json('results/'+run+'_history.json',history)
 print(json.dumps(dict(stage=stage,budget=budget,seconds=elapsed,converged=stats['converged'],
  objective=row['objective'],validation_auc=metrics['macro_auc'],
  validation_efficiency=metrics['signal_efficiency_at_fixed_fpr'])),flush=True)
 with out.with_suffix('.pkl.tmp').open('wb') as f:
  pickle.dump(dict(model=best_model,config=new_config,metadata=metadata,history=history),f,
              protocol=pickle.HIGHEST_PROTOCOL)
 out.with_suffix('.pkl.tmp').replace(out)
 write_json('configs/'+run+'.yaml',new_config)
 write_json('results/'+run+'_validation.json',dict(run_name=run,best_stage=best_stage,
  validation=history[best_stage]['validation'],complexity=best_model.complexity(),
  test_evaluated=False,routing_unchanged=True))
 if stats['converged']:break
fig,axes=plt.subplots(1,2,figsize=(10,4))
indices=np.arange(len(history));labels=[str(h['additional_iteration_budget']) for h in history]
axes[0].plot(indices,[h['objective'] for h in history],'o-');axes[0].set_ylabel('Training CE + L1')
axes[1].plot(indices,[h['validation']['macro_auc'] for h in history],'o-');axes[1].set_ylabel('Validation macro AUC')
for ax in axes:ax.set_xticks(indices,labels);ax.set_xlabel('Additional leaf-optimizer iteration budget');ax.grid(alpha=.2)
fig.suptitle(f'Fixed hyperplanes | {model.ntrees} hard tree(s), depth {model.depth} | {model.leaf_mode} leaves')
fig.tight_layout();fig.savefig(resolve('results/'+run+'_convergence.png'),dpi=170);plt.close(fig)
print(json.dumps(dict(run_name=run,best_stage=best_stage,best_auc=best_auc,test_evaluated=False)),flush=True)
