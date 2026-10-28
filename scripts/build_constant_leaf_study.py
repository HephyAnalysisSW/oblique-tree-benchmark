"""Freeze and evaluate the controlled constant-leaf factorial scan for the paper.

Only numerically converged source runs are eligible. Select the best saved
1%-background checkpoint, then re-evaluate on the unchanged validation arrays.
An incomplete case remains explicit and cannot contribute a selected model.
"""
from pathlib import Path
import argparse,hashlib,json,pickle,shutil,sys,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from sklearn.metrics import roc_curve
from evaluate import compute_metrics
from tao.sparsity import sparsity_profile
from tao.portable import export_model
import common.syncer
import matplotlib.pyplot as plt


def read(path):return json.loads((ROOT/path).read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2)+'\n')
def efficiency(record):return record['signal_efficiency_at_fixed_fpr']
def strict(y,score):
 bg=np.sort(score[y==0])[::-1];budget=int(.01*len(bg));threshold=float(bg[budget]);accepted=score>threshold
 return dict(threshold=threshold,accepted_background=int(accepted[y==0].sum()),
  background_efficiency=float(accepted[y==0].mean()),signal_efficiency=float(accepted[y==1].mean()),budget=budget)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-complete',action='store_true');args=p.parse_args()
 scan=read('configs/tao_hyper_scan_v1.json');cases=[c for c in scan['cases'] if c['leaf_mode']=='constant']
 audit=read('results/tao_split_audit.json');dataset=ROOT/'data/qg_vs_wzt_standardized.npz'
 assert sha(dataset)==audit['dataset_sha256']
 contract=read('fpga_handoff/v3/input_contract.json')
 with np.load(dataset,allow_pickle=False) as d:x,y=d['X_val'].astype(float),d['y_val']
 records=[];pending=[];models={};probabilities={}
 for c in cases:
  config=read(c['config']);assert config['initialization']=='cart' and config['split_search']=='hard_threshold'
  layout=config['layouts'][0]
  run=(f"{config['task']}_hardtao_constant_d{layout['depth']}_n{layout['trees']}"
       f"_sa{config['split_alpha']:g}_la{config['leaf_alpha']:g}_seed{config['seed']}_{config['run_suffix']}")
  marker=ROOT/f"results/{config['scan_name']}_member_0.json"
  status_path=ROOT/f'results/{run}_status.json'
  status=json.loads(status_path.read_text()) if status_path.exists() else {}
  complete=marker.exists() and json.loads(marker.read_text()).get('converged') and status.get('converged')
  if not complete:
   pending.append(dict(case=c,run_name=run,status=status));continue
  history=read(f'results/{run}_history.json');candidates=[]
  for suffix in ['_best_efficiency','_best_validation','']:
   path=ROOT/f'models/{run}{suffix}.pkl'
   if not path.exists():continue
   with path.open('rb') as f:payload=pickle.load(f)
   m=payload.get('validation') or payload['history'][-1]['validation']
   candidates.append((efficiency(m),m['signal_auc'],path,payload,m))
  _,_,path,payload,expected=max(candidates,key=lambda a:a[:2])
  model=payload['model'];assert model.leaf_mode=='constant'
  assert payload['config']['dataset_sha256']==audit['dataset_sha256']
  assert model.ntrees==c['trees'] and model.depth==c['depth']
  prob=model.predict_proba(x);m,_=compute_metrics(y,prob,contract['classes'])
  assert abs(efficiency(m)-efficiency(expected))<1e-9 and abs(m['per_class_auc'][contract['classes'][1]]-expected['signal_auc'])<1e-9
  checkpoint_sha=sha(path);archive=ROOT/'models/constant_study_v1_selected'/f'{checkpoint_sha}.pkl';archive.parent.mkdir(parents=True,exist_ok=True)
  if not archive.exists():shutil.copy2(path,archive)
  assert sha(archive)==checkpoint_sha
  terminal=history[-1]['validation']
  row=dict(case=c,run_name=run,configuration=config,source_checkpoint=str(path.relative_to(ROOT)),
   checkpoint_sha256=checkpoint_sha,immutable_checkpoint=str(archive.relative_to(ROOT)),
   selected_epoch=payload.get('epoch',len(history)),source_terminal_epoch=len(history),source_converged=True,
   selected_converged_at_save=payload.get('training_converged_at_save',payload.get('training_converged',False)),
   terminal_metrics=terminal,selected_metrics=m,strict_probability_working_point=strict(y,prob[:,1]),
   strict_working_point=strict(y,-model.decision_function(x)[:,0]),
   sparsity=sparsity_profile(model),training_compute_seconds=status['training_seconds'],terminal_convergence=history[-1]['convergence'])
  records.append(row);models[c['index']]=model;probabilities[c['index']]=prob[:,1]
  print(c['index'],c['trees'],c['depth'],100*efficiency(m),flush=True)
 records.sort(key=lambda r:efficiency(r['selected_metrics']),reverse=True)
 assert records
 best_by_layout=[]
 for trees in [1,3,5]:
  for depth in [2,3,4]:
   eligible=[r for r in records if r['case']['trees']==trees and r['case']['depth']==depth]
   if eligible:best_by_layout.append(max(eligible,key=lambda r:(efficiency(r['selected_metrics']),r['selected_metrics']['per_class_auc'][contract['classes'][1]])))
 result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),dataset_sha256=audit['dataset_sha256'],test_evaluated=False,
  complete=len(records)==len(cases),completed=len(records),total=len(cases),pending=pending,
  primary_metric='interpolated signal efficiency at 1% background; signal AUC secondary',
  selection='best available saved checkpoint from a numerically converged source run; incomplete runs excluded',
  models=records,best_by_layout=best_by_layout,winner=records[0],
  source_hashes={f:sha(ROOT/f) for f in ['src/tao/hard.py','src/tao/split_search.py','src/tao/initialization.py','src/tao/convergence.py','scripts/train_tao_fully.py','scripts/build_constant_leaf_study.py']})
 write(ROOT/'paper/sources/constant_leaf_study.json',result)
 # Export the actual selected model separately; never changes CURRENT_MODEL or v3.
 winner=records[0];model=models[winner['case']['index']]
 spec=export_model(model,contract['features'],contract['classes'],contract['mean'],contract['scale'])
 write(ROOT/'paper/sources/constant_leaf_winner.json',spec)
 # ROCs compare the best completed constant model at each depth-four layout.
 selections=[r for r in best_by_layout if r['case']['depth']==4]
 def timing(predict):
  predict();wall=[];cpu=[]
  for _ in range(9):
   start=time.perf_counter();c=time.process_time();predict();cpu.append(time.process_time()-c);wall.append(time.perf_counter()-start)
  return dict(batch=len(x),repeats=9,threads=1,median_microseconds_per_event=float(np.median(wall)*1e6/len(x)),median_cpu_seconds=float(np.median(cpu)),note='Bulk API throughput; not FPGA or single-event latency')
 import xgboost as xgb
 booster=xgb.Booster();booster.load_model(ROOT/'fpga_handoff/v3/xgboost_reference/model.json');booster.set_param({'nthread':1})
 matrix=xgb.DMatrix(x);bdt=booster.predict(matrix)
 # inplace_predict bypasses XGBoost's DMatrix prediction cache on every repeat.
 xgb_input=np.ascontiguousarray(x,dtype=np.float32)
 np.testing.assert_array_equal(booster.inplace_predict(xgb_input),bdt)
 bdt_timing=timing(lambda:booster.inplace_predict(xgb_input))
 curves=[(f"XGBoost (100 × D4; {bdt_timing['median_microseconds_per_event']:.2f} μs/event)",bdt)]
 for row in selections:
  member=models[row['case']['index']];row['inference_timing']=timing(lambda:member.predict_proba(x))
  curves.append((f"Constant ({row['case']['trees']} × D4; {row['inference_timing']['median_microseconds_per_event']:.2f} μs/event)",probabilities[row['case']['index']]))
 frozen=read('paper/sources/results_snapshot.json')['selections']['compact_match_xgb']['record']
 path=ROOT/frozen['immutable_snapshot'];assert sha(path)==frozen['sha256']
 with path.open('rb') as f:linear=pickle.load(f)['model']
 linear_timing=timing(lambda:linear.predict_proba(x))
 result['comparison_timing']=dict(host=__import__('socket').gethostname(),xgboost=bdt_timing,compact_refit=linear_timing,input_dtype=str(x.dtype),xgboost_input='contiguous float32, conversion excluded',xgboost_api='uncached Booster.inplace_predict',tao_api='NumPy predict_proba')
 curves.append((f"Linear Compact refit (3 × D4; {linear_timing['median_microseconds_per_event']:.2f} μs/event)",linear.predict_proba(x)[:,1]))
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for label,score in curves:
  fpr,tpr,_=roc_curve(y,score)
  for ax in axes:ax.plot(fpr,tpr,label=label,lw=1.25)
 axes[0].set(xlim=(0,1),ylim=(0,1),title='Validation ROC')
 axes[1].set_xscale('log');axes[1].set(xlim=(1e-4,.1),ylim=(0,1),title='Low background acceptance')
 for ax in axes:ax.set_xlabel('Background efficiency');ax.set_ylabel('Signal efficiency');ax.grid(alpha=.2);ax.axvline(.01,color='gray',ls=':',lw=.7)
 axes[1].legend(fontsize=7,loc='lower right');fig.tight_layout()
 fig.savefig(ROOT/'paper/figures/constant_leaf_roc.pdf');fig.savefig(ROOT/'paper/figures/constant_leaf_roc.png',dpi=160);plt.close(fig)
 rows=[]
 for r in best_by_layout:
  c=r['case'];s=r['sparsity'];m=r['selected_metrics'];w=r['strict_working_point']
  alpha={0.:'0',1e-7:r'$10^{-7}$',1e-5:r'$10^{-5}$'}[c['split_alpha']]
  rows.append(f"{c['trees']} $\\times$ D{c['depth']} & {alpha} & {s['dense_parameter_slots']} & {s['nonzero_coefficients']} & {100*efficiency(m):.2f} & {100*w['signal_efficiency']:.2f} & {m['per_class_auc'][contract['classes'][1]]:.5f} \\\\")
 (ROOT/'paper/generated/constant_leaf_results.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
 c=winner['case'];s=winner['sparsity']
 macros={'ConstantCompleted':result['completed'],'ConstantTotal':result['total'],'ConstantWinnerTrees':c['trees'],
  'ConstantWinnerDepth':c['depth'],'ConstantWinnerEpoch':winner['selected_epoch'],'ConstantWinnerTerminalEpoch':winner['source_terminal_epoch'],'ConstantWinnerDense':s['dense_parameter_slots'],'ConstantWinnerNonzero':s['nonzero_coefficients'],
  'ConstantWinnerEfficiency':f"{100*efficiency(winner['selected_metrics']):.2f}",
  'ConstantWinnerStrict':f"{100*winner['strict_working_point']['signal_efficiency']:.2f}"}
 (ROOT/'paper/generated/constant_leaf_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n')
 lines=['# Constant-leaf factorial study','',f"Converged source runs: {result['completed']}/{result['total']}.",
  '', '| Trees | Depth | Split L1 | Dense / nonzero | Interpolated signal efficiency [%] | Realizable signal efficiency [%] | Training compute [min] |',
  '|---:|---:|---:|---:|---:|---:|---:|']
 for r in best_by_layout:
  c=r['case'];s=r['sparsity'];lines.append(f"| {c['trees']} | {c['depth']} | {c['split_alpha']:g} | {s['dense_parameter_slots']} / {s['nonzero_coefficients']} | {100*efficiency(r['selected_metrics']):.3f} | {100*r['strict_working_point']['signal_efficiency']:.3f} | {r['training_compute_seconds']/60:.1f} |")
 lines+=['','Every selected snapshot is archived by SHA-256. Selected and terminal epochs, convergence flags, configuration and metrics are separate in the JSON.',
  '', 'Pending cases: '+(', '.join(str(r['case']['index']) for r in pending) or 'none')+'.',
  '', 'Constant leaves have no penalized feature weights. The nominal leaf penalty does not affect their objective.',
  '', 'Validation only. This controlled CART-initialized cohort is separate from the older seven-layout random-initialization study.']
 (ROOT/'docs/constant_leaf_study_v1.md').write_text('\n'.join(lines)+'\n')
 result['asset_hashes']={str(path.relative_to(ROOT/'paper')):sha(path) for path in [ROOT/'paper/sources/constant_leaf_winner.json',ROOT/'paper/generated/constant_leaf_results.tex',ROOT/'paper/generated/constant_leaf_macros.tex',ROOT/'paper/figures/constant_leaf_roc.pdf']}
 write(ROOT/'paper/sources/constant_leaf_study.json',result)
 print(json.dumps({'complete':result['complete'],'completed':len(records),'total':len(cases),'winner':winner['case']['index']}),flush=True)
 if args.require_complete and pending:raise SystemExit('Constant scan remains incomplete; no incomplete case was selected')

if __name__=='__main__':main()
