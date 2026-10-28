"""Validation-only status, parameter accounting and eligible ROCs for the axis study."""
import hashlib,json,pickle,shutil,socket,time
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from evaluate import compute_metrics
from tao.axis import assert_axis,axis_parameter_counts
from tao.portable import predict_margin,deterministic_working_point
from tao.sparsity import sparsity_profile
import common.syncer as syncer


def read(path,default=None):
    path=ROOT/path
    return json.loads(path.read_text()) if path.exists() else default

def metric(payload):
    return payload.get('validation') or payload.get('metrics',{}).get('validation') or payload['history'][-1]['validation']

def main():
    scan=read('configs/axis_linear_v1.json');audit=read('results/tao_split_audit.json')
    dataset=ROOT/'data/qg_vs_wzt_standardized.npz'
    assert hashlib.sha256(dataset.read_bytes()).hexdigest()==audit['dataset_sha256']
    with np.load(dataset,allow_pickle=False) as data:x,y=data['X_val'].astype(float),data['y_val']
    contract=read('fpga_handoff/v3/input_contract.json');rows=[];curves=[];best=None
    for case in scan['cases']:
        cfg=read(case['config']);marker=read(f"results/{cfg['scan_name']}_member_0.json",{})
        status=read('results/'+case['run_name']+'_status.json',{})
        refit=read('results/'+case['refit_run_name']+'_status.json',{})
        history=read('results/'+case['run_name']+'_history.json',[])
        trained=bool(marker.get('complete') and marker.get('converged'))
        row=dict(case=case,training=status,refit=refit,source_converged=trained,
            latest_provisional_validation=history[-1]['validation'] if history else None,selected=None)
        eligible=[]
        if trained:
            eligible+=[('regularized',ROOT/('models/'+case['run_name']+suffix+'.pkl')) for suffix in ['_best_efficiency','']]
        if refit.get('converged'):
            eligible+=[('leaf_refit',ROOT/('models/'+case['refit_run_name']+suffix+'.pkl')) for suffix in ['_leaf_only_best_efficiency','_leaf_only']]
        candidates=[]
        for stage,path in eligible:
            if path.exists():
                payload=pickle.loads(path.read_bytes());m=metric(payload)
                candidates.append((m['signal_efficiency_at_fixed_fpr'],m['signal_auc'],stage,path,payload))
        if candidates:
            _,_,stage,path,payload=max(candidates,key=lambda r:r[:2]);model=payload['model'];assert_axis(model)
            prob=model.predict_proba(x);m,_=compute_metrics(y,prob,contract['classes'])
            expected=metric(payload)
            assert abs(m['signal_efficiency_at_fixed_fpr']-expected['signal_efficiency_at_fixed_fpr'])<1e-9
            margin=-model.decision_function(x)[:,0];counts=axis_parameter_counts(model)
            model.predict_proba(x);times=[]
            for _ in range(9):
                start=time.perf_counter();model.predict_proba(x);times.append(time.perf_counter()-start)
            selected=dict(stage=stage,checkpoint=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                selected_epoch=payload.get('epoch'),metrics=m,strict_working_point=deterministic_working_point(margin,y),
                parameters=counts,sparsity=sparsity_profile(model),cpu_microseconds_per_event=float(np.median(times)*1e6/len(x)),
                cpu_note='NumPy API, one thread, full batch, nine repeats; not FPGA latency or the matched C++ benchmark',
                source_converged=True,refit_converged=bool(refit.get('converged')))
            row['selected']=selected
            curves.append((f"Axis {case['trees']} × D{case['depth']} ({counts['learned_nonzero_scalars']} learned nnz; {selected['cpu_microseconds_per_event']:.2f} μs/event NumPy)",prob[:,1]))
            if best is None or (m['signal_efficiency_at_fixed_fpr'],m['signal_auc'])>(best['selected']['metrics']['signal_efficiency_at_fixed_fpr'],best['selected']['metrics']['signal_auc']):best=row
        rows.append(row)
    for label,path in [('Oblique linear Compact refit','fpga_handoff/v3/compact_refit_3t_d4/standardized/model.json'),('Oblique constant winner','paper/sources/constant_leaf_winner.json')]:
        margin,_,_=predict_margin(read(path),x);curves.append((label,margin))
    import xgboost as xgb
    bdt=xgb.Booster();bdt.load_model(ROOT/'fpga_handoff/v3/xgboost_reference/model.json');bdt.set_param({'nthread':1})
    curves.append(('XGBoost',bdt.inplace_predict(np.asarray(x,dtype=np.float32),predict_type='margin')))
    terminal=all(row['refit'].get('converged') or row['refit'].get('phase') in ['separated','inconclusive','stalled'] for row in rows)
    report=dict(updated_utc=datetime.now(timezone.utc).isoformat(),scan=scan,models=rows,best_eligible_case=None if best is None else best['case']['index'],
        complete=terminal,converged_sources=sum(r['source_converged'] for r in rows),converged_refits=sum(bool(r['refit'].get('converged')) for r in rows),
        test_evaluated=False,dataset_sha256=audit['dataset_sha256'],host=socket.gethostname())
    folder=ROOT/'docs/axis_linear_v1';folder.mkdir(parents=True,exist_ok=True)
    (folder/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Axis-aligned cuts with sparse linear leaves','',f"Converged sources: {report['converged_sources']}/4; converged frozen-leaf refits: {report['converged_refits']}/4.",
        '', 'Only checkpoints from converged source stages enter the selected comparison. Latest values from incomplete sources remain explicitly provisional. Same fixed train/validation arrays; no test evaluation.',
        '', '| Layout | Phase | Selected stage | Continuous slots + feature indices | Learned nonzero scalars | Signal efficiency at 1% bkg [%] | Realizable [%] |',
        '|---|---|---|---|---:|---:|---:|']
    for row in rows:
        c=row['case'];s=row['selected'];phase='refit complete' if row['refit'].get('converged') else row['refit'].get('phase') or row['training'].get('phase','pending')
        if s:
            p=s['parameters'];lines.append(f"| {c['trees']} × D{c['depth']} | {phase} | {s['stage']} | {p['continuous_slots']} + {p['feature_index_slots']} | {p['learned_nonzero_scalars']} | {100*s['metrics']['signal_efficiency_at_fixed_fpr']:.3f} | {100*s['strict_working_point']['signal_efficiency']:.3f} |")
        else:
            provisional=row['latest_provisional_validation'];note='pending' if provisional is None else f"{100*provisional['signal_efficiency_at_fixed_fpr']:.3f} provisional"
            lines.append(f"| {c['trees']} × D{c['depth']} | {phase} | ineligible | — | — | {note} | — |")
    lines+=['', 'Axis cuts use a learned feature index, a threshold and a direction bit. Fixed unit coefficients are excluded from learned-scalar counts; their sparse portable representation is reported separately in JSON. Dense continuous slots count one threshold per internal node plus all 17 leaf coefficients. Feature indices, orientation bits and constant-routing flags are separate storage costs. All routing multiplications can be eliminated; affine leaves still require products.',
        '', 'Training searches every feature, both directions and every representable threshold interval using current child-subtree cross-entropies. Updates are accepted only if the true hard loss decreases. Tied child-loss events have no influence on the split. Leaf L1 is 1e-5; split L1 is zero because unit directions are fixed/discrete. CART initialization uses only training rows and seed 42. Each source must pass the existing objective/KKT/routing stability rules. Frozen-support, unpenalized leaf-only refits follow, with unchanged cuts and explicit separation checks.',
        '', '`configs/axis_linear_v1.json` lists four cases and their exact policies. `scripts/train_axis_tao.py` isolates the subclass from ongoing oblique runs. `scripts/monitor_axis_study.py` submits at most two jobs, continues only clean exit 75 and automatically starts refits from converged terminal parents. Numerical failures require inspection; held refits are not promoted.',
        '', 'CPU timings in the ROC legend are NumPy full-batch API timings, not the separate matched C++ benchmark. [ROC](roc.pdf) · [machine-readable status](status.json).',
        '', 'Reproduce in the pinned training environment with `PYTHONPATH=src:. python scripts/train_axis_tao.py --config configs/axis_linear_v1/case_0.json --index 0`. Use your own checkout/output area. `python scripts/summarize_axis_study.py` refreshes this comparison. No existing frozen handover is replaced.']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for label,scores in curves:
        fpr,tpr,_=roc_curve(y,scores)
        for ax in axes:ax.plot(fpr,tpr,label=label,lw=1.2)
    axes[0].set(xlim=(0,1),ylim=(0,1),title='Validation ROC');axes[1].set_xscale('log');axes[1].set(xlim=(1e-4,.1),ylim=(0,1),title='Low background acceptance')
    for ax in axes:ax.set_xlabel('Background efficiency');ax.set_ylabel('Signal efficiency');ax.grid(alpha=.2);ax.axvline(.01,color='gray',ls=':',lw=.8)
    axes[1].legend(fontsize=6.5,loc='lower right');fig.suptitle(f"Axis-aligned linear leaves: {report['converged_sources']}/4 sources, {report['converged_refits']}/4 refits converged")
    fig.tight_layout();fig.savefig(folder/'roc.pdf');fig.savefig(folder/'roc.png',dpi=170);plt.close(fig)
    for name in ['README.md','status.json']:syncer.register_file(folder/name)
    print(json.dumps({k:report[k] for k in ['complete','converged_sources','converged_refits','best_eligible_case']}),flush=True)

if __name__=='__main__':main()
