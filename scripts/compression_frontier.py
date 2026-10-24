"""Checkpoint-matched efficiency/sparsity frontier across the active studies."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import hashlib,json,math,pickle
from datetime import datetime,timezone
import matplotlib.pyplot as plt
from data_utils import resolve,write_json
from tao.portable import restore_model
from tao.sparsity import sparsity_profile
import common.syncer as syncer


def pareto_frontier(points):
    """Maximize efficiency and minimize exact nonzero coefficients."""
    return sorted([p for p in points if not any(
        q['nonzero_coefficients']<=p['nonzero_coefficients'] and q['efficiency']>=p['efficiency']
        and (q['nonzero_coefficients']<p['nonzero_coefficients'] or q['efficiency']>p['efficiency'])
        for q in points)],key=lambda p:(p['nonzero_coefficients'],-p['efficiency']))


def smallest_at_efficiency(points,target):
    eligible=[p for p in points if p['efficiency']>=target]
    return min(eligible,key=lambda p:(p['nonzero_coefficients'],p['mac_bound'],-p['efficiency'])) if eligible else None


def publish():
    plan=json.loads(resolve('configs/tao_l1_path_v1.json').read_text());manifest_path=resolve(plan['frozen_manifest'])
    manifest=json.loads(manifest_path.read_text());points=[];errors=[];blobs={}
    for case in manifest['candidates']:
        spec=json.loads((manifest_path.parent/case['id']/'standardized/model.json').read_text())
        sp=sparsity_profile(restore_model(spec))
        points.append(dict(id='frozen:'+case['id'],label=case['id'],source='frozen',
            checkpoint_sha256=case['checkpoint_sha256'],efficiency=case['metrics']['signal_efficiency_at_fixed_fpr'],auc=case['metrics']['signal_auc'],
            nonzero_coefficients=sp['nonzero_coefficients'],dense_slots=sp['dense_parameter_slots'],mac_bound=sp['selected_path_multiplications_upper_bound'],
            run_converged=case['training_run_converged'],selected_epoch=case['selected_epoch'],
            snapshot=str(manifest_path.parent.relative_to(resolve('.'))/case['id']/'standardized/model.json')))
    seen={p['checkpoint_sha256'] for p in points}
    paths=list(resolve('models').glob('*_hyper_scan_v1_c*_best_efficiency.pkl'))+list(resolve('models').glob('*_l1_path_v1*_best_efficiency.pkl'))
    for path in paths:
        try:
            blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
            if digest in seen:continue
            payload=pickle.loads(blob);cfg=payload['config'];metric=payload['validation']
            if cfg.get('dataset_sha256')!=manifest['dataset_sha256'] or metric['fixed_fpr']!=.01:continue
            if not math.isfinite(metric['signal_efficiency_at_fixed_fpr']):continue
            sp=sparsity_profile(payload['model']);run=cfg['run_name']
            status_path=resolve('results/'+run+'_status.json');status=json.loads(status_path.read_text()) if status_path.exists() else {}
            source='l1_path' if '_l1_path_v1' in run else 'architecture_scan'
            points.append(dict(id=digest,label=run.split('_seed42_')[-1],source=source,run_name=run,
                checkpoint_sha256=digest,efficiency=metric['signal_efficiency_at_fixed_fpr'],auc=metric['signal_auc'],
                nonzero_coefficients=sp['nonzero_coefficients'],dense_slots=sp['dense_parameter_slots'],mac_bound=sp['selected_path_multiplications_upper_bound'],
                run_converged=bool(status.get('converged')),selected_epoch=payload['epoch'],snapshot=None))
            blobs[digest]=blob;seen.add(digest)
        except Exception as exc:errors.append(dict(path=str(path),error=str(exc)))
    frontier=pareto_frontier(points);frozen=next(p for p in points if p['label']==plan['candidate_id'])
    reference=manifest['xgboost']['metrics']['signal_efficiency_at_fixed_fpr'];bands=[]
    for name,efficiency in [('XGBoost',reference),('frozen winner',frozen['efficiency'])]:
        for loss_pp in [0.,.25,.5,1.]:
            chosen=smallest_at_efficiency(points,efficiency-loss_pp/100)
            bands.append(dict(reference=name,loss_percentage_points=loss_pp,target_efficiency=efficiency-loss_pp/100,
                candidate=chosen['id'] if chosen else None))
    retained={p['id'] for p in frontier}|{b['candidate'] for b in bands if b['candidate']}
    directory=resolve('models/compression_frontier');directory.mkdir(parents=True,exist_ok=True)
    for p in points:
        if p['id'] in retained and p['id'] in blobs:
            target=directory/(p['checkpoint_sha256']+'.pkl')
            if not target.exists():
                temporary=target.with_suffix('.tmp');temporary.write_bytes(blobs[p['id']]);temporary.replace(target)
            assert hashlib.sha256(target.read_bytes()).hexdigest()==p['checkpoint_sha256']
            p['snapshot']=str(target.relative_to(resolve('.')))
    root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),points=points,
        pareto_checkpoint_ids=[p['id'] for p in frontier],efficiency_bands=bands,errors=errors,test_evaluated=False,
        note='Best saved checkpoints, including provisional runs. Sparsity and efficiency always refer to the same coefficients. Loss bands are descriptive, not accepted degradation budgets.')
    write_json(root/'compression_frontier_summary.json',result)
    lookup={p['id']:p for p in points}
    lines=['# Efficiency versus sparsity: best saved checkpoint frontier','', 'Updated UTC: '+result['updated_utc'],'',
        'Exact nonzero coefficients include biases. All checkpoints use the same validation data and 1% background working point. Some source training runs remain unfinished; this is the best known frontier, not a claim of numerical convergence or a test result.',
        '', '| Reference | Efficiency loss band (pp) | Candidate | Nonzero coefficients | Dense slots | Signal efficiency | Sparse path MAC bound | Source run converged |',
        '|---|---:|---|---:|---:|---:|---:|---|']
    for band in bands:
        p=lookup.get(band['candidate'])
        if p:lines.append(f"| {band['reference']} | {band['loss_percentage_points']:.2f} | {p['label']} | {p['nonzero_coefficients']} | {p['dense_slots']} | {p['efficiency']:.2%} | {p['mac_bound']} | {p['run_converged']} |")
    lines+=['','Loss bands are descriptive alternatives; no loss tolerance is silently selected. Retained frontier and band-winning checkpoints are copied by SHA-256 under models/compression_frontier/ so later training cannot overwrite these exact weights.',
        '', 'Nonzero-count and MAC savings require a sparsity-aware inference implementation. They are not measured FPGA resources or latency. The existing FPGA handoff remains unchanged.']
    if errors:lines+=['','Read errors: '+json.dumps(errors)]
    (root/'compression_frontier_report.md').write_text('\n'.join(lines)+'\n')
    for name in ['compression_frontier_report.md','compression_frontier_summary.json']:syncer.register_file(root/name)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    colors=dict(frozen='black',architecture_scan='tab:blue',l1_path='tab:green')
    for source in colors:
        for converged in [True,False]:
            subset=[p for p in points if p['source']==source and p['run_converged']==converged]
            if not subset:continue
            for ax in axes:ax.scatter([p['nonzero_coefficients'] for p in subset],[100*p['efficiency'] for p in subset],
                facecolors=colors[source] if converged else 'none',edgecolors=colors[source],marker='o',s=35,
                label=source+(' (run complete)' if converged else ' (provisional run)'))
    for ax in axes:
        ax.plot([p['nonzero_coefficients'] for p in frontier],[100*p['efficiency'] for p in frontier],color='.5',lw=1)
        ax.axhline(100*reference,color='tab:red',ls='--',lw=1,label='XGBoost')
        ax.axhline(100*frozen['efficiency'],color='black',ls=':',lw=1,label='frozen winner')
        ax.set_xlabel('Exact nonzero scalar coefficients, including biases');ax.set_ylabel('Signal efficiency at 1% background [%]');ax.grid(alpha=.2)
    axes[0].set_xscale('log');axes[0].set_title('Full tradeoff');axes[1].set_ylim(100*reference-1.05,max(100*reference,100*max(p['efficiency'] for p in points))+.35);axes[1].set_title('Near the reference working point')
    axes[0].legend(fontsize=7);fig.suptitle('Checkpoint-matched compression frontier | validation | scalar MAC bounds are reported in the table')
    fig.tight_layout();fig.savefig(root/'compression_frontier.png',dpi=180);plt.close(fig)
    print('Compression frontier:',len(points),'matched checkpoints;',len(frontier),'nondominated',flush=True)
    return result


if __name__=='__main__':publish()
