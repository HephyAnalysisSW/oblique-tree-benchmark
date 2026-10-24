"""Warm-start three L1 regularization paths from the exact frozen FPGA winner."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,fcntl,hashlib,json,os,socket,subprocess,time
from datetime import datetime,timezone
import matplotlib.pyplot as plt
from data_utils import resolve,write_json
from tao.portable import restore_model
from tao.sparsity import sparsity_profile
import common.syncer as syncer


def read(path,default=None):
    path=resolve(path);return json.loads(path.read_text()) if path.exists() else default


def command(args):
    return subprocess.run(args,cwd=resolve('.'),text=True,capture_output=True,check=True).stdout.strip()


def case(config,branch,multiplier):
    cfg=read(config);run=(f"{cfg['task']}_hardtao_linear_d4_n3_sa{cfg['split_alpha']:g}_la{cfg['leaf_alpha']:g}_seed42_{cfg['run_suffix']}")
    status=read('results/'+run+'_status.json',dict(phase='not started',epochs=0,training_seconds=0.,converged=False))
    history=read('results/'+run+'_history.json',[]);marker=read(f"results/{cfg['scan_name']}_member_0.json",{})
    best=max(history,key=lambda r:(r['validation']['signal_efficiency_at_fixed_fpr'],r['validation']['signal_auc'])) if history else None
    return dict(best_checkpoint=dict(epoch=best['epoch'],validation=best['validation'],sparsity=best.get('sparsity')) if best else None,config=config,scan_name=cfg['scan_name'],run_name=run,branch=branch,multiplier=multiplier,
        split_alpha=cfg['split_alpha'],leaf_alpha=cfg['leaf_alpha'],status=status,
        complete=bool(marker.get('complete') and marker.get('converged')),
        latest=history[-1] if history else None,best_efficiency=max((r['validation']['signal_efficiency_at_fixed_fpr'] for r in history),default=None))


def schedule(plan,state,rows,jobs):
    """At most one point per branch; a completed terminal model seeds the next."""
    active_names={j['name'] for j in jobs};active_count=len(jobs);launched=[]
    def submit(row):
        nonlocal active_count
        if row['config'] in state['submissions'] or active_count>=plan['concurrency']:return
        job=command(['sbatch','--parsable','--array=0','--job-name='+row['scan_name'],'scripts/train_tao_study.slurm',row['config']])
        state['submissions'][row['config']]=dict(job_id=job,submitted_utc=datetime.now(timezone.utc).isoformat())
        path=resolve('results/tao_l1_path_v1_controller.json');tmp=path.with_suffix('.tmp');write_json(tmp,state);tmp.replace(path)
        launched.append(dict(config=row['config'],job_id=job));active_count+=1
    control=rows[0]
    if not control['complete']:submit(control)
    elif control['scan_name'] not in active_names:
        for branch in plan['paths']:
            prior=control
            for row in [r for r in rows if r['branch']==branch]:
                if row['complete']:prior=row;continue
                if prior['complete'] and prior['scan_name'] not in active_names:submit(row)
                break
    attention=[]
    for row in rows:
        if row['config'] in state['submissions'] and not row['complete'] and row['scan_name'] not in active_names and not any(r['config']==row['config'] for r in launched):
            attention.append(dict(config=row['config'],reason='No active job; inspect training log and Slurm history'))
    return launched,attention


def publish(plan,rows,jobs,launched,attention):
    path=resolve(plan['frozen_model']);assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['frozen_model_sha256']
    specification=read(path);baseline=sparsity_profile(restore_model(specification))
    manifest=read(plan['frozen_manifest']);frozen=next(c for c in manifest['candidates'] if c['id']==plan['candidate_id'])
    reference=frozen['metrics']['signal_efficiency_at_fixed_fpr'];xgb=manifest['xgboost']['metrics']['signal_efficiency_at_fixed_fpr']
    root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),plan=plan,models=rows,active_jobs=jobs,
        launched=launched,attention=attention,frozen_baseline=dict(validation=frozen['metrics'],sparsity=baseline),
        xgboost_efficiency=xgb,completed=sum(r['complete'] for r in rows),total=len(rows),test_evaluated=False)
    write_json(root/'l1_path_v1_summary.json',result)
    lines=['# L1 compression paths from the frozen FPGA winner','', 'Updated UTC: '+result['updated_utc'],'',
        f"Frozen winner efficiency: {100*reference:.2f}% at 1% background; {baseline['dense_parameter_slots']} dense slots; {baseline['nonzero_coefficients']} nonzero coefficients. XGBoost: {100*xgb:.2f}%.",
        '', 'The same-penalty control continues the exact frozen snapshot to convergence. Each stronger point starts from the previous converged terminal model. The control and paths do not modify the handoff.',
        '', '| Path | Multiplier | Split L1 | Leaf L1 | Latest efficiency | Best observed | Best-point nonzeros | Latest nonzeros | Reachable nonzeros | Path MAC bound | Passes | Training min | Status |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        last=row['latest'];s=last.get('sparsity') if last else None;eff=last['validation']['signal_efficiency_at_fixed_fpr'] if last else None
        fmt=lambda value:'—' if value is None else str(value)
        status='converged' if row['complete'] else row['status']['phase']+' (provisional)'
        lines.append('| '+' | '.join([row['branch'],str(row['multiplier']),f"{row['split_alpha']:g}",f"{row['leaf_alpha']:g}",
            '—' if eff is None else f'{100*eff:.2f}%', '—' if row['best_efficiency'] is None else f"{100*row['best_efficiency']:.2f}%",
            fmt(row['best_checkpoint']['sparsity']['nonzero_coefficients'] if row['best_checkpoint'] and row['best_checkpoint']['sparsity'] else None),
            fmt(s['nonzero_coefficients'] if s else None),fmt(s['reachable_nonzero_coefficients'] if s else None),fmt(s['selected_path_multiplications_upper_bound'] if s else None),
            str(row['status']['epochs']),f"{row['status']['training_seconds']/60:.1f}",status])+' |')
    if attention:lines+=['','Attention: '+json.dumps(attention)]
    lines+=['','Latest sparsity/MAC counts correspond to the latest evaluated checkpoint. Best-point nonzeros correspond to the saved best-efficiency checkpoint; do not mix the two when judging compression.', '', 'Dense tensor dimensions remain fixed at 1,581 slots. Sparsity counts exact zeros and includes biases. Reachability removes only algebraically constant routing; no validation/test examples are used to declare a branch dead.',
        '', 'MAC counts are sparse scalar arithmetic bounds, not FPGA timing or DSP measurements. Training times are additional compute after the frozen winner. The selected best-efficiency checkpoint is retained separately; terminal points are required before advancing a path.',
        '', 'Training remains cross-entropy plus L1, with L1-logistic split proposals. No least-squares Lasso regression on class IDs is used. AUC remains secondary.']
    (root/'l1_path_v1_report.md').write_text('\n'.join(lines)+'\n')
    for name in ['l1_path_v1_report.md','l1_path_v1_summary.json']:syncer.register_file(root/name)
    available=[r for r in rows if r['latest'] and r['latest'].get('sparsity')]
    if available:
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        control=next((row for row in available if row['branch']=='control'),None)
        if control:
            efficiency=100*control['latest']['validation']['signal_efficiency_at_fixed_fpr'];count=control['latest']['sparsity']['nonzero_coefficients']
            axes[0].scatter([1],[efficiency],marker='D',color='black',label='same-penalty control')
            axes[1].scatter([1],[count],marker='D',color='black')
            axes[2].scatter([count],[efficiency],marker='D',color='black')
        colors=dict(split='tab:blue',leaf='tab:orange',both='tab:green')
        for branch in plan['paths']:
            series=[r for r in available if r['branch']==branch]
            if not series:continue
            x=[r['multiplier'] for r in series];eff=[100*r['latest']['validation']['signal_efficiency_at_fixed_fpr'] for r in series]
            nz=[r['latest']['sparsity']['nonzero_coefficients'] for r in series];mac=[r['latest']['sparsity']['selected_path_multiplications_upper_bound'] for r in series]
            axes[0].plot(x,eff,'o-',color=colors[branch],label=branch)
            axes[1].plot(x,nz,'o-',color=colors[branch],label=branch)
            axes[2].plot(nz,eff,'o-',color=colors[branch],label=branch)
            for r,n,e in zip(series,nz,eff):
                axes[2].annotate(f"{r['status']['training_seconds']/60:.0f}m{' *' if r['complete'] else ''}",(n,e),xytext=(4,4),textcoords='offset points',fontsize=7)
        for ax in [axes[0],axes[2]]:
            ax.axhline(100*reference,color='black',ls=':',label='frozen winner');ax.axhline(100*xgb,color='grey',ls='--',label='XGBoost');ax.set_ylabel('Signal efficiency at 1% background [%]')
        axes[0].set_xscale('log');axes[0].set_xlabel('L1 multiplier');axes[1].set_xscale('log');axes[1].set_xlabel('L1 multiplier');axes[1].set_ylabel('Nonzero scalar coefficients')
        axes[2].set_xlabel('Nonzero scalar coefficients')
        for ax in axes:ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.suptitle('L1 compression paths | validation | labels: additional training time; * converged')
        fig.tight_layout();fig.savefig(root/'l1_path_v1_tradeoffs.png',dpi=180);plt.close(fig)
    syncer.sync();print(json.dumps(dict(updated_utc=result['updated_utc'],complete=result['completed'],total=len(rows),launched=launched,attention=attention)),flush=True)
    return result['completed']==len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    with resolve('results/tao_l1_path_v1_monitor.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('L1 path monitor already running')
        write_json(resolve('results/tao_l1_path_v1_monitor_process.json'),dict(host=socket.gethostname(),pid=os.getpid()))
        while True:
            try:
                plan=read('configs/tao_l1_path_v1.json');state=read('results/tao_l1_path_v1_controller.json',dict(submissions={}))
                rows=[case(plan['control_config'],'control',1)]
                for branch,points in plan['paths'].items():rows.extend(case(point['config'],branch,point['multiplier']) for point in points)
                queue=command(['squeue','-r','-h','-u','robert.schoefbeck','-o','%i|%T|%j'])
                jobs=[dict(zip(['job_id','state','name'],line.split('|',2))) for line in queue.splitlines() if line and '|tao_l1_path_v1_' in line]
                launched,attention=schedule(plan,state,rows,jobs)
                if publish(plan,rows,jobs,launched,attention) or args.once:return
            except Exception as exc:
                print(datetime.now(timezone.utc).isoformat(),type(exc).__name__,str(exc),flush=True)
                if args.once:raise
            time.sleep(120)


if __name__=='__main__':main()
