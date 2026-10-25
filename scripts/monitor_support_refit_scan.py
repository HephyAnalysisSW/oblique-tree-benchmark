"""Bounded, resumable scan controller; scientifically unsuccessful runs stay visible."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,fcntl,json,os,socket,subprocess,time
from datetime import datetime,timezone
from data_utils import resolve,write_json
from scripts.train_tao_support_refit import atomic_json
import common.syncer as syncer
import matplotlib.pyplot as plt

TERMINAL={'complete','separated','inconclusive','stalled'}


def read(path,default=None):
    path=resolve(path);return json.loads(path.read_text()) if path.exists() else default


def schedule(plan,state,status,jobs,receipt,submit):
    """Submit never-started cases or clean budget continuations; never auto-retry errors."""
    active={j['name'] for j in jobs};slots=plan['concurrency']-len(jobs);launched=[];attention=[];eligible=[]
    for case in plan['cases']:
        name=case['id'];phase=status[name].get('phase','not_started')
        if case['job_name'] in active or phase in TERMINAL:continue
        prior=state['submissions'].get(name)
        if case.get('adopted_pilot'):
            attention.append(dict(case=name,reason='Pilot absent from queue before terminal status; inspect its self-resubmitting wrapper'));continue
        if prior:
            result=receipt(case,prior['job_id'])
            if result is None:
                attention.append(dict(case=name,job_id=prior['job_id'],reason='No active job or exit receipt; inspect Slurm accounting'));continue
            if result['exit_code']!=75:
                attention.append(dict(case=name,job_id=prior['job_id'],reason='Non-budget exit without terminal worker status',exit_code=result['exit_code']));continue
        eligible.append(case)
    eligible.sort(key=lambda c:state['submissions'].get(c['id'],{}).get('allocations',1 if c['id'] in state['submissions'] else 0))
    for case in eligible[:max(0,slots)]:
        name=case['id'];prior=state['submissions'].get(name)
        allocations=prior.get('allocations',1) if prior else 0
        job=submit(case)
        state['submissions'][name]=dict(job_id=job,config=case['config'],allocations=allocations+1,submitted_utc=datetime.now(timezone.utc).isoformat())
        launched.append(dict(case=name,job_id=job));slots-=1
    return launched,attention


def publish(plan,status,jobs,attention):
    root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
    xgb=read('fpga_handoff/v2/manifest.json')['xgboost']['metrics']['signal_efficiency_at_fixed_fpr']
    rows=[];points=[]
    for case in plan['cases']:
        s=status[case['id']];row=dict(case=case,status=s,archived_best=read('results/'+case['run_name']+'_archived_best.json',{}));rows.append(row)
        for stage,item in s.get('stages',{}).items():
            points.append(dict(case=case['id'],stage=stage,efficiency=item['metrics']['validation']['signal_efficiency_at_fixed_fpr'],support=case['parent_nonzeros'],converged=True))
        history=s.get('history',[])
        if history and history[-1]['stage'] not in s.get('stages',{}):
            r=history[-1];points.append(dict(case=case['id'],stage=r['stage'],efficiency=r['metrics']['validation']['signal_efficiency_at_fixed_fpr'],support=case['parent_nonzeros'],converged=False))
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),models=rows,active_jobs=jobs,attention=attention,
        total=len(rows),completed=sum(r['status'].get('phase')=='complete' for r in rows),
        held=sum(r['status'].get('phase') in TERMINAL-{'complete'} for r in rows),xgboost_efficiency=xgb,test_evaluated=False)
    atomic_json(root/'support_refit_scan_v1_summary.json',result)
    lines=['# Frozen-support unpenalized refit scan','',result['updated_utc'],'',
        f"Completed {result['completed']}/{len(rows)}; held scientific/numerical outcomes: {result['held']}. Validation at 1% background; test untouched.",
        '', '| Parent selector | Architecture | Allocated | Support budget | Parent eff. | Leaf-only eff. | Full latest eff. | Best archived full eff. | Full passes | CPU min | Status |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    fmt=lambda v:'—' if v is None else f'{100*v:.2f}%'
    for row in rows:
        c,s=row['case'],row['status'];parent=s.get('parent',{}).get('metrics',{});leaf=s.get('stages',{}).get('leaf_only',{}).get('metrics',{})
        full=[r for r in s.get('history',[]) if r['stage']=='full'];last=full[-1] if full else {}
        best=row['archived_best'].get('full',{}).get('score',[None])[0]
        metrics=lambda m:m.get('validation',{}).get('signal_efficiency_at_fixed_fpr')
        lines.append('| '+' | '.join([c['id'],f"{c['trees']} × D{c['depth']} linear",str(c['trees']*527),str(c['parent_nonzeros']),
            fmt(metrics(parent)),fmt(metrics(leaf)),fmt(metrics(last.get('metrics',{}))),fmt(best),str(len(full)),
            f"{s.get('training_seconds',0)/60:.1f}",s.get('phase','not_started')])+' |')
    lines+=['','Parent and refit metrics use the same pinned support. Support includes biases; all three-tree models allocate 1,581 slots and the one-tree model 527. Leaf-only routing is fixed; full refits preserve the mask. Best archived scores have exact matching saved coefficients. Latest values from unfinished runs are provisional.',
        '', 'Convergence is numerical within the fixed support and configured proposals, not a global optimum. Separated, inconclusive and stalled runs remain explicitly held. No L2 fallback or support regrowth is used.',
        '', '![Tradeoff](support_refit_scan_v1_tradeoffs.png)']
    if attention:lines+=['','Attention: '+json.dumps(attention)]
    (root/'support_refit_scan_v1_report.md').write_text('\n'.join(lines)+'\n')
    if points:
        fig,ax=plt.subplots(figsize=(10,6))
        for stage,color in [('leaf_only','tab:orange'),('full','tab:blue')]:
            for converged in [True,False]:
                selected=[p for p in points if p['stage']==stage and p['converged']==converged]
                if selected:
                    ax.scatter([p['support'] for p in selected],[100*p['efficiency'] for p in selected],
                        facecolors=color if converged else 'none',edgecolors=color,s=50,label=stage+(' converged' if converged else ' provisional'))
                    for p in selected:ax.annotate(p['case'],(p['support'],100*p['efficiency']),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.axhline(100*xgb,color='black',ls='--',label='XGBoost');ax.set_xlabel('Frozen support budget, including biases')
        ax.set_ylabel('Signal efficiency at 1% background [%]');ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set_title('Frozen-support refits | validation | CPU times in report');fig.tight_layout();fig.savefig(root/'support_refit_scan_v1_tradeoffs.png',dpi=180);plt.close(fig)
    for name in ['support_refit_scan_v1_report.md','support_refit_scan_v1_summary.json']:syncer.register_file(root/name)
    syncer.sync();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    with resolve('results/support_refit_scan_v1_monitor.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Support-refit scan monitor already running')
        atomic_json('results/support_refit_scan_v1_monitor_process.json',dict(host=socket.gethostname(),pid=os.getpid()))
        while True:
            try:
                plan=read('configs/tao_support_refit_scan_v1.json');state=read('results/support_refit_scan_v1_controller.json',dict(submissions={}))
                status={c['id']:read('results/'+c['run_name']+'_status.json',dict(phase='not_started')) for c in plan['cases']}
                raw=subprocess.check_output(['squeue','-r','-h','-u','robert.schoefbeck','-o','%i|%T|%j'],text=True)
                jobs=[dict(zip(['job_id','state','name'],line.split('|',2))) for line in raw.splitlines()
                    if '|tao-sref-' in line or '|tao-support-pilot' in line]
                def receipt(case,job):return read('results/'+case['run_name']+'_job_'+job+'.json')
                def submit(case):
                    job=subprocess.check_output(['sbatch','--parsable','--job-name='+case['job_name'],
                        'scripts/train_tao_support_refit_scan.slurm',case['config']],text=True).strip().split(';')[0]
                    prior=state['submissions'].get(case['id'])
                    allocations=prior.get('allocations',1) if prior else 0
                    state['submissions'][case['id']]=dict(job_id=job,config=case['config'],allocations=allocations+1,submitted_utc=datetime.now(timezone.utc).isoformat())
                    atomic_json('results/support_refit_scan_v1_controller.json',state);return job
                launched,attention=schedule(plan,state,status,jobs,receipt,submit)
                atomic_json('results/support_refit_scan_v1_controller.json',state)
                result=publish(plan,status,jobs,attention)
                print(json.dumps(dict(updated_utc=result['updated_utc'],completed=result['completed'],held=result['held'],launched=launched,attention=attention)),flush=True)
                if all(s.get('phase') in TERMINAL for s in status.values()):
                    from scripts.publish_support_refit_results import main as finalize
                    finalize()
                    from scripts.deliver_support_refit_results import main as deliver
                    deliver();return
                if args.once:return
            except Exception as exc:
                print(datetime.now(timezone.utc).isoformat(),type(exc).__name__,str(exc),flush=True)
                if args.once:raise
            time.sleep(120)


if __name__=='__main__':main()
