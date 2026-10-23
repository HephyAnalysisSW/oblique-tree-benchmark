"""Lightweight persistent monitor; publishes checkpoints without changing training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,fcntl,json,subprocess,time
from datetime import datetime,timezone
from data_utils import resolve,write_json
import common.syncer as syncer


def command(args):
    return subprocess.run(args,cwd=resolve('.'),text=True,capture_output=True,check=True).stdout.strip()


def snapshot():
    scan=json.loads(resolve('configs/tao_constant_full_training_v1.json').read_text())
    queue=command(['squeue','-r','-h','-u','robert.schoefbeck','-o','%i|%T|%j'])
    jobs=[dict(zip(['job_id','state','name'],line.split('|',2))) for line in queue.splitlines() if line]
    active=[j for j in jobs if j['name']=='tao-constant-full']
    rows=[]
    for i,layout in enumerate(scan['layouts']):
        run=(f"{scan['task']}_hardtao_constant_d{layout['depth']}_n{layout['trees']}"
             f"_sa{scan['split_alpha']:g}_la{scan['leaf_alpha']:g}_seed{scan['seed']}_{scan['run_suffix']}")
        path=resolve('results/'+run+'_status.json')
        status=json.loads(path.read_text()) if path.exists() else dict(phase='not started',epochs=0,stable_epochs=0,converged=False)
        history=resolve('results/'+run+'_history.json')
        records=json.loads(history.read_text()) if history.exists() else []
        last=records[-1] if records else {}
        own_jobs=[j for j in active if j['job_id'].endswith('_'+str(i))]
        rows.append(dict(index=i,layout=layout,status=status,jobs=own_jobs,validation=last.get('validation'),
            attention_needed=not status['converged'] and not own_jobs))
    complete=all(r['status']['converged'] for r in rows)
    if complete:
        command([sys.executable,'scripts/submit_tao_constant_summary.py'])
    receipt=resolve('results/tao_constant_full_training_v1_summary_job.json')
    summary_job=json.loads(receipt.read_text()) if receipt.exists() else None
    reports={}
    for prefix in ['constant_full_training','overall_training']:
        path=resolve('plots/comparisons/'+prefix+'_summary.json')
        reports[prefix]=path.exists() and json.loads(path.read_text()).get('all_training_converged',False)
    # A job can be between its terminal checkpoint and successor submission.
    # Report attention only after the next monitor cycle confirms the gap.
    return dict(updated_utc=datetime.now(timezone.utc).isoformat(),models=rows,all_converged=complete,
        active_jobs=active,summary_job=summary_job,reports_ready=reports,
        finished=complete and all(reports.values()),test_evaluated=False)


def publish(report):
    root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
    lines=['# Constant-leaf TAO job monitor','', 'Updated UTC: '+report['updated_utc'],'',
        'Checkpoint results are provisional until convergence. Test data remain unused.','',
        '| Trees | Depth | Phase | Passes | Stable passes | AUC | Efficiency at 1% FPR | Job |',
        '|---:|---:|---|---:|---:|---:|---:|---|']
    for row in report['models']:
        d=row['status'];m=row['validation'];layout=row['layout']
        auc=f"{m['signal_auc']:.6f}" if m else '—'
        eff=f"{m['signal_efficiency_at_fixed_fpr']:.2%}" if m else '—'
        jobs=', '.join(j['job_id']+' '+j['state'] for j in row['jobs']) or ('converged' if d['converged'] else 'no active job; checking')
        lines.append(f"| {layout['trees']} | {layout['depth']} | {d['phase']} | {d['epochs']} | {d['stable_epochs']} | {auc} | {eff} | {jobs} |")
    lines+=['','Converged: '+str(sum(r['status']['converged'] for r in report['models']))+'/7.',
        '','The monitor checks scheduler jobs and saved checkpoints every two minutes. Training workers resume at scheduler budgets. No convergence thresholds or model settings are changed.',
        '', 'Final reports: [constant leaves](constant_full_training_report.md), [overall parameter table](overall_training_report.md). These links become available after all models converge.',
        '', 'Reports ready: '+json.dumps(report['reports_ready'])]
    (root/'constant_training_status.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'constant_training_status.json',report)
    for name in ['constant_training_status.md','constant_training_status.json']:syncer.register_file(root/name)
    syncer.sync()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    with resolve('results/tao_constant_monitor.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another constant-leaf monitor is already active')
        while True:
            try:
                report=snapshot();publish(report)
                print(json.dumps(dict(updated_utc=report['updated_utc'],converged=sum(r['status']['converged'] for r in report['models']),
                    active_jobs=len(report['active_jobs']),attention_indices=[r['index'] for r in report['models'] if r['attention_needed']],
                    reports_ready=report['reports_ready'])),flush=True)
                if report['finished'] or args.once:return
            except Exception as exc:
                print(datetime.now(timezone.utc).isoformat(),type(exc).__name__,str(exc),flush=True)
                if args.once:raise
            time.sleep(120)


if __name__=='__main__':main()
