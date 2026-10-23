"""Run isolated training studies in order, measuring each before the next."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,fcntl,json,subprocess,time
from datetime import datetime,timezone
from data_utils import resolve,write_json
import common.syncer as syncer


def command(args):
    return subprocess.run(args,cwd=resolve('.'),text=True,capture_output=True,check=True).stdout.strip()


def complete(scan,index):
    marker=resolve(f"results/{scan['scan_name']}_member_{index}.json")
    row=json.loads(marker.read_text()) if marker.exists() else {}
    return bool(row.get('complete') and row.get('converged'))


def advance():
    policy=json.loads(resolve('configs/tao_training_improvements_v1.json').read_text())
    state_path=resolve('results/tao_improvements_controller_state.json')
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(submitted={},summaries={})
    # Save each submission immediately; a restart never blindly duplicates work.
    def save():write_json(state_path,state)
    progress=[];current=None;all_finished=True
    for phase in policy['phases']:
        scans=[json.loads(resolve(path).read_text()) for path in phase['configs']]
        done=sum(complete(scan,i) for scan in scans for i in range(len(scan['layouts'])))
        total=sum(len(scan['layouts']) for scan in scans)
        report=resolve('plots/comparisons/'+phase['report']+'_summary.json')
        measured=report.exists() and json.loads(report.read_text()).get('all_training_converged',False)
        progress.append(dict(name=phase['name'],converged=done,total=total,measured=measured))
        if measured:continue
        all_finished=False;current=phase['name']
        for path,scan in zip(phase['configs'],scans):
            submitted=state['submitted'].setdefault(path,[])
            missing=[i for i in range(len(scan['layouts'])) if not complete(scan,i) and i not in submitted]
            if missing:
                # The first pilots are measured before expanding the split-search scan.
                if phase['name']=='hard_search':
                    pilot=resolve('results/qg_vs_wzt_hardtao_linear_d3_n1_sa1e-05_la1e-05_seed42_hard_search_v1_history.json')
                    if not pilot.exists() or not json.loads(pilot.read_text()):continue
                job=command(['sbatch','--parsable','--array='+','.join(map(str,missing)),
                    '--job-name='+scan['scan_name'],'scripts/train_tao_study.slurm',path])
                submitted.extend(missing);state.setdefault('jobs',[]).append(dict(config=path,indices=missing,job_id=job));save()
                print('Submitted',path,missing,job,flush=True)
        if done==total and phase['name']=='hard_search':
            command([sys.executable,'scripts/submit_tao_hard_search_summary.py'])
            receipt=resolve('results/tao_hard_search_v1_summary_job.json')
            if receipt.exists():
                state['summaries'][phase['name']]=json.loads(receipt.read_text())['job_id'];save()
        elif done==total and phase['name'] not in state['summaries']:
            args=['--config',phase.get('baseline',phase['configs'][0]),'--output-prefix',phase['report']]
            extras=phase['configs'] if phase.get('baseline') else phase['configs'][1:]
            for path in extras:args.extend(['--extra-config',path])
            if phase.get('include_linear'):args.append('--include-linear')
            if phase.get('focus_layouts'):args.extend(['--focus-layouts',phase['focus_layouts']])
            job=command(['sbatch','--parsable','scripts/summarize_tao_study.slurm']+args)
            state['summaries'][phase['name']]=job;save();print('Submitted measurement',phase['name'],job,flush=True)
        break
    queue=command(['squeue','-r','-h','-u','robert.schoefbeck','-o','%i|%T|%j'])
    jobs=[line.split('|',2) for line in queue.splitlines() if line]
    attention=[]
    if current:
        phase=next(p for p in policy['phases'] if p['name']==current)
        for path in phase['configs']:
            scan=json.loads(resolve(path).read_text());names={scan['scan_name']}
            if current=='hard_search':names.add('tao-hard-search')
            for index in state['submitted'].get(path,[]):
                if not complete(scan,index) and not any(name in names and job.endswith('_'+str(index)) for job,_,name in jobs):
                    attention.append(dict(config=path,index=index,reason='No active training job; inspect logs and scheduler history'))
        if current in state['summaries'] and not any(job==state['summaries'][current] for job,_,_ in jobs):
            attention.append(dict(phase=current,reason='No active summary job and report not ready; inspect summary log'))
    report=dict(attention=attention,updated_utc=datetime.now(timezone.utc).isoformat(),current_phase=current,finished=all_finished,
        progress=progress,scheduler=queue.splitlines(),submissions=state)
    root=resolve('plots/comparisons');root.mkdir(parents=True,exist_ok=True)
    write_json(root/'training_improvements_status.json',report)
    lines=['# Training improvements: staged studies','', 'Updated UTC: '+report['updated_utc'],'',
        'Current phase: '+str(current),'', '| Study | Converged models | Measured |','|---|---:|---|']
    for row in progress:lines.append(f"| {row['name']} | {row['converged']}/{row['total']} | {row['measured']} |")
    if attention:lines+=['','Attention: '+json.dumps(attention)]
    lines+=['',policy.get('strategy','Studies run in order: split proposals, initialization, sequential growth to two then three trees, and L1 penalties.'),
        'Each phase uses the same exported train/validation arrays and must converge and produce its comparison before the next starts. Test data remain unused.',
        '', 'Scheduler snapshot:', '', '```', queue, '```']
    (root/'training_improvements_status.md').write_text('\n'.join(lines)+'\n')
    for name in ['training_improvements_status.md','training_improvements_status.json']:syncer.register_file(root/name)
    syncer.sync();print(json.dumps(dict(updated_utc=report['updated_utc'],current_phase=current,finished=all_finished,progress=progress)),flush=True)
    return all_finished


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    with resolve('results/tao_improvements_monitor.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Improvement monitor already running')
        while True:
            try:
                if advance() or args.once:return
            except Exception as exc:
                print(datetime.now(timezone.utc).isoformat(),type(exc).__name__,str(exc),flush=True)
                if args.once:raise
            time.sleep(120)


if __name__=='__main__':main()
