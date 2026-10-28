"""Continue clean axis allocations, then frozen-leaf refits; never retry failures blindly."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):
    path=ROOT/path
    return json.loads(path.read_text()) if path.exists() else default


def save(path,value):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def status(case):
    plan=read(case['config'])
    trained=read(f"results/{plan['scan_name']}_member_0.json",{})
    refit=read('results/'+case['refit_run_name']+'_status.json',{})
    if refit.get('converged'):return 'complete'
    if refit.get('phase') in ['separated','inconclusive','stalled']:return 'held'
    return 'refit' if trained.get('complete') and trained.get('converged') else 'train'


def tick():
    scan=read('configs/axis_linear_v1.json');state=read('results/axis_linear_v1/scheduler.json',dict(attempts={}))
    active=set(subprocess.check_output(['squeue','-h','-u','robert.schoefbeck','-o','%i'],text=True).split())
    owned={a['job_id'] for group in state['attempts'].values() for a in group}
    free=max(0,scan['concurrency']-len(active&owned));attention=[];phases=[];launched=[]
    for case in scan['cases']:
        phase=status(case);phases.append(dict(index=case['index'],phase=phase))
        if phase in ['complete','held']:continue
        if any(a['job_id'] in active for key,group in state['attempts'].items() if key.startswith(str(case['index'])+':') for a in group):continue
        key=f"{case['index']}:{phase}";attempts=state['attempts'].get(key,[])
        if attempts:
            ended=read(f"results/axis_linear_v1/case_{case['index']}_{phase}_exit.json",{})
            if ended.get('job_id')!=attempts[-1]['job_id'] or ended.get('exit_code')!=75:
                attention.append(dict(index=case['index'],stage=phase,reason='No matching clean continuation exit',exit_record=ended));continue
        if free:
            job=subprocess.check_output(['sbatch','--parsable','scripts/train_axis_case.slurm',str(case['index']),phase],cwd=ROOT,text=True).strip().split(';')[0]
            state['attempts'].setdefault(key,[]).append(dict(job_id=job,submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            save('results/axis_linear_v1/scheduler.json',state);free-=1;launched.append(dict(index=case['index'],stage=phase,job_id=job))
    report=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phases=phases,launched=launched,attention=attention,
        active_job_ids=sorted(active&owned),terminal=all(p['phase'] in ['complete','held'] for p in phases))
    save('results/axis_linear_v1/monitor.json',report)
    print(json.dumps(report),flush=True)
    if report['terminal']:
        subprocess.run([sys.executable,'scripts/summarize_axis_study.py'],cwd=ROOT,check=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');args=p.parse_args()
    directory=ROOT/'results/axis_linear_v1';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'monitor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last_summary=time.time()
        while True:
            report=tick()
            if args.once or report['terminal']:return
            if time.time()-last_summary>=600:
                with (ROOT/'build/axis_summary_monitor.log').open('a') as log:
                    result=subprocess.run([sys.executable,'scripts/summarize_axis_study.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:print('Axis summary failed; see build/axis_summary_monitor.log',flush=True)
                last_summary=time.time()
            time.sleep(60)

if __name__=='__main__':main()
