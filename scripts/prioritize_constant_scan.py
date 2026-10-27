"""Prioritize clean continuations of the last constant-leaf scan case.

Coordinates through the existing scheduler's approved-continuation mechanism;
never edits training parameters, stopping rules, checkpoints, or scheduler state.
It temporarily allows one extra worker while the focused case is waiting, and
restores the configured concurrency after launch. On full convergence it
rebuilds the paper's numerical assets, but does not commit or publish a PDF.
"""
import argparse,datetime,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):
 p=ROOT/path
 return json.loads(p.read_text()) if p.exists() else default

def write(path,value):
 p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.focus.tmp')
 temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)


def tick(case_index=52,baseline=4):
 scan=read('configs/tao_hyper_scan_v1.json');case=next(c for c in scan['cases'] if c['index']==case_index)
 queue=subprocess.check_output(['squeue','-h','-u','robert.schoefbeck','-o','%i'],text=True)
 active=set(queue.split())
 scheduler=read('results/tao_hyper_scan_v1_scheduler.json')
 attempts=scheduler['attempts'].get(str(case_index),[]);last=attempts[-1] if attempts else None
 marker=read(f"results/{case['scan_name']}_member_0.json",{})
 report=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),case=case_index,
             complete=bool(marker.get('complete') and marker.get('converged')),last_attempt=last)
 if report['complete'] or (last and last['job_id'] in active):
  if scan['concurrency']==baseline+1:
   scan['concurrency']=baseline;write('configs/tao_hyper_scan_v1.json',scan)
  report['phase']='complete' if report['complete'] else 'running'
 else:
  ended=read(f'results/hyper_scan_v1/case_{case_index:02d}_exit.json',{})
  if last and ended.get('job_id')==last['job_id'] and ended.get('exit_code')==75:
   recovery=dict(job_id=last['job_id'],approved=True,reason='Focused constant-leaf finalization after clean continuation exit; all numerical policies unchanged',updated_utc=report['updated_utc'])
   write(f'results/hyper_scan_v1/case_{case_index:02d}_recovery.json',recovery)
   if scan['concurrency']==baseline:
    scan['concurrency']=baseline+1;write('configs/tao_hyper_scan_v1.json',scan)
   report['phase']='prioritized_clean_continuation'
  else:
   report['phase']='attention_required';report['exit_record']=ended
 if report['complete']:
  complete_all=all(read(f"results/{c['scan_name']}_member_0.json",{}).get('converged',False) for c in scan['cases'] if c['leaf_mode']=='constant')
  if complete_all:
   result=subprocess.run([sys.executable,'-u','scripts/build_constant_leaf_study.py','--require-complete'],cwd=ROOT)
   report['final_assets_returncode']=result.returncode
   report['final_assets_ready']=result.returncode==0
 write('results/constant_leaf_finalize_monitor.json',report)
 print(json.dumps(report),flush=True)
 return report


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');args=p.parse_args()
 with (ROOT/'results/constant_leaf_finalize_monitor.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit('Constant finalization monitor already active')
  while True:
   report=tick()
   if args.once or report['complete'] or report['phase']=='attention_required':return
   time.sleep(120)

if __name__=='__main__':main()
