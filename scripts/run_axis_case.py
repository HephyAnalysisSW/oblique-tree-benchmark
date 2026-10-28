"""Run one isolated axis training/refit allocation; the monitor handles continuations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import fcntl
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def prepare_refit(case):
    path=ROOT/case['refit_config']
    source=ROOT/('models/'+case['run_name']+'.pkl')
    if not source.exists():raise ValueError('No converged terminal parent')
    import pickle
    parent=pickle.loads(source.read_bytes())
    from tao.axis import assert_axis
    assert_axis(parent['model'])
    if parent.get('training_converged') is not True:raise ValueError('Parent not converged')
    plan=dict(run_name=case['refit_run_name'],source_checkpoint=str(source.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_config='configs/bdt_qg_vs_wzt_seed42.yaml',stages=['leaf_only'],exact_zero_support=True,
        minimum_epochs=5,max_epochs=200,local_leaf_max_iter=400,global_leaf_chunk_iter=400,
        max_global_chunks=100,split_max_iter=800,leaf_ftol=1e-12,leaf_gtol=1e-8,
        kkt_tol=1e-6,loss_rtol=1e-5,route_tol=1e-4,separation_time_limit=600)
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Frozen refit parent/policy changed')
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(plan,indent=2)+'\n')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=int,required=True);parser.add_argument('--stage',choices=['train','refit'],required=True)
    args=parser.parse_args();scan=json.loads((ROOT/'configs/axis_linear_v1.json').read_text());case=scan['cases'][args.index]
    result=ROOT/f'results/axis_linear_v1/case_{args.index}_{args.stage}_exit.json';result.parent.mkdir(parents=True,exist_ok=True)
    with (result.with_suffix('.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.stage=='train':
                command=[sys.executable,'-u','scripts/train_axis_tao.py','--config',case['config'],'--index','0','--wall-seconds','2700']
            else:
                prepare_refit(case)
                command=[sys.executable,'-u','scripts/run_support_refit_scan_case.py','--config',case['refit_config'],'--wall-seconds','2700']
            code=subprocess.run(command,cwd=ROOT).returncode
        except Exception:
            result.write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),exit_code=1))+'\n');raise
        temp=result.with_suffix('.tmp');temp.write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),exit_code=code))+'\n');temp.replace(result)
        return code

if __name__=='__main__':sys.exit(main())
