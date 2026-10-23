"""Run one resumable scan case and record its exit for the rolling scheduler."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,json,os,subprocess
from datetime import datetime,timezone
from data_utils import resolve,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True,type=int);args=p.parse_args()
    scan=json.loads(resolve('configs/tao_hyper_scan_v1.json').read_text());case=scan['cases'][args.index]
    result=subprocess.run([sys.executable,'-u','-B','scripts/train_tao_fully.py','--config',case['config'],'--index','0'],cwd=resolve('.'))
    path=resolve(f"results/hyper_scan_v1/case_{args.index:02d}_exit.json");path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,dict(index=args.index,job_id=os.environ.get('SLURM_JOB_ID'),exit_code=result.returncode,
        updated_utc=datetime.now(timezone.utc).isoformat()))
    return 0 if result.returncode in [0,75] else result.returncode


if __name__=='__main__':sys.exit(main())
