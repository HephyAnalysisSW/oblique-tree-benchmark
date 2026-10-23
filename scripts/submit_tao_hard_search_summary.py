"""Submit hard-search-only and overall comparisons after all hard-search models converge."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import fcntl,json,subprocess
from data_utils import resolve,write_json
scan=json.loads(resolve('configs/tao_hard_search_v1.json').read_text())
receipt=resolve('results/'+scan['scan_name']+'_summary_job.json')
with resolve('results/'+scan['scan_name']+'_summary.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    complete=True
    for i in range(len(scan['layouts'])):
        marker=resolve(f"results/{scan['scan_name']}_member_{i}.json")
        row=json.loads(marker.read_text()) if marker.exists() else {}
        complete=complete and row.get('complete',False) and row.get('converged',False)
    if complete and not receipt.exists():
        result=subprocess.run(['sbatch','--parsable','scripts/summarize_tao_hard_search.slurm'],
            cwd=resolve('.'),text=True,capture_output=True,check=True)
        write_json(receipt,dict(job_id=result.stdout.strip(),scan_name=scan['scan_name']))
        print('Submitted hard-search and overall comparison:',result.stdout.strip(),flush=True)
    else:print('Waiting for remaining hard-search models, or summary already submitted.',flush=True)
