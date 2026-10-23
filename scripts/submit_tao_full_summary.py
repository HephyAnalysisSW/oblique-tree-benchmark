"""Submit the comparison once, after every layout has passed convergence checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import fcntl,hashlib,json,subprocess
from data_utils import resolve,write_json

scan=json.loads(resolve('configs/tao_full_training_v1.json').read_text())
fingerprint=hashlib.sha256(json.dumps(scan,sort_keys=True).encode()).hexdigest()
receipt=resolve('results/'+scan['scan_name']+'_summary_job.json')
with resolve('results/'+scan['scan_name']+'_summary.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    complete=True
    for i in range(len(scan['layouts'])):
        marker=resolve(f"results/{scan['scan_name']}_member_{i}.json")
        complete=complete and marker.exists()
        if marker.exists():
            r=json.loads(marker.read_text());complete=complete and r['converged'] and r['complete']
    previous=json.loads(receipt.read_text()) if receipt.exists() else {}
    if complete and previous.get('policy_sha256')!=fingerprint:
        result=subprocess.run(['sbatch','--parsable','scripts/summarize_tao_full_training.slurm'],
            cwd=resolve('.'),text=True,capture_output=True,check=True)
        write_json(receipt,dict(job_id=result.stdout.strip(),scan_name=scan['scan_name'],policy_sha256=fingerprint))
        print('Submitted full-training comparison:',result.stdout.strip(),flush=True)
    else:print('Waiting for remaining converged layouts; no comparison submitted.',flush=True)

subprocess.run([sys.executable,'scripts/publish_tao_full_status.py'],cwd=resolve('.'),check=True)
