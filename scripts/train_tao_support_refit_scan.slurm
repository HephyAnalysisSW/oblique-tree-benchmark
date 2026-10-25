#!/bin/bash
#SBATCH --job-name=tao-support-scan
#SBATCH --partition=c
#SBATCH --qos=c_rapid
#SBATCH --account=cms
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --chdir=/users/robert.schoefbeck/CMS/tree_benchmark
#SBATCH --output=results/tao-support-scan-%j.log
set -eo pipefail
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda activate tree-benchmark
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set +e
/usr/bin/time -v -o "results/support_refit_scan_${SLURM_JOB_ID}_resources.txt" python -u -B scripts/run_support_refit_scan_case.py --config "$1" --wall-seconds 2700
refit_status=$?
set -e
python - "$1" "$SLURM_JOB_ID" "$refit_status" <<'PY'
import datetime,json,pathlib,sys
config,job,code=sys.argv[1:];plan=json.loads(pathlib.Path(config).read_text())
path=pathlib.Path('results/'+plan['run_name']+'_job_'+job+'.json');tmp=path.with_suffix('.tmp')
tmp.write_text(json.dumps(dict(config=config,job_id=job,exit_code=int(code),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n');tmp.replace(path)
PY
exit "$refit_status"
