#!/bin/bash
#SBATCH --job-name=tao-support-pilot
#SBATCH --partition=c
#SBATCH --qos=c_rapid
#SBATCH --account=cms
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --chdir=/users/robert.schoefbeck/CMS/tree_benchmark
#SBATCH --output=results/tao-support-pilot-%j.log
set -eo pipefail
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda activate tree-benchmark
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set +e
/usr/bin/time -v -o "results/support_refit_pilot_${SLURM_JOB_ID}_resources.txt" python -u -B scripts/run_support_refit_pilot.py --config "$1" --wall-seconds 2700
refit_exit_status=$?
set -e
if [ "$refit_exit_status" -eq 75 ]; then
    sbatch scripts/train_tao_support_refit_pilot.slurm "$1"
elif [ "$refit_exit_status" -ne 0 ]; then
    exit "$refit_exit_status"
fi
