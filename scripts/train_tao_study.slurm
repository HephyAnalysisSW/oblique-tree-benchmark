#!/bin/bash
#SBATCH --job-name=tao-study
#SBATCH --partition=c
#SBATCH --qos=c_rapid
#SBATCH --account=cms
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --chdir=/users/robert.schoefbeck/CMS/tree_benchmark
#SBATCH --output=results/tao-study-%A_%a.log
set -eo pipefail
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda activate tree-benchmark
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export X509_USER_PROXY=/users/robert.schoefbeck/.private/.proxy CERN_USER=schoef
set +e
python -u -B scripts/train_tao_fully.py --config "$1" --index "$SLURM_ARRAY_TASK_ID"
tao_exit_status=$?
set -e
if [ "$tao_exit_status" -eq 75 ]; then
    sbatch --job-name="$SLURM_JOB_NAME" --array="$SLURM_ARRAY_TASK_ID" scripts/train_tao_study.slurm "$1"
elif [ "$tao_exit_status" -ne 0 ]; then
    exit "$tao_exit_status"
fi
