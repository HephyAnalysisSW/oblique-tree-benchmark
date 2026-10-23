#!/bin/bash
#SBATCH --job-name=tao-constant-full
#SBATCH --partition=c
#SBATCH --qos=c_rapid
#SBATCH --account=cms
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --array=0-6
#SBATCH --chdir=/users/robert.schoefbeck/CMS/tree_benchmark
#SBATCH --output=results/tao-constant-full-%A_%a.log
set -eo pipefail
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda activate tree-benchmark
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export X509_USER_PROXY=/users/robert.schoefbeck/.private/.proxy CERN_USER=schoef
set +e
python -u -B scripts/train_tao_fully.py --config configs/tao_constant_full_training_v1.json --index "$SLURM_ARRAY_TASK_ID"
tao_exit_status=$?
set -e
if [ "$tao_exit_status" -eq 75 ]; then
    sbatch --array="$SLURM_ARRAY_TASK_ID" scripts/train_tao_constant_fully.slurm
elif [ "$tao_exit_status" -eq 0 ]; then
    python -u -B scripts/submit_tao_constant_summary.py
else
    exit "$tao_exit_status"
fi
