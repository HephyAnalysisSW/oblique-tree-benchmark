#!/bin/bash
#SBATCH --job-name=axis-linear
#SBATCH --partition=c
#SBATCH --qos=c_rapid
#SBATCH --account=cms
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/users/robert.schoefbeck/CMS/tree_benchmark
#SBATCH --output=results/axis-linear-%j.log
set -eo pipefail
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda activate tree-benchmark
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python -u -B scripts/run_axis_case.py --index "$1" --stage "$2"
