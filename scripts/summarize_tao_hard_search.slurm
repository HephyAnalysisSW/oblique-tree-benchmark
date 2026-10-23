#!/bin/bash
#SBATCH --job-name=tao-hard-search-figures
#SBATCH --partition=c
#SBATCH --qos=c_rapid
#SBATCH --account=cms
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --chdir=/users/robert.schoefbeck/CMS/tree_benchmark
#SBATCH --output=results/tao-hard-search-summary-%j.log
set -eo pipefail
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda activate tree-benchmark
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export X509_USER_PROXY=/users/robert.schoefbeck/.private/.proxy CERN_USER=schoef
python -u -B scripts/summarize_tao_full_training.py --config configs/tao_hard_search_v1.json
python -u -B scripts/summarize_tao_full_training.py --config configs/tao_hard_search_v1.json --include-linear
