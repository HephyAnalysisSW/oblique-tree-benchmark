#!/bin/bash
#SBATCH --job-name=tree-bdt-retrain
#SBATCH --partition=c
#SBATCH --qos=c_rapid
#SBATCH --account=cms
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/users/robert.schoefbeck/CMS/tree_benchmark
#SBATCH --output=results/retrain_qg_vs_wzt_seed42.slurm-%j.log

set -eo pipefail
source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
conda activate tree-benchmark
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
hostname
python --version
python -B src/train_bdt.py --config configs/bdt_qg_vs_wzt_seed42_retrained.yaml
python -B src/compare_binary_runs.py \
  --reference-config configs/bdt_qg_vs_wzt_seed42.yaml \
  --candidate-config configs/bdt_qg_vs_wzt_seed42_retrained.yaml \
  --output-prefix results/qg_vs_wzt_seed42_reference_vs_retrained
