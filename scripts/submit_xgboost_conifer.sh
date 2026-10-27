#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/build/conifer"
exec sbatch --chdir="$root" --export="ALL,TREE_BENCHMARK_ROOT=$root" --job-name=xgb-conifer --partition=c --cpus-per-task=2 --mem=8G \
  --time=02:00:00 --output="$root/build/conifer/full-%j.log" \
  "$root/scripts/run_xgboost_conifer.sh" "$@"
