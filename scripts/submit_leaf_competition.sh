#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p build
exec sbatch --parsable --partition=c --cpus-per-task=2 --mem=8G --time=01:00:00 \
 --job-name=leaf-emulation --chdir="$root" --export="ALL,TREE_BENCHMARK_ROOT=$root" \
 --output='build/leaf_competition_%j.log' "$root/scripts/run_leaf_competition.sh" "$@"
