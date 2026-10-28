#!/usr/bin/env bash
set -euo pipefail
root=${TREE_BENCHMARK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
if [[ ${1:-} != --inside ]]; then
 exec "$root/scripts/conifer_container.sh" bash "$root/scripts/run_leaf_competition.sh" --inside "$@"
fi
shift
source "$root/scripts/conifer_runtime.sh"
exec "$root/$CONIFER_VENV/bin/python" "$root/scripts/validate_leaf_competition.py" "$@"
