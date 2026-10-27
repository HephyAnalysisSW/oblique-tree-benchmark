#!/usr/bin/env bash
# Run an argv safely inside the CMS EL9 container, from the repository root.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
command -v cmssw-el9 >/dev/null || { echo 'cmssw-el9 is required (CVMFS CMS common tools).' >&2; exit 1; }
printf -v directory '%q' "$root"
printf -v command '%q ' "$@"
exec cmssw-el9 --command-to-run "cd $directory && $command"
