#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} != --inside ]]; then
  exec "$root/scripts/conifer_container.sh" bash "$root/scripts/setup_conifer_workspace.sh" --inside
fi
source "$root/configs/environments/cmssw-conifer.env"
export SCRAM_ARCH
set +u
source /cvmfs/cms.cern.ch/cmsset_default.sh
set -u
mkdir -p "$root/workspaces/cmssw"
cd "$root/workspaces/cmssw"
if [[ ! -d $CMSSW_RELEASE ]]; then scram project CMSSW "$CMSSW_RELEASE"; fi
cd "$root"
if [[ ! -x $CONIFER_BASE_PYTHON ]]; then
  # Resolve the existing training interpreter without altering that environment.
  set +u
  source /software/f2022/software/mamba/24.3.0-0/etc/profile.d/conda.sh
  conda activate tree-benchmark
  set -u
  CONIFER_BASE_PYTHON=$(command -v python)
fi
if [[ ! -x $CONIFER_VENV/bin/python ]]; then
  "$CONIFER_BASE_PYTHON" -m venv "$CONIFER_VENV"
fi
"$CONIFER_VENV/bin/python" -m pip install -r configs/environments/conifer-emulation-lock.txt
"$CONIFER_VENV/bin/python" -m pip check
"$CONIFER_VENV/bin/python" -m pip freeze > "$CONIFER_VENV/installed.txt"
source scripts/conifer_runtime.sh
"$CONIFER_VENV/bin/python" -c 'import conifer,xgboost; print("conifer",conifer.__version__,"xgboost",xgboost.__version__)'
(
  cd "$CMSSW_BASE/src"
  scram tool info conifer
  scram tool info hls
)
