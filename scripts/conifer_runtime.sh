#!/usr/bin/env bash
# Source from another script inside the EL9 container.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$root/configs/environments/cmssw-conifer.env"
export SCRAM_ARCH
# CMS setup/runtime scripts reference optional unset variables.
set +u
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$root/workspaces/cmssw/$CMSSW_RELEASE/src"
eval "$(scram runtime -sh)"
set -u
export JSON_ROOT=$(scram tool info json | sed -n 's/^JSON_BASE=//p')/include
export XILINX_AP_INCLUDE=$(scram tool info hls | sed -n 's/^HLS_BASE=//p')/include
cd "$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$root:$root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
