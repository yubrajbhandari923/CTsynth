#!/usr/bin/env bash
# Run a CTsynth python entry point inside a SLURM job (cwd = bundle source/).
# CTSYNTH_PY overrides the interpreter; default = torch-venv (torch cu126, numpy, matplotlib).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/xcist_env_path.sh"
PY="${CTSYNTH_PY:-$CTSYNTH_XCIST_ENV/torch-venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/projects/p05_aapm" MPLBACKEND=Agg
exec "$PY" "$@"
