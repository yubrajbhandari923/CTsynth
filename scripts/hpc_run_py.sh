#!/usr/bin/env bash
# Run a CTsynth python entry point inside a SLURM job (cwd = bundle source/).
# CTSYNTH_PY overrides the interpreter; default = the isolated env python (numpy; torch once built).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/xcist_env_path.sh"
PY="${CTSYNTH_PY:-$CTSYNTH_XCIST_ENV/xcist-venv/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/projects/p05_aapm" MPLBACKEND=Agg
exec "$PY" "$@"
