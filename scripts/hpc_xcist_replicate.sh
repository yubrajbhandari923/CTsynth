#!/usr/bin/env bash
# Run inside the SLURM job (cwd = bundle source/). Uses the isolated env built by hpc_setup_xcist_env.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/xcist_env_path.sh"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load gcc/11.2.0   # the rebuilt XCIST C library links its runtime
export PYTHONPATH="$PWD/src:$PWD/projects/p04_xcist" MPLBACKEND=Agg
exec "$CTSYNTH_XCIST_ENV/xcist-venv/bin/python" projects/p04_xcist/hpc_replicate.py
