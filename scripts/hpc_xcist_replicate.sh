#!/usr/bin/env bash
# Run inside the SLURM job (cwd = bundle source/). Uses the isolated env built by hpc_setup_xcist_env.sh.
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
module load gcc/11.2.0   # the rebuilt XCIST C library links its runtime
export PYTHONPATH="$PWD/src:$PWD/projects/p04_xcist" MPLBACKEND=Agg
exec /home/yb107/research-os/projects/CTsynth/env/xcist-venv/bin/python projects/p04_xcist/hpc_replicate.py
