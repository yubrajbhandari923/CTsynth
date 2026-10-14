#!/bin/bash
# RUN-0028 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_setup_torch_env.sh
