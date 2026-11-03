#!/bin/bash
# RUN-0098 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_run_py.sh projects/p05_aapm/stage1_solve.py --data /scratch/railabs/yb107/research-os/shared-data/aapm-dl-spectral-ct-zenodo-14262737 --split configs/aapm_split_v1.json --rank low --max-iter 1000
