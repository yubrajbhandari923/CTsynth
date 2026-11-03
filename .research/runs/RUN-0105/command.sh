#!/bin/bash
# RUN-0105 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_run_py.sh projects/p05_aapm/stage1_solve.py --data /scratch/railabs/yb107/research-os/shared-data/aapm-dl-spectral-ct-zenodo-14262737 --split configs/aapm_split_v1.json --rank high --max-iter 50 --runs C:rough C:null A_img:uniform D:uniform
