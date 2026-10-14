#!/bin/bash
# RUN-0053 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_run_py.sh projects/p05_aapm/exp028_hybrid.py --data /scratch/railabs/yb107/research-os/shared-data/aapm-dl-spectral-ct-zenodo-14262737 --split configs/aapm_split_v1.json --ckpt /home/yb107/research-os/projects/CTsynth/runs/RUN-0041/checkpoint.pt --mu 0
