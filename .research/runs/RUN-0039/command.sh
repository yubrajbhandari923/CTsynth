#!/bin/bash
# RUN-0039 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_run_py.sh projects/p05_aapm/milestone_a.py --data /scratch/railabs/yb107/research-os/shared-data/aapm-dl-spectral-ct-zenodo-14262737 --split configs/aapm_split_v1.json --arm supervised --n-train 800 --n-val 100 --steps 6000 --batch 8 --depth 4 --width 32 --cosine --val-every 100 --seed 1
