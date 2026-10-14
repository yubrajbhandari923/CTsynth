#!/bin/bash
# RUN-0066 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_run_py.sh projects/p05_aapm/milestone_a.py --data /scratch/railabs/yb107/research-os/shared-data/aapm-dl-spectral-ct-zenodo-14262737 --split configs/aapm_split_v1.json --arm physics --n-train 800 --n-val 100 --steps 6000 --batch 4 --views 32 --depth 4 --width 32 --cosine --warmup 300 --lr 3e-4 --clip 1.0 --val-every 100 --seed 2 --output occ_softmax
