#!/bin/bash
# RUN-0091 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_run_py.sh projects/p05_aapm/exp024_anomaly_diag.py --data /scratch/railabs/yb107/research-os/shared-data/aapm-dl-spectral-ct-zenodo-14262737 --split configs/aapm_split_v1.json --run-root /home/yb107/research-os/projects/CTsynth/runs --runs RUN-0047:softplus RUN-0049:softplus RUN-0064:occ_softmax RUN-0065:occ_softmax RUN-0066:occ_softmax RUN-0083:softmax:oracle RUN-0089:softmax:oracle RUN-0090:softmax:oracle
