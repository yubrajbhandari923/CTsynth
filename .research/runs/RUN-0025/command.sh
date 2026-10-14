#!/bin/bash
# RUN-0025 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
bash scripts/hpc_run_py.sh projects/p05_aapm/freeze_split.py --data /scratch/railabs/yb107/research-os/shared-data/aapm-dl-spectral-ct-zenodo-14262737 --out-dir /scratch/railabs/yb107/research-os/shared-data/aapm-dl-spectral-ct-zenodo-14262737/splits
