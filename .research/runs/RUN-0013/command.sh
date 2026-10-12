#!/bin/bash
# RUN-0013 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
python3 scripts/hpc_env_probe.py
