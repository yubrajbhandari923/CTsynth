#!/bin/bash
# RUN-0010 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
uv run python projects/p02_aapm_benchmark/evaluate.py configs/exp003_aapm_smoke.yaml --data-dir DATA --out OUT
