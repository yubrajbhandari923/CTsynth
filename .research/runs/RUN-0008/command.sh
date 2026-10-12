#!/bin/bash
# RUN-0008 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
uv run python projects/p03_unsupervised/run_exp004.py configs/exp004_ladder.yaml --out projects/p03_unsupervised/results
