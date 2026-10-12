#!/bin/bash
# RUN-0018 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
uv run python projects/p04_xcist/compare.py projects/p04_xcist/results/dataset_v0 projects/p04_xcist/results/mismatch_v0
