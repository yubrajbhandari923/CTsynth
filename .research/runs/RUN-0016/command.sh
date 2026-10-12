#!/bin/bash
# RUN-0016 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv-xcist/bin/python projects/p04_xcist/generate.py projects/p04_xcist/results/dataset_v0
