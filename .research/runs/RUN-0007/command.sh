#!/bin/bash
# RUN-0007 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
uv run --with scipy python projects/p03_unsupervised/verify_identifiability.py --out projects/p03_unsupervised/results_verify
