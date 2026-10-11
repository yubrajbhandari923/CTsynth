#!/bin/bash
# RUN-0002 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
uv run python projects/p01_material_inference/exp002/run_exp002.py
