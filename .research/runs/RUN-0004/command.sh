#!/bin/bash
# RUN-0004 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
uv run python projects/p02_aapm_benchmark/evaluate.py configs/exp003_aapm_smoke.yaml --data-dir /private/tmp/claude-501/-Users-yubraj-Code-CTsynth/445890b5-68ad-4cf8-8b1c-fcce0c22387e/scratchpad/synth_data --out projects/p02_aapm_benchmark/results_synthetic --split val --purpose development
