#!/bin/bash
# RUN-0009 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
uv run python projects/p02_aapm_benchmark/verify_real_kit.py --kit /private/tmp/claude-501/-Users-yubraj-Code-CTsynth/445890b5-68ad-4cf8-8b1c-fcce0c22387e/scratchpad/kit/starting_kit --out projects/p02_aapm_benchmark/results_real_kit
