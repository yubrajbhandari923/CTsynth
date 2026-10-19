#!/bin/bash
# RUN-0087 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv/bin/python projects/p05_aapm/identifiability_theory.py
