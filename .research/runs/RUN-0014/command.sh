#!/bin/bash
# RUN-0014 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
.venv-xcist/bin/python projects/p04_xcist/official_example.py
