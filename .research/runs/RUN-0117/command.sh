#!/bin/bash
# RUN-0117 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
aggregate_phys.py
