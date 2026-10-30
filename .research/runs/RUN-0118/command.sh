#!/bin/bash
# RUN-0118 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
aggregate_phys.py --resim resim_v2
