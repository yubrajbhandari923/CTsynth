#!/bin/bash
# RUN-0114 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
train_phys.py --arm C --seed 1
