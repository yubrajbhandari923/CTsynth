#!/bin/bash
# RUN-0112 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
train_phys.py --arm B --seed 1
