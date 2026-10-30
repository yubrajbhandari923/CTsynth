#!/bin/bash
# RUN-0111 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
train_phys.py --arm B --seed 0
