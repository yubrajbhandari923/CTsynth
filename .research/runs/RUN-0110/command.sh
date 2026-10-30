#!/bin/bash
# RUN-0110 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
train_phys.py --arm A --seed 1
