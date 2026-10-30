#!/bin/bash
# RUN-0115 — re-run this command (paths relative to the project root)
cd "$(dirname "$0")/../../.." && cd .
train_phys.py --arm D --seed 0
