#!/bin/bash -l
# Stage A: overfit/export check (GPU 0) and a train->kill->resume check of arm D (GPU 1), in parallel.
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src:$HOME/ctsynth_xcat/scripts
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; D=$O/cache_v1; W=$O/physical/smoke; rm -rf $W; mkdir -p $W
CUDA_VISIBLE_DEVICES=0 $PY $S/overfit_phys.py --cache $D --patient 22728 --out $W/overfit > $W/overfit.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  # 2 train patients, 1 val patient; 120 steps total, interrupted at 60 (max-minutes tiny then restart)
  $PY $S/train_phys.py --arm D --cache $D --train-patients 22728 21535 --val-patients 10379 --out $W/resume --steps 120 --val-every 60 --ckpt-every 20 --workers 6 --max-minutes 0.4 > $W/resume1.log 2>&1
  $PY $S/train_phys.py --arm D --cache $D --train-patients 22728 21535 --val-patients 10379 --out $W/resume --steps 120 --val-every 60 --ckpt-every 20 --workers 6 > $W/resume2.log 2>&1 ) &
wait; tail -n 3 $W/*.log; cat $W/resume/status.json
