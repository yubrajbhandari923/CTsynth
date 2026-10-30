#!/bin/bash -l
# Score all 8 trained runs on the held-out TEST patients (one process per GPU). usage: predict_job.sh VARIANT ...
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src:$HOME/ctsynth_xcat/scripts
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; D=$O/cache_v1; SPLIT=$HOME/ctsynth_xcat/configs/split_v1_frozen.json; R=$O/physical/runs; P=$O/physical/preds
TE=$($PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['test']))"); mkdir -p $P
i=0
for arm in A B C D; do for seed in 0 1; do
  CUDA_VISIBLE_DEVICES=$i $PY $S/predict_phys.py --run $R/${arm}_s${seed} --cache $D --patients $TE --variants "$@" --save-variants clean --out $P/${arm}_s${seed} > $P/${arm}_s${seed}.log 2>&1 &
  i=$((i+1))
done; done; wait; echo "== done $(date -Is)"; grep -c Traceback $P/*.log | paste -sd' '
