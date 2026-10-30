#!/bin/bash -l
# Closed-loop LEAP resimulation of predicted vs true phantoms for the test patients, one worker per GPU (round-robin over patients).
# usage: resim_job.sh [EXPORT_PATIENT ...]
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src:$HOME/ctsynth_xcat/scripts
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; D=$O/cache_v1; SPLIT=$HOME/ctsynth_xcat/configs/split_v1_frozen.json; P=$O/physical/preds; X=$O/physical/resim
TE=$($PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['test']))"); mkdir -p $X
PR=""; for arm in A B C D; do for seed in 0 1; do PR="$PR ${arm}_s${seed}=$P/${arm}_s${seed}"; done; done
NG=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((NG-1))); do
  SL=$(echo $TE | tr ' ' '\n' | awk -v i=$i -v n=$NG '(NR-1)%n==i')
  [ -z "$SL" ] && continue
  CUDA_VISIBLE_DEVICES=$i $PY $S/resim_phys.py --cache $D --patients $SL --preds $PR --out $X --export "$@" > $X/worker$i.log 2>&1 &
done; wait; echo "== done $(date -Is)"; grep -c Traceback $X/worker*.log | paste -sd' '
