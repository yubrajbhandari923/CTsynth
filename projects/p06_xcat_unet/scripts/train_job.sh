#!/bin/bash -l
# Stage 3: train arms A,B,C,D x seeds 0,1 as 8 independent processes, one per GPU (no LEAP anywhere). Resumable: rerun the same job.
# usage: train_job.sh STEPS [MAX_MINUTES]
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
STEPS=${1:-40000}; MAXM=${2:-540}
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src:$HOME/ctsynth_xcat/scripts
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; D=$O/cache_v1; L=/tmp/yb107_xcat_cache_train; SPLIT=$HOME/ctsynth_xcat/configs/split_v1_frozen.json
mkdir -p $L $O/runs
echo "== $(date -Is) staging cache NFS -> NVMe"; /usr/bin/time -f "stage-in %e s" rsync -a $D/ $L/ 2>&1 | tail -1; du -sh $L
TR=$($PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['train']))"); VA=$($PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['val']))")
i=0
for arm in A B C D; do for seed in 0 1; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 $PY $S/train_unet.py --arm $arm --seed $seed --cache $L --train-patients $TR --val-patients $VA \
     --out $O/runs/${arm}_s${seed} --steps $STEPS --val-every 5000 --ckpt-every 500 --workers 8 --max-minutes $MAXM > $O/runs/${arm}_s${seed}.log 2>&1 &
  i=$((i+1))
done; done; wait
echo "== done $(date -Is)"; for r in $O/runs/*_s?; do echo $(basename $r) $(cat $r/status.json); done
