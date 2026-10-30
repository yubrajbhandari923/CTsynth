#!/bin/bash -l
# Train arms A,B,C,D x seeds 0,1 as 8 independent processes, one per GPU (no LEAP). Resumable: rerun the same job.
# usage: train_phys_job.sh STEPS [MAX_MINUTES] [WORKERS]
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
STEPS=${1:-40000}; MAXM=${2:-540}; NW=${3:-14}
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src:$HOME/ctsynth_xcat/scripts
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; D=$O/cache_v1; L=/tmp/yb107_xcat_cache_train; SPLIT=$HOME/ctsynth_xcat/configs/split_v1_frozen.json
R=$O/physical/runs; mkdir -p $L $R
TR=$($PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['train']))"); VA=$($PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['val']))")
echo "== $(date -Is) staging train+val patients NFS -> NVMe (16 parallel rsyncs)"
echo $TR $VA | tr ' ' '\n' | /usr/bin/time -f "stage-in %e s" xargs -P 16 -I{} rsync -a --exclude labels.npy --exclude 'slices_*' $D/p{}/ $L/p{}/ ; du -sh $L
i=0
for arm in A B C D; do for seed in 0 1; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=4 $PY $S/train_phys.py --arm $arm --seed $seed --cache $L --train-patients $TR --val-patients $VA \
     --out $R/${arm}_s${seed} --steps $STEPS --val-every 5000 --ckpt-every 500 --workers $NW --max-minutes $MAXM > $R/${arm}_s${seed}.log 2>&1 &
  i=$((i+1))
done; done; wait
echo "== done $(date -Is)"; for r in $R/*_s?; do echo $(basename $r) $(cat $r/status.json); done
