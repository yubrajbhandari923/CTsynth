#!/bin/bash -l
# Stage 2: simulate the planned variants for all 67 patients on 8 GPUs (one worker per GPU), cache on node-local NVMe, persist per patient.
# Resumable: finished variants (ct_*.json) are skipped; each finished patient is rsynced to durable storage immediately.
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet
L=/tmp/yb107_xcat_cache; D=$O/cache_v1; PLAN=$HOME/ctsynth_xcat/configs/dataset_plan_v1.json
mkdir -p $L $D $O/dataset_gen
echo "== $(date -Is) start; staging any already-durable patients back to node-local NVMe"; /usr/bin/time -f "stage-in %e s" rsync -a $D/ $L/
s=$(date +%s.%N)
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 $PY $S/generate_variants.py --root $HOME/datasets/truect_pilot/COPD --plan $PLAN --worker $i \
     --cache $L/w$i --durable $D > $O/dataset_gen/worker$i.log 2>&1 &
done; wait; e=$(date +%s.%N)
echo "== generation wall: $(echo "$e - $s" | bc -l) s"
grep -h -c "Traceback" $O/dataset_gen/worker*.log | paste -sd+ | bc | xargs echo "tracebacks:"
echo "== cache footprint"; du -sh $D; find $D -name "ct_*.npy" | wc -l | xargs echo "CT volumes:"; find $D -name "ct_*.json" | wc -l | xargs echo "metadata files:"
echo "== done $(date -Is)"
