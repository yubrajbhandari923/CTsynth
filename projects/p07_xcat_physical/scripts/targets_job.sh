#!/bin/bash -l
# Build physical targets (conc_v1.npy) for the patients given as arguments, one process per allocated GPU. usage: targets_job.sh ID ... (or ALL)
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; D=$O/cache_v1; R=$HOME/datasets/truect_pilot/COPD
if [ "$1" = ALL ]; then IDS=$(ls $D | sed 's/^p//' | sort); else IDS="$@"; fi
N=$(echo $IDS | wc -w); echo "patients: $N"; mkdir -p $O/targets_gen
NG=$(nvidia-smi -L | wc -l); echo "GPUs: $NG"
for i in $(seq 0 $((NG-1))); do
  SL=$(echo $IDS | tr ' ' '\n' | awk -v i=$i -v n=$NG '(NR-1)%n==i')
  [ -z "$SL" ] && continue
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 $PY $S/build_targets.py --root $R --cache $D --patients $SL > $O/targets_gen/worker$i.log 2>&1 &
done; wait; echo "== done $(date -Is)"; grep -h -c Traceback $O/targets_gen/worker*.log | paste -sd+ | bc | xargs echo tracebacks:
