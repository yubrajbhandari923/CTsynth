#!/bin/bash -l
# Does running N independent simulators (one per GPU, same node) scale? Same patient/variants in each process, separate caches.
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; C=/dev/shm/yb107_scal_$SLURM_JOB_ID
mkdir -p $C $O/scaling
for N in 1 2 4 8; do
  rm -rf $C/*; s=$(date +%s.%N)
  for i in $(seq 0 $((N-1))); do
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 $PY $S/generate_variants.py --root $HOME/datasets/truect_pilot/COPD --patients 22728 --variants clean n0_1e5 e60 dec2 --cache $C/proc$i > $C/log$i.txt 2>&1 &
  done; wait; e=$(date +%s.%N)
  w=$(echo "$e - $s" | bc -l); echo "N=$N wall_s=$w sim_variants_done=$((4*N))  throughput_variants_per_min=$(echo "4*$N*60/$w" | bc -l)"
  grep -h "done in" $C/log*.txt | tr '\n' ' '; echo; grep -c "fbp" $C/log*.txt | tr '\n' ' '; echo
done | tee $O/scaling/scaling.txt
rm -rf $C
