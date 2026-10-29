#!/bin/bash -l
# Smoke evaluation of the Stage-1 model + staging-cost measurement (durable NFS -> node-local NVMe). NOT a held-out-patient result.
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src:$HOME/ctsynth_xcat/scripts
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; L=/tmp/yb107_stage_$SLURM_JOB_ID
mkdir -p $L; s=$(date +%s.%N); rsync -a $O/cache_smoke/ $L/; e=$(date +%s.%N)
echo "staging 2.1 GB NFS -> /tmp NVMe: $(echo "$e - $s" | bc -l) s"; du -sh $L
$PY $S/evaluate_unet.py --cache $L --patients 21535 --variants clean n0_1e5 lowpass_w4 e80 pitch1p4 dec2 --ckpt $O/smoke/run_armC/best.pt \
    --baseline-fit-patients 22728 --out $O/smoke/eval_smoke --export 1 2>&1 | grep -v -E "UserWarning|from_numpy"
rm -rf $L
