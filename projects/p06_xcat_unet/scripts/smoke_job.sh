#!/bin/bash -l
# Stage 1 (two available patients): cache -> persist -> overfit one batch -> train arm C with a forced stop and a resume.
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src:$HOME/ctsynth_xcat/scripts
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; C=/dev/shm/yb107_smoke_$SLURM_JOB_ID
mkdir -p $C $O/smoke $O/cache_smoke
echo "== $(date -Is) generate"; $PY $S/generate_variants.py --root $HOME/datasets/truect_pilot/COPD --patients 22728 21535 \
  --variants clean n0_1e6 n0_3e5 n0_1e5 n0_3e4 lowpass_w2 lowpass_w4 apod_0p75 e60 dec2 e80 pitch1p4 --cache $C 2>&1 | grep -v -E "UserWarning|from_numpy"
echo "== $(date -Is) persist to durable storage"; /usr/bin/time -f "rsync %e s" rsync -a $C/ $O/cache_smoke/; du -sh $C $O/cache_smoke
echo "== overfit"; $PY $S/overfit_test.py --cache $C --patient 22728 --variant clean --out $O/smoke/overfit_report.json 2>&1 | tail -3
echo "== train arm C: 1st segment (forced stop after 1.5 min), then resume to 300 steps"
R=$O/smoke/run_armC
$PY $S/train_unet.py --arm C --cache $C --train-patients 22728 --val-patients 21535 --out $R --steps 300 --val-every 150 --ckpt-every 50 --max-minutes 1.5 --workers 6 2>&1 | tail -4
cat $R/status.json; echo
$PY $S/train_unet.py --arm C --cache $C --train-patients 22728 --val-patients 21535 --out $R --steps 300 --val-every 150 --ckpt-every 50 --workers 6 2>&1 | tail -4
cat $R/status.json; echo
echo "== D arm (conditioning) 40-step smoke"; $PY $S/train_unet.py --arm D --cache $C --train-patients 22728 --val-patients 21535 --out $O/smoke/run_armD --steps 40 --val-every 40 --ckpt-every 20 --workers 4 2>&1 | tail -2
rm -rf $C; echo "== done $(date -Is)"
