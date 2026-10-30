#!/bin/bash -l
# Physics baselines P0/P1/O1. usage: baselines_job.sh SPLIT(val|test) VARIANT ...   (P1 threshold fitted on TRAIN patients only)
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src
O=$HOME/datasets/truect_pilot/outputs/xcat_unet; D=$O/cache_v1; SPLIT=$HOME/ctsynth_xcat/configs/split_v1_frozen.json
SP=$1; shift; mkdir -p $O/physical
J() { $PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['$1']))"; }
$PY $HOME/ctsynth_xcat/scripts/eval_baselines.py --cache $D --patients $(J $SP) --variants "$@" --fit-patients $(J train) --out $O/physical/baselines_$SP.json
