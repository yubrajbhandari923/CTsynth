#!/bin/bash -l
# Arm D conditioning diagnosis on validation patients (1 GPU). usage: diag_cond_job.sh
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src
O=$HOME/datasets/truect_pilot/outputs/xcat_unet; SPLIT=$HOME/ctsynth_xcat/configs/split_v1_frozen.json
J() { $PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['$1']))"; }
VAL=$(J val | cut -d' ' -f1-4)
mkdir -p $O/physical/diagnostics
$PY $HOME/ctsynth_xcat/scripts/diag_cond.py --cache $O/cache_v1 --runs $O/physical/runs/D_s0 $O/physical/runs/D_s1 --patients $VAL --variants clean n0_1e5 var1 --train-patients $(J train) --out $O/physical/diagnostics/cond_diag_val.json
