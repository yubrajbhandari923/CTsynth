#!/bin/bash -l
# CPU job: material-recovery analysis of all 8 runs on the 10 test patients (clean-input predictions).
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src
O=$HOME/datasets/truect_pilot/outputs/xcat_unet; SPLIT=$HOME/ctsynth_xcat/configs/split_v1_frozen.json; P=$O/physical/preds
TE=$($PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['test']))")
PR=""; for arm in A B C D; do for s in 0 1; do PR="$PR ${arm}_s$s=$P/${arm}_s$s"; done; done
mkdir -p $O/physical/diagnostics
$PY $HOME/ctsynth_xcat/scripts/analyze_materials.py --cache $O/cache_v1 --preds $PR --patients $TE --out $O/physical/diagnostics/material_recovery_test.json
