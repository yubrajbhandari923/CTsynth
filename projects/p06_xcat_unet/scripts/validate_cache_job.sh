#!/bin/bash -l
PY=$HOME/ctsynth_xcat/venv/bin/python; O=$HOME/datasets/truect_pilot/outputs/xcat_unet
$PY $HOME/ctsynth_xcat/scripts/validate_cache.py $O/cache_v1 $HOME/ctsynth_xcat/configs/dataset_plan_v1.json $O/cache_validation.json
