#!/bin/bash -l
# CPU job: aggregate baselines, predictions and the corrected resimulation (resim_v2) into tables.
PH=$HOME/datasets/truect_pilot/outputs/xcat_unet/physical
PYTHONPATH=$HOME/ctsynth_xcat/src $HOME/ctsynth_xcat/venv/bin/python $HOME/ctsynth_xcat/scripts/aggregate_phys.py --baselines $PH/baselines_test.json --preds $PH/preds --resim $PH/resim_v2 --out $PH/aggregate_v2
