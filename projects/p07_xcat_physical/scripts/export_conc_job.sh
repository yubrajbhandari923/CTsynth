#!/bin/bash -l
# Concentration NIfTI export (CPU). usage: export_conc_job.sh ID ...
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src
O=$HOME/datasets/truect_pilot/outputs/xcat_unet
for arm in C_s0 A_s0; do $PY $HOME/ctsynth_xcat/scripts/export_conc.py --cache $O/cache_v1 --pred-dir $O/physical/preds/$arm/preds --name $arm --patients "$@" --out $O/physical/resim/conc_nifti; done
