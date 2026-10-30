#!/bin/bash -l
# CPU job: aggregate corrected resim (v2), export concentration NIfTI (arm C_s0, patients 23235 25816), NIfTI consistency check + previews.
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src
O=$HOME/datasets/truect_pilot/outputs/xcat_unet; PH=$O/physical; S=$HOME/ctsynth_xcat/scripts
$PY $S/aggregate_phys.py --baselines $PH/baselines_test.json --preds $PH/preds --resim $PH/resim_v2 --out $PH/aggregate_v2
$PY $S/export_conc.py --cache $O/cache_v1 --pred-dir $PH/preds/C_s0/preds --name C_s0 --patients 23235 25816 --out $PH/resim_v2/conc_nifti
$PY $S/make_previews.py --resim $PH/resim_v2 --patients 23235 25816 --arm C_s0 --out $PH/previews
