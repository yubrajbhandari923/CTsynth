#!/bin/bash -l
PY=$HOME/ctsynth_xcat/venv/bin/python; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; X=$O/slicer_export
mkdir -p $X
for p in 11989 13799; do for v in clean n0_1e5 e80 pitch1p4; do
  cp $O/evals/A_s0/p${p}_${v}_ct.nii.gz $O/evals/A_s0/p${p}_${v}_truth.nii.gz $X/ 2>/dev/null
  for a in A B C D; do cp $O/evals/${a}_s0/p${p}_${v}_pred_${a}_s0.nii.gz $X/ 2>/dev/null; done
done; done
$PY $HOME/ctsynth_xcat/scripts/make_overlay_figure.py $O/evals $O/analysis/test_patient_11989_overlays.png 11989 clean n0_1e5 e80
cp $O/cache_v1/p11989/group_names.json $X/class_names.json
ls -la $X | awk '{print $5, $9}' | tail -n +4; du -sh $X
