#!/bin/bash -l
# Final evaluation on the 10 TEST patients (touched once): 8 trained models (one GPU each) + the intensity-only baseline, 12 test variants each,
# then the paired analysis. Representative held-out cases are exported as NIfTI (CT + truth once, predictions per arm for seed 0).
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src:$HOME/ctsynth_xcat/scripts
S=$HOME/ctsynth_xcat/scripts; O=$HOME/datasets/truect_pilot/outputs/xcat_unet; D=$O/cache_v1; L=/tmp/yb107_xcat_cache_train
SPLIT=$HOME/ctsynth_xcat/configs/split_v1_frozen.json
[ -d $L/p11989 ] || { echo "staging cache"; mkdir -p $L; rsync -a $D/ $L/; }
TEST=$($PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['test']))"); TRAIN=$($PY -c "import json;print(' '.join(json.load(open('$SPLIT'))['train']))")
VARS="clean n0_1e5 var1 n0_1e6 n0_3e5 n0_3e4 lowpass_w2 lowpass_w4 dec2 e60 e80 pitch1p4"
EXP="11989:clean 11989:n0_1e5 11989:e80 11989:pitch1p4 13799:clean 13799:n0_1e5 13799:e80 13799:pitch1p4"
mkdir -p $O/evals; $PY -c "import json;json.dump(json.load(open('$O/cache_validation.json'))['summary']['group_names'], open('$O/evals/group_names.json','w'))"
i=0
for arm in A B C D; do for seed in 0 1; do
  EX=""; [ $seed = 0 ] && EX="--export-cases $EXP"; [ $arm = A ] && [ $seed = 0 ] && EX="$EX --export-ct"
  CUDA_VISIBLE_DEVICES=$i $PY $S/evaluate_unet.py --cache $L --patients $TEST --variants $VARS --ckpt $O/runs/${arm}_s${seed}/best.pt --stride 64 \
     --out $O/evals/${arm}_s${seed} --name ${arm}_s${seed} $EX > $O/evals/${arm}_s${seed}.log 2>&1 &
  i=$((i+1))
done; done
$PY $S/evaluate_unet.py --cache $L --patients $TEST --variants $VARS --baseline-fit-patients $TRAIN --out $O/evals/baseline > $O/evals/baseline.log 2>&1 &
wait
$PY $S/analyze_arms.py $O/evals $O/analysis --group-names $O/evals/group_names.json
echo "== done $(date -Is)"
