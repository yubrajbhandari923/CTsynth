#!/bin/bash -l
# usage: run_xcat_benchmark.sh   one A5000 job, serial: end-to-end pipeline runs (0.8 and 0.6 mm), 7-rep operator benchmark, coarse-grid gradient checks, aggregation.
set -e
D=$(cd "$(dirname "$0")" && pwd); O=/home/yb107/datasets/truect_pilot/outputs; B=$O/xcat_pipeline/benchmarks
sbatch --job-name=xcat_bench --partition=vram24 --gres=gpu:a5000:1 --cpus-per-task=8 --mem=96G --time=04:00:00 --output=$O/logs/%x_%j.out --wrap="
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0; cd $D; export PYTHONPATH=$HOME/ctsynth_truect/pylib; P=/home/yb107/.conda/envs/myBase/bin/python
mkdir -p $B
for c in reference copd_22728 copd_21535; do for v in 0.8 0.6; do \$P run_xcat_pipeline.py --case \$c --voxel-size-mm \$v --output $B/e2e/\${c}_\${v}mm || echo FAILED e2e \$c \$v; done; done
\$P benchmark_xcat.py --cases reference copd_22728 copd_21535 --vox 0.8 0.6 --reps 7 --warmup 2 --out $B || echo FAILED bench
\$P benchmark_xcat.py --cases reference copd_22728 copd_21535 --vox 2.4 --grad-checks --out $B || echo FAILED gradchecks
\$P benchmark_xcat_aggregate.py $B
"
