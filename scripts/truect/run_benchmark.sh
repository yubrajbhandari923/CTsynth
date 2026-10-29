#!/bin/bash -l
# usage: run_benchmark.sh [GPU_TYPE] -- serial small/ref/copd0 LEAP benchmark on ONE GPU in ONE job (no co-tenants), then aggregate.
# GPU_TYPE: a5000 (default, 24 GB) | a6000 (48 GB). Output: outputs/benchmarks/ (raw per-config files; a6000 runs go to benchmarks_a6000/).
set -e
G=${1:-a5000}; D=$(cd "$(dirname "$0")" && pwd); O=/home/yb107/datasets/truect_pilot/outputs
BD=$O/benchmarks; [ "$G" = a6000 ] && BD=$O/benchmarks_a6000
PART=vram24; [ "$G" = a6000 ] && PART=vram48
sbatch --job-name=bench_$G --partition=$PART --gres=gpu:$G:1 --cpus-per-task=8 --mem=96G --time=02:30:00 --output=$O/logs/%x_%j.out \
  --wrap="module load cuda12.2/toolkit/12.2.1 gcc/11.2.0; cd $D; P=/home/yb107/.conda/envs/myBase/bin/python; for c in small ref copd0; do \$P benchmark_leap.py \$c --out $BD || true; done; \$P benchmark_aggregate.py $BD"
