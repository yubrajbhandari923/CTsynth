#!/bin/bash -l
# usage: run_gpu.sh JOBNAME CPUS MEM TIME script.py [args...]   (1 GPU; CUDA 12.2 runtime for libleapct; python = myBase)
set -e
NAME=$1; CPUS=$2; MEM=$3; TIME=$4; shift 4
D=$(cd "$(dirname "$0")" && pwd)
sbatch --job-name="$NAME" --partition=defq --gres=gpu:1 --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
  --output=/home/yb107/datasets/truect_pilot/outputs/logs/%x_%j.out \
  --wrap="module load cuda12.2/toolkit/12.2.1 gcc/11.2.0; cd $D && /home/yb107/.conda/envs/myBase/bin/python $*"
