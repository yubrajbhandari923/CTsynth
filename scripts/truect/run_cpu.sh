#!/bin/bash -l
# usage: run_cpu.sh JOBNAME CPUS MEM TIME script.py [args...]   (submits a CPU job; python = myBase conda env)
set -e
export PATH=$PATH:/cm/shared/apps/slurm/current/bin
NAME=$1; CPUS=$2; MEM=$3; TIME=$4; shift 4
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p /home/yb107/datasets/truect_pilot/outputs/logs
sbatch --job-name="$NAME" --partition=defq --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
  --output=/home/yb107/datasets/truect_pilot/outputs/logs/%x_%j.out \
  --wrap="cd $D && /home/yb107/.conda/envs/myBase/bin/python $*"
