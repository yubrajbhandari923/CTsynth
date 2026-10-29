#!/bin/bash -l
# Single entry point for every job of this project: ALWAYS node004 (one node's worth of resources at most; never two nodes).
# usage: sub.sh NAME GPUS CPUS MEM TIME SCRIPT [args...]      e.g. sub.sh vary 1 16 96G 01:00:00 variation_job.sh
# Refuses requests above one node (128 CPUs, 8 GPUs) and refuses to submit while another job of ours is queued/running with a
# combined request above one node.
set -euo pipefail
NAME=$1; GPUS=$2; CPUS=$3; MEM=$4; TIME=$5; SCRIPT=$6; shift 6
[ "$GPUS" -le 8 ] && [ "$CPUS" -le 128 ] || { echo "request exceeds one node (8 GPUs / 128 CPUs)"; exit 1; }
used=$(squeue -u "$USER" -h -o "%b %C" 2>/dev/null | awk '{g=($1 ~ /gpu/)?gensub(/.*:/,"",1,$1):0; G+=g; C+=$2} END {print G+0, C+0}')
read -r UG UC <<< "$used"
[ $((UG + GPUS)) -le 8 ] && [ $((UC + CPUS)) -le 128 ] || { echo "combined allocation would exceed one node (in use: gpus=$UG cpus=$UC)"; exit 1; }
OUT=$HOME/datasets/truect_pilot/outputs/xcat_unet/logs; mkdir -p "$OUT"
GRES=""; [ "$GPUS" -gt 0 ] && GRES="--gres=gpu:$GPUS"
sbatch --partition=vram24 --nodelist=node004 $GRES --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --job-name="$NAME" \
       --output="$OUT/%x_%j.out" "$HOME/ctsynth_xcat/scripts/$SCRIPT" "$@"
