#!/bin/bash -l
# usage: run_xcat.sh CASE VOXEL_MM [extra pipeline args...]   one A5000 (24 GB, partition vram24), 8 CPUs, 64 GB host RAM, 1 h.
# Output: outputs/xcat_pipeline/<CASE>[_<VOXEL>mm unless 0.8 default is used]/ ; zarr needs the private pylib (pip --target, not a global install).
set -e
CASE=$1; VOX=$2; shift 2
D=$(cd "$(dirname "$0")" && pwd); O=/home/yb107/datasets/truect_pilot/outputs
OUTDIR=$O/xcat_pipeline/$CASE; [ "$VOX" != "0.8" ] && OUTDIR=${OUTDIR}_${VOX}mm
sbatch --job-name=xcat_${CASE}_${VOX} --partition=vram24 --gres=gpu:a5000:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --output=$O/logs/%x_%j.out \
  --wrap="module load cuda12.2/toolkit/12.2.1 gcc/11.2.0; cd $D; PYTHONPATH=$HOME/ctsynth_truect/pylib /home/yb107/.conda/envs/myBase/bin/python run_xcat_pipeline.py --case $CASE --voxel-size-mm $VOX --energy-kev 70 --geometry truect_helical --output $OUTDIR $*"
