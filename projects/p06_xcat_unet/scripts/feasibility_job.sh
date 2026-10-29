#!/bin/bash -l
# One-case feasibility benchmark on node004: process A (simulate + cache + measure) EXITS, then process B (U-Net probe) starts fresh.
# submit: sbatch -p vram24 --nodelist=node004 --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=01:30:00 --job-name=feas feasibility_job.sh [PATIENT] [VOX]
module load cuda12.2/toolkit/12.2.1 gcc/11.2.0
P=${1:-22728}; V=${2:-0.8}
PY=$HOME/ctsynth_xcat/venv/bin/python; export PYTHONPATH=$HOME/ctsynth_xcat/src
OUT=$HOME/datasets/truect_pilot/outputs/xcat_unet/feasibility; S=$HOME/ctsynth_xcat/scripts
FAST=/dev/shm/yb107_ctsynth_${SLURM_JOB_ID:-x}; NVME=/tmp/yb107_ctsynth_${SLURM_JOB_ID:-x}; HOMEM=$OUT/home_matrix_${SLURM_JOB_ID:-x}
mkdir -p "$OUT" "$FAST" "$NVME" "$HOMEM"
echo "== $(hostname) $(date -Is) job ${SLURM_JOB_ID:-} cpus=${SLURM_CPUS_PER_TASK:-} mem=${SLURM_MEM_PER_NODE:-}MB"
echo "== GPU memory before process A:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
/usr/bin/time -v $PY $S/sim_one_case.py --root $HOME/datasets/truect_pilot/COPD --patient $P --vox $V --fast $FAST --nvme $NVME --home $HOMEM \
   --out $OUT/sim_report_p${P}_${V}mm.json 2> $OUT/sim_time_v_p${P}.txt
echo "== process A exit code $?"; grep -E "Maximum resident|Elapsed \(wall" $OUT/sim_time_v_p${P}.txt
echo "== GPU memory after process A exited:"; sleep 2; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "== cache footprint:"; du -sh $FAST; df -h /dev/shm | tail -1
/usr/bin/time -v $PY $S/unet_probe.py --cache $FAST/p$P --out $OUT/unet_probe_p${P}_${V}mm.json 2> $OUT/unet_time_v_p${P}.txt
echo "== process B exit code $?"; grep -E "Maximum resident|Elapsed \(wall" $OUT/unet_time_v_p${P}.txt
echo "== GPU memory after process B:"; sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader
rm -rf "$FAST" "$NVME" "$HOMEM"; echo "== cleaned $(date -Is)"
