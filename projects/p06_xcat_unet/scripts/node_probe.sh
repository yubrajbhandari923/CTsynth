#!/bin/bash -l
# Probe the compute node (storage tiers, memory, GPU, ssh to tara) and measure write/read throughput per tier.
# usage: sbatch -p vram24 --nodelist=node004 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:20:00 node_probe.sh
set -u
echo "== host $(hostname) $(date -Is) job ${SLURM_JOB_ID:-none}"; echo "== allocation: cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}MB gpus=${SLURM_GPUS_ON_NODE:-?} SLURM_TMPDIR=${SLURM_TMPDIR:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
echo "== cpu"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket|NUMA node\("; nproc
echo "== memory"; free -h; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null; grep -E "Hugepagesize|Shmem:" /proc/meminfo
echo "== filesystems"; df -hT / /dev/shm /tmp /local "${SLURM_TMPDIR:-/tmp}" "$HOME" /scratch 2>&1 | sort -u; echo; lsblk -o NAME,SIZE,TYPE,ROTA,MOUNTPOINT 2>/dev/null | head -20; mount | grep -E " /local| /tmp| /scratch| /home| /dev/shm" | cut -c1-160
echo "== ssh tara from this node"; timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 tara "ls /data3/TrueCT22_Challenge_Phantoms | head -3" 2>&1 | head -4
bench() {  # dir label
  d="$1/ctsynth_probe_$$"; mkdir -p "$d" 2>/dev/null || { echo "$2: cannot create $d"; return; }
  avail=$(df -Pk "$d" | awk 'NR==2{print $4}'); [ "$avail" -lt 12000000 ] && { echo "$2: <12 GB free, skipped"; rmdir "$d"; return; }
  w=$( { dd if=/dev/zero of="$d/t.bin" bs=1M count=4096 conv=fdatasync 2>&1 | tail -1; } ); r=$( { dd if="$d/t.bin" of=/dev/null bs=1M 2>&1 | tail -1; } )
  echo "$2 write(4 GiB, fdatasync): $w"; echo "$2 read (page-cache warm): $r"
  sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; r2=$( { dd if="$d/t.bin" of=/dev/null bs=1M iflag=direct 2>&1 | tail -1; } ); echo "$2 read (O_DIRECT, bypasses cache): $r2"
  rm -rf "$d"; }
echo "== throughput"; bench /dev/shm shm; bench /tmp tmp; [ -d /local ] && bench /local local; [ -n "${SLURM_TMPDIR:-}" ] && bench "$SLURM_TMPDIR" slurm_tmpdir; bench "$HOME/datasets/truect_pilot/outputs" home_nfs; bench /scratch/yb107 scratch 2>/dev/null
/home/yb107/.conda/envs/myBase/bin/python - <<'P'
import torch
print("== torch", torch.__version__, torch.cuda.device_count(), "gpus")
free, tot = torch.cuda.mem_get_info(); print("gpu0 free/total MiB", free/2**20, tot/2**20)
P
