#!/bin/bash -l
# Probe the 7 TB node-local NVMe at /cachedata on the compute node (no GPU): permissions, free space, existing users, sequential + parallel throughput.
echo "== $(hostname) $(date -Is) job ${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?}"; id -un
ls -ld /cachedata; df -hT /cachedata; ls -la /cachedata | head -15; du -sh /cachedata/* 2>/dev/null | sort -h | tail -5
mkdir -p /cachedata/yb107_ctsynth_probe 2>&1 && D=/cachedata/yb107_ctsynth_probe || D=
if [ -z "$D" ]; then echo "cannot create a directory in /cachedata (permission?)"; exit 0; fi
echo "== sequential 16 GiB"; dd if=/dev/zero of=$D/t.bin bs=1M count=16384 conv=fdatasync 2>&1 | tail -1; dd if=$D/t.bin of=/dev/null bs=1M iflag=direct 2>&1 | tail -1
echo "== 4 parallel writers x 8 GiB"; s=$(date +%s.%N); for i in 1 2 3 4; do dd if=/dev/zero of=$D/p$i.bin bs=1M count=8192 conv=fdatasync 2>/dev/null & done; wait; e=$(date +%s.%N); echo "aggregate write GB/s: $(echo "32*1.0737/($e-$s)" | bc -l)"
s=$(date +%s.%N); for i in 1 2 3 4; do dd if=$D/p$i.bin of=/dev/null bs=1M iflag=direct 2>/dev/null & done; wait; e=$(date +%s.%N); echo "aggregate O_DIRECT read GB/s: $(echo "32*1.0737/($e-$s)" | bc -l)"
rm -rf $D; echo "== retention hints"; ls /etc/cron* 2>/dev/null | head -3; grep -r -l -i cachedata /etc/cron* /etc/slurm* /cm/local/apps/slurm 2>/dev/null | head -5
