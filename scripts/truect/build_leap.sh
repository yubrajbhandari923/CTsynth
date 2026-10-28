#!/bin/bash -l
# Build LLNL LEAP (pinned clone at ~/ctsynth_truect/leap/LEAP, commit 0c8846f) for sm_86 (A5000/A6000). Run via sbatch on a GPU node.
set -e
module load gcc/11.2.0 cmake/3.24.2 cuda12.2/toolkit/12.2.1
export CUDAARCHS=86
cd ~/ctsynth_truect/leap/LEAP
rm -f src/libleapct.so
sh ./etc/build.sh
ls -la build/lib
ldd build/lib/libleapct.so | grep -i "not found" || echo "all deps resolved"
