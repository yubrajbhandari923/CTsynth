#!/bin/bash -l
# Verify the COPD transfer: (1) byte size of every listed file vs Tara, (2) sha256 of the first and last 256 MiB of each phantom vs Tara,
# (3) size == product of the filename dimensions, (4) the two pre-existing phantoms were not modified (mtime check).
LIST=$HOME/datasets/truect_pilot/outputs/xcat_unet/inventory/transfer_missing_copd.filelist
SRC=/data3/TrueCT22_Challenge_Phantoms/COPD; DST=$HOME/datasets/truect_pilot/COPD; OUT=$HOME/datasets/truect_pilot/outputs/xcat_unet/inventory/verify_report.txt
: > $OUT
bad=0; n=0
while read -r f; do
  a=$(ssh -n -o BatchMode=yes tara "stat -c %s $SRC/$f"); b=$(stat -c %s "$DST/$f" 2>/dev/null || echo -1); n=$((n+1))
  [ "$a" = "$b" ] || { echo "SIZE MISMATCH $f remote=$a local=$b" >> $OUT; bad=$((bad+1)); }
done < $LIST
echo "size check: $n files, mismatches=$bad" | tee -a $OUT
chk() { f=$1; s=$(stat -c %s $DST/$f); t=$((s-268435456)); [ $t -lt 0 ] && t=0
  r1=$(ssh -o BatchMode=yes tara "head -c 268435456 $SRC/$f | sha256sum | cut -c1-16; tail -c 268435456 $SRC/$f | sha256sum | cut -c1-16" | paste -sd' ')
  l1=$( (head -c 268435456 $DST/$f | sha256sum | cut -c1-16; tail -c 268435456 $DST/$f | sha256sum | cut -c1-16) | paste -sd' ')
  [ "$r1" = "$l1" ] && echo "OK $f" || echo "CONTENT MISMATCH $f remote=[$r1] local=[$l1]"; }
export -f chk; export SRC DST
grep "^phantom_" $LIST | xargs -P 4 -I{} bash -c 'chk {}' > $OUT.content
echo "content check: $(grep -c '^OK' $OUT.content) ok, $(grep -c MISMATCH $OUT.content) mismatches of $(wc -l < $OUT.content)" | tee -a $OUT
python3 - <<'P' | tee -a $OUT
import os, re, glob
bad = 0; n = 0
for f in sorted(glob.glob(os.path.expanduser("~/datasets/truect_pilot/COPD/phantom_*.bin"))):
    a, b, c = (int(x) for x in re.match(r".*phantom_\d+_(\d+)_(\d+)_(\d+)_", f).groups()); n += 1
    if os.path.getsize(f) != a * b * c: bad += 1; print("DIM MISMATCH", f)
print(f"dimension check: {n} phantoms, size != product(dims) for {bad}")
P
ls -l --time-style=long-iso $DST/phantom_22728_* $DST/phantom_21535_* | awk '{print "pre-existing:", $6, $7, $5, $8}' | tee -a $OUT
echo "total bins: $(ls $DST/phantom_*.bin | wc -l); xlsx: $(ls $DST/material_tables/*.xlsx | wc -l); lac dirs: $(ls -d $DST/material_lac/*/ | wc -l); du: $(du -sh $DST | cut -f1)" | tee -a $OUT
