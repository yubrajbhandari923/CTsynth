#!/bin/bash -l
# Size-only verification of every transferred file against Tara (ssh -n so the loop reads the whole list).
LIST=$HOME/datasets/truect_pilot/outputs/xcat_unet/inventory/transfer_missing_copd.filelist
SRC=/data3/TrueCT22_Challenge_Phantoms/COPD; DST=$HOME/datasets/truect_pilot/COPD
ssh -o BatchMode=yes tara "cd $SRC && xargs stat -c '%n %s'" < $LIST > /tmp/remote_sizes_$$.txt
(cd $DST && xargs -a $LIST stat -c '%n %s') > /tmp/local_sizes_$$.txt
echo "files listed: $(wc -l < $LIST); remote rows: $(wc -l < /tmp/remote_sizes_$$.txt); local rows: $(wc -l < /tmp/local_sizes_$$.txt)"
diff <(sort /tmp/remote_sizes_$$.txt) <(sort /tmp/local_sizes_$$.txt) > /tmp/diff_$$.txt && echo "SIZE CHECK: all $(wc -l < $LIST) files identical in size to Tara ($(awk '{s+=$2} END {printf "%.2f GB", s/1e9}' /tmp/local_sizes_$$.txt))" || { echo "SIZE MISMATCHES:"; head /tmp/diff_$$.txt; }
