#!/bin/bash
# Resumable transfer of the COPD phantoms that are NOT yet on DRCC: 65 patients, 181.2 GB (bins + xlsx + LAC tables only).
# Run ON DRCC (needs `ssh tara`). Default = DRY RUN (lists what would be copied). Nothing is copied without --go.
#   ./transfer_missing_copd.sh            dry run
#   ./transfer_missing_copd.sh --go       copy on THIS shell (use --sbatch instead of the login node)
#   ./transfer_missing_copd.sh --sbatch   submit the copy as a small CPU job on node004 (2 CPUs, 4 GB, no GPU)
#   ./transfer_missing_copd.sh --verify   after the copy: size check of every file against Tara
# Resume: just run --go (or --sbatch) again; rsync --partial --inplace continues interrupted files and skips complete ones.
set -euo pipefail
SRC=tara:/data3/TrueCT22_Challenge_Phantoms/COPD/
DST=$HOME/datasets/truect_pilot/COPD/
LIST=$(dirname "$(readlink -f "$0")")/transfer_missing_copd.filelist
OPTS="-a --partial --inplace --info=stats2,progress2 --files-from=$LIST"
case "${1:-}" in
  --go)     mkdir -p "$DST"; rsync $OPTS "$SRC" "$DST" ;;
  --sbatch) sbatch --partition=defq --nodelist=node004 --cpus-per-task=2 --mem=4G --time=06:00:00 --job-name=copd_transfer \
              --output=$HOME/datasets/truect_pilot/outputs/xcat_unet/logs/%x_%j.out --wrap="$(readlink -f "$0") --go" ;;
  --verify) bad=0; while read -r f; do a=$(ssh tara "stat -c %s /data3/TrueCT22_Challenge_Phantoms/COPD/$f"); b=$(stat -c %s "$DST$f" 2>/dev/null || echo -1); [ "$a" = "$b" ] || { echo "MISMATCH $f remote=$a local=$b"; bad=1; }; done < "$LIST"; echo "verify done, bad=$bad" ;;
  *)        echo "DRY RUN: would copy 65 patients (181.2 GB):"; rsync -n --info=stats2 $OPTS "$SRC" "$DST" | tail -15 ;;
esac
