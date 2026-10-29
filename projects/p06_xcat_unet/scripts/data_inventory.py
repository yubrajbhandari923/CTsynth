#!/usr/bin/env python3
"""Inventory of the COPD XCAT phantoms on Tara versus DRCC (read-only, stdlib only; run on the DRCC login node, it needs `ssh tara`).

Writes data_manifest.csv, data_inventory.md and transfer_missing_copd.sh (+ its file list) into OUT. Nothing is transferred.
usage: data_inventory.py OUT_DIR
"""

import csv
import math
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

TARA = "tara"
TARA_ROOT = "/data3/TrueCT22_Challenge_Phantoms/COPD"
DRCC_ROOT = Path.home() / "datasets/truect_pilot/COPD"
NATIVE_VOX = 0.25  # assumed (documented only for the reference phantom)
BIN_RE = re.compile(r"^phantom_(\d+)_(\d+)_(\d+)_(\d+)_8bit_textured_copd_eroded_flipped\.bin$")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def tara(cmd):
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", TARA, cmd], capture_output=True, text=True, check=True
    )
    return r.stdout


def sizes(listing):
    return {
        ln.rsplit(" ", 1)[0]: int(ln.rsplit(" ", 1)[1]) for ln in listing.splitlines() if ln.strip()
    }


# remote listings: one ssh call each, names + byte sizes only
top = sizes(tara(f"cd {TARA_ROOT} && stat -c '%n %s' * 2>/dev/null"))
tabs = sizes(tara(f"cd {TARA_ROOT}/material_tables && stat -c '%n %s' *"))
lacs = sizes(tara(f"cd {TARA_ROOT}/material_lac && stat -c '%n %s' */*"))
non_bin = sorted(
    k
    for k in top
    if not k.endswith(".bin") and k not in ("material_lac", "material_mcgpu", "material_tables")
)
other_bins = sorted(k for k in top if k.endswith(".bin") and not BIN_RE.match(k))

rows, ids = [], {}
for name, size in sorted(top.items()):
    m = BIN_RE.match(name)
    if not m:
        continue
    pid, a, b, c = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    ids.setdefault(pid, []).append(name)
    xlsx = f"challenge_materials_copd_smeared_{pid}.xlsx"
    lac_files = sorted(k.split("/", 1)[1] for k in lacs if k.startswith(pid + "/"))
    d_bin, d_x = DRCC_ROOT / name, DRCC_ROOT / "material_tables" / xlsx
    d_lac = [DRCC_ROOT / "material_lac" / pid / f for f in lac_files]
    on_bin = d_bin.exists() and d_bin.stat().st_size == size
    on_x = d_x.exists() and d_x.stat().st_size == tabs.get(xlsx, -1)
    on_lac = bool(d_lac) and all(
        p.exists() and p.stat().st_size == lacs[f"{pid}/{p.name}"] for p in d_lac
    )
    z_mm, y_mm, x_mm = (
        c * NATIVE_VOX,
        b * NATIVE_VOX,
        a * NATIVE_VOX,
    )  # array (z,y,x) = (c,b,a) per the layout established by the survey
    g = [math.ceil(v / 0.8) + 2 for v in (z_mm, y_mm, x_mm)]
    views = math.ceil((g[0] * 0.8 + 80) / 35.05) * 1000
    rows.append(
        dict(
            patient_id=pid,
            bin_file=name,
            bytes=size,
            dims_name_a_b_c=f"{a}x{b}x{c}",
            array_shape_zyx=f"{c}x{b}x{a}",
            bytes_equal_prod_dims=(size == a * b * c),
            assumed_extent_mm_zyx=f"{z_mm:.0f}x{y_mm:.0f}x{x_mm:.0f}",
            grid_0p8mm_zyx=f"{g[0]}x{g[1]}x{g[2]}",
            views_margin40=views,
            proj_GiB_0p8=round(views * 64 * 900 * 4 / 2**30, 2),
            vol_GiB_0p8=round(g[0] * g[1] * g[2] * 4 / 2**30, 3),
            xlsx=xlsx,
            xlsx_bytes=tabs.get(xlsx, ""),
            lac_files=";".join(lac_files),
            lac_bytes=sum(v for k, v in lacs.items() if k.startswith(pid + "/")),
            on_drcc_bin=on_bin,
            on_drcc_xlsx=on_x,
            on_drcc_lac=on_lac,
            drcc_path=str(d_bin) if on_bin else "",
        )
    )
fields = list(rows[0])
with open(OUT / "data_manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# other copies on DRCC (bounded search, not the whole NFS)
found = subprocess.run(
    "find $HOME/datasets $HOME/ctsynth_truect -maxdepth 4 -name 'phantom_*copd*.bin' 2>/dev/null",
    shell=True,
    capture_output=True,
    text=True,
).stdout.split()
missing = [r for r in rows if not (r["on_drcc_bin"] and r["on_drcc_xlsx"] and r["on_drcc_lac"])]
have = [r for r in rows if r["on_drcc_bin"]]
mb = [r["bytes"] for r in rows]
need_bytes = sum(r["bytes"] + (r["xlsx_bytes"] or 0) + r["lac_bytes"] for r in missing)
du = shutil.disk_usage(Path.home())
dups = {k: v for k, v in ids.items() if len(v) > 1}
GB = 1e9

list_rel = []
for r in missing:
    list_rel += [r["bin_file"], f"material_tables/{r['xlsx']}"] + [
        f"material_lac/{r['patient_id']}/{f}" for f in r["lac_files"].split(";")
    ]
(OUT / "transfer_missing_copd.filelist").write_text("\n".join(list_rel) + "\n")
(OUT / "transfer_missing_copd.sh").write_text(f"""#!/bin/bash
# Resumable transfer of the COPD phantoms that are NOT yet on DRCC: {len(missing)} patients, {need_bytes / GB:.1f} GB (bins + xlsx + LAC tables only).
# Run ON DRCC (needs `ssh tara`). Default = DRY RUN (lists what would be copied). Nothing is copied without --go.
#   ./transfer_missing_copd.sh            dry run
#   ./transfer_missing_copd.sh --go       copy on THIS shell (use --sbatch instead of the login node)
#   ./transfer_missing_copd.sh --sbatch   submit the copy as a small CPU job on node004 (2 CPUs, 4 GB, no GPU)
#   ./transfer_missing_copd.sh --verify   after the copy: size check of every file against Tara
# Resume: just run --go (or --sbatch) again; rsync --partial --inplace continues interrupted files and skips complete ones.
set -euo pipefail
SRC={TARA}:{TARA_ROOT}/
DST=$HOME/datasets/truect_pilot/COPD/
LIST=$(dirname "$(readlink -f "$0")")/transfer_missing_copd.filelist
OPTS="-a --partial --inplace --info=stats2,progress2 --files-from=$LIST"
case "${{1:-}}" in
  --go)     mkdir -p "$DST"; rsync $OPTS "$SRC" "$DST" ;;
  --sbatch) sbatch --partition=defq --nodelist=node004 --cpus-per-task=2 --mem=4G --time=06:00:00 --job-name=copd_transfer \\
              --output=$HOME/datasets/truect_pilot/outputs/xcat_unet/logs/%x_%j.out --wrap="$(readlink -f "$0") --go" ;;
  --verify) bad=0; while read -r f; do a=$(ssh {TARA} "stat -c %s {TARA_ROOT}/$f"); b=$(stat -c %s "$DST$f" 2>/dev/null || echo -1); [ "$a" = "$b" ] || {{ echo "MISMATCH $f remote=$a local=$b"; bad=1; }}; done < "$LIST"; echo "verify done, bad=$bad" ;;
  *)        echo "DRY RUN: would copy {len(missing)} patients ({need_bytes / GB:.1f} GB):"; rsync -n --info=stats2 $OPTS "$SRC" "$DST" | tail -15 ;;
esac
""")
(OUT / "transfer_missing_copd.sh").chmod(0o755)

(
    OUT / "data_inventory.md"
).write_text(f"""# COPD XCAT phantom inventory (generated by data_inventory.py)

Source: `{TARA}:{TARA_ROOT}` (read-only listing). DRCC target: `{DRCC_ROOT}`.

## Counts
- Unique COPD patient phantoms (`phantom_<id>_<a>_<b>_<c>_8bit_textured_copd_eroded_flipped.bin`): **{len(ids)}** ({len(rows)} files).
- Duplicate or alternate variants of the same patient: {len(dups)} ({dups or "none"}). `.bin` files not matching the naming pattern: {other_bins or "none"}. Other top-level entries: {non_bin or "none"} (besides material_lac / material_mcgpu / material_tables).
- Material tables (`material_tables/*.xlsx`): {len(tabs)}; per-patient LAC folders (`material_lac/<id>/`, 2 files each: conversion table + `lin_att_120kV_121materials.bin`): {len({k.split("/")[0] for k in lacs})}; shared `material_mcgpu` (45 MB) is already on DRCC as `mcgpu_materials` and is not needed.
- Every file size equals the product of the three numbers in its name (uint8): {all(r["bytes_equal_prod_dims"] for r in rows)}.

## Sizes
- Phantom `.bin` total {sum(mb) / GB:.1f} GB ({sum(mb) / 2**30:.1f} GiB); per file min / median / max = {min(mb) / GB:.2f} / {statistics.median(mb) / GB:.2f} / {max(mb) / GB:.2f} GB. Tables are ~5 MB in total.
- Already on DRCC (bin + table + LAC complete): **{len(rows) - len(missing)}** patients ({", ".join(r["patient_id"] for r in have)}) in `{DRCC_ROOT}`. Other bounded searches under `~/datasets` and `~/ctsynth_truect` found: {found or "none"}.
- **Missing: {len(missing)} patients, {need_bytes / GB:.1f} GB** to copy.

## DRCC storage
- `$HOME` (NFS `/home`): {du.free / 1e12:.1f} TB free of {du.total / 1e12:.1f} TB (shared by the lab; no per-user quota tool on the login node). `/scratch`: ~1.8 TB free of 74 TB (98% used) - not recommended for a 170 GB copy. Node-local tiers are measured separately in the node probe.
- The copy therefore fits comfortably on `/home` ({need_bytes / GB:.0f} GB = {100 * need_bytes / du.free:.1f}% of the free space) but is large for a shared NFS: run it as the CPU job (`--sbatch`) when you decide to proceed.

## What 0.8 mm simulation costs per patient (from the filename dimensions, assumed 0.25 mm native voxels, 40 mm margin per side)
- Simulation grid (z,y,x): see `data_manifest.csv`; views {min(r["views_margin40"] for r in rows)}-{max(r["views_margin40"] for r in rows)}; projection tensor {min(r["proj_GiB_0p8"] for r in rows):.2f}-{max(r["proj_GiB_0p8"] for r in rows):.2f} GiB; attenuation volume {min(r["vol_GiB_0p8"] for r in rows):.2f}-{max(r["vol_GiB_0p8"] for r in rows):.2f} GiB (float32). Earlier benchmark: autograd peaked at 21-24 GB for 11k-12k views, so the largest patients need an explicit memory check before any differentiable use (not needed for plain forward + FBP).

## Caveats
- The COPD native voxel size (0.25 mm) is ASSUMED, not documented. Axis order [z,y,x] = (c,b,a) from the filename follows the layout survey of the two available patients; it is not re-verified for the other 65.
- Nothing was transferred by this script.
""")
print(
    f"{len(rows)} patients; on DRCC: {len(have)}; missing: {len(missing)} = {need_bytes / GB:.1f} GB; dups: {len(dups)}; free home: {du.free / 1e12:.2f} TB"
)
