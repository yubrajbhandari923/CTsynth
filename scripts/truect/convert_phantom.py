"""Phantom .bin -> material-label NIfTI (uint8, full res) + density NIfTI (g/cm3, stride-downsampled), + legend JSON.

usage: convert_phantom.py CASE AXCODES VOXEL_MM STRIDE_DENSITY [TAG]
AXCODES e.g. IPR = array axes (z,y,x) increase toward Inferior, Posterior, Right (PROVISIONAL unless confirmed).
Labels are only ever sampled by index (nearest), never interpolated. Density = LUT[label] from the case's own xlsx
(Organ ID -> Density), cross-checked against the COPD conversion_table .in (same IDs), unmapped labels -> 0 and reported.
"""
import json
import re
import sys

import numpy as np
import openpyxl

from common import INFILES, OUT, PHANTOMS, array_shape
from niiutil import write_nii

case, ax, vox, stride = sys.argv[1], tuple(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
tag = sys.argv[5] if len(sys.argv) > 5 else "provisional"
path, dims, xlsx, _ = PHANTOMS[case]
vol = np.memmap(path, dtype=np.uint8, mode="r").reshape(array_shape(dims))

rows = list(openpyxl.load_workbook(xlsx, data_only=True).active.iter_rows(values_only=True))
hdr = rows[0]
ci = {h: i for i, h in enumerate(hdr)}
table = {int(r[ci["Organ ID"]]): (str(r[0]), float(r[ci["Density"]]), int(r[ci["MC ID"]])) for r in rows[1:] if r[0] is not None}
cross = {}
if case in INFILES:
    inp = INFILES[case]
    for ln in inp.read_text().splitlines():
        m = re.match(r"^(\d+)\s+(\S+)\s+(\d+)\s+([0-9.eE+-]+)", ln)
        if m:
            cross[int(m[1])] = (m[2], float(m[4]), int(m[3]))
    bad = [k for k in table if k not in cross or abs(cross[k][1] - table[k][1]) > 1e-9 or cross[k][0] != table[k][0]]
    print("xlsx vs .in mismatches:", bad)
lut = np.zeros(256, np.float32)
for k, (_, d, _) in table.items():
    lut[k] = d
present = np.flatnonzero(np.bincount(np.asarray(vol[::4, ::4, ::4]).ravel(), minlength=256))
unmapped = [int(i) for i in present if int(i) not in table]
print("labels present (stride 4):", len(present), "unmapped:", unmapped)

legend = {int(k): {"name": v[0], "density_g_cm3": v[1], "mc_material_id": v[2]} for k, v in sorted(table.items())}
(OUT / "nifti").mkdir(parents=True, exist_ok=True)
(OUT / f"nifti/{case}_legend.json").write_text(json.dumps(
    {"case": case, "legend": legend, "unmapped_labels_present": unmapped, "xlsx": str(xlsx)}, indent=1))

desc = f"{case} {{}} axes {''.join(ax)} {tag.upper()} vox={vox}mm"
lab = np.ascontiguousarray(vol)  # full-res labels (uint8)
r1 = write_nii(lab, (vox,) * 3, ax, OUT / f"nifti/{case}_material_labels_{tag}.nii.gz", desc.format("labels"))
sub = np.ascontiguousarray(vol[::stride, ::stride, ::stride])
dens = lut[sub]
r2 = write_nii(dens, (vox * stride,) * 3, ax, OUT / f"nifti/{case}_density_{tag}.nii.gz", desc.format("density g/cm3"))
if stride > 1:
    r3 = write_nii(sub, (vox * stride,) * 3, ax, OUT / f"nifti/{case}_material_labels_s{stride}_{tag}.nii.gz",
                   desc.format(f"labels stride{stride}"))
print(r1["shape"], r2["shape"])
