"""Which (swap, flip_z, flip_y, flip_x) maps phantom inserts onto the recon? Mean recon HU at each insert's predicted centre.
Insert labels/densities (xlsx): 2=1.19 g/cc (~+190 HU), 3=0.91 (~-90 HU), 4=1.52 (~+500 HU), body=1.0 (~0 HU).
Offsets are taken from each volume's own body centre, recon voxel = 0.6 mm in-plane, 0.5476 mm per slice."""
import itertools
import json

import nibabel as nib
import numpy as np

from common import OUT, PHANTOMS

ph = np.asarray(np.memmap(PHANTOMS["reference"][0], dtype=np.uint8, mode="r").reshape(320, 860, 860)[::2, ::2, ::2])
hu = np.asarray(nib.load(OUT / "nifti/reference_ct_provisional.nii.gz").dataobj).astype(np.float32)
pb = np.argwhere(ph > 0).mean(0) * 0.5  # body centre mm (z,y,x)
cen = {k: np.argwhere(ph == k).mean(0) * 0.5 - pb for k in (2, 3, 4)}
rb = np.argwhere(hu > -500).mean(0)
rows = []
for swap, fz, fy, fx in itertools.product([0, 1], [1, -1], [1, -1], [1, -1]):
    got = {}
    for k, (dz, dy, dx) in cen.items():
        s = rb[0] + fz * dz / 0.5476
        a, b = (dy * fy, dx * fx)  # phantom (y,x) offsets mapped to recon (row, col) in mm
        if swap:
            a, b = b, a
        r, c = rb[1] + a / 0.6, rb[2] + b / 0.6
        s, r, c = int(round(s)), int(round(r)), int(round(c))
        got[k] = float(hu[s - 6:s + 7, r - 8:r + 9, c - 8:c + 9].mean())
    err = (got[2] - 190) ** 2 + (got[3] + 90) ** 2 + (got[4] - 500) ** 2
    rows.append({"swap": swap, "flip_z": fz, "flip_y": fy, "flip_x": fx, "HU_ins2(1.19)": got[2], "HU_ins3(0.91)": got[3],
                 "HU_ins4(1.52)": got[4], "sq_err": err})
rows.sort(key=lambda r: r["sq_err"])
for r in rows[:8]:
    print({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()})
(OUT / "reports/reference_insert_test.json").write_text(json.dumps(rows, indent=1))
