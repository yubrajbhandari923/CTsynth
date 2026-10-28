"""Pick the sign conventions (rotation sign, channel flip, row flip) by correlating a coarse LEAP recon of the REFERENCE case
with the supplied reference reconstruction over the 8 in-plane dihedral transforms x 2 slice orders. Reference only."""
import itertools
import json
import sys

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from common import OUT
from leap_recon import recon

ref = np.asarray(nib.load(OUT / "nifti/reference_ct_provisional.nii.gz").dataobj).astype(np.float32)  # (slice,row,col)
ref = ndi.zoom(ref, (0.5, 0.5, 0.5), order=1)  # ~1.095 mm slices, 1.2 mm pixels
res = []
for rot, chf, rowf in itertools.product([1, -1], [False, True], [False, True]):
    f, zp, info = recon("reference", 384, 1.2, 1.0975, rot_sign=rot, ch_flip=chf, row_flip=rowf)
    f = np.asarray(f)
    print(rot, chf, rowf, f.shape, float(f.min()), float(f.max()), flush=True)
    if f.ndim != 3 or not np.isfinite(f).all():
        continue
    best = None
    for k in range(8):
        a = np.rot90(f, k % 4, axes=(1, 2))
        if k >= 4:
            a = a[:, :, ::-1]
        for zf in (1, -1):
            b = a[::zf]
            nz = min(b.shape[0], ref.shape[0])
            for dz in range(-12, 13, 2):
                lo = max(0, dz); hi = min(nz, nz + dz)
                if hi - lo < 30:
                    continue
                x = b[lo - dz: hi - dz]; y = ref[lo:hi]
                if x.shape != y.shape:
                    continue
                c = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
                if best is None or c > best["corr"]:
                    best = {"corr": c, "dihedral_k": k, "z_flip": zf, "dz_slices": dz}
    res.append({"rot_sign": rot, "ch_flip": chf, "row_flip": rowf, **best, "mu_max": float(f.max())})
    print(res[-1], flush=True)
res.sort(key=lambda r: -r["corr"])
(OUT / "reports/recon_search.json").write_text(json.dumps(res, indent=1))
print("BEST", res[0])
