"""Rigid shift / scale between the supplied reference recon and the LEAP recon (same frame): body-mask centroids, equivalent
diameters, and a sub-voxel phase-correlation estimate. Diagnoses detector-centre / pixel-size assumptions."""
import json

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from skimage.registration import phase_cross_correlation

from common import OUT

ref = np.asarray(nib.load(OUT / "nifti/reference_ct_provisional.nii.gz").dataobj).astype(np.float32)
mu = np.asarray(nib.load(OUT / "nifti/reference_leap_recon_mu_per_mm.nii.gz").dataobj).astype(np.float32)
mw = json.loads((OUT / "reports/mu_water_calibration.json").read_text())["mu_water_per_mm"]
hu = 1000 * (mu / mw - 1)
mid = slice(40, 150)
out = {}
for nm, v in (("ref", ref), ("leap", hu)):
    m = ndi.binary_fill_holes(v[mid] > -500)
    lab, n = ndi.label(m[len(m) // 2]); big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    c = np.argwhere(big).mean(0); area = big.sum()
    out[nm] = {"centroid_row_col_vox": c.tolist(), "equiv_diam_vox": float(2 * np.sqrt(area / np.pi))}
    zc = (v > -500).sum((1, 2)); idx = np.flatnonzero(zc > 0.5 * zc.max())
    out[nm]["z_extent_slices_first_last"] = [int(idx[0]), int(idx[-1])]
sl = [90, 100, 110]
sh = [phase_cross_correlation(ref[s], hu[s], upsample_factor=20)[0].tolist() for s in sl]
out["shift_rowcol_vox_(register leap to ref)"] = sh
out["diam_ratio_leap_over_ref"] = out["leap"]["equiv_diam_vox"] / out["ref"]["equiv_diam_vox"]
print(json.dumps(out, indent=1))
