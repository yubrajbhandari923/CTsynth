"""Scan detector centre offsets (channels, rows) for the REFERENCE recon; score vs the supplied recon with the smoothed in-body RMSE
(HU est via fixed mu_water) and body-centroid offset. Only reference data is used."""
import json

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from common import OUT
from leap_recon import recon

ref = np.asarray(nib.load(OUT / "nifti/reference_ct_provisional.nii.gz").dataobj).astype(np.float32)
mw = json.loads((OUT / "reports/mu_water_calibration.json").read_text())["mu_water_per_mm"]
body = ndi.binary_erosion(ref > -500, iterations=8); body[:6] = body[-6:] = False
sm = lambda x: ndi.gaussian_filter(x, 2.0)
sref = sm(ref)
rows = []
for cs, ox, oy in [(0, .35, -.25), (-1, -.1, -.3), (-1.5, 0, -.3), (-0.5, .15, -.3), (0, .35, -.3)]:
    rs = 0
    f, _, _ = recon("reference", 768, 0.6, 0.6, nz=192, rot_sign=-1, row_flip=True, col_shift=cs, row_shift=rs, off_x_mm=ox, off_y_mm=oy)
    hu = 1000 * (np.ascontiguousarray(np.asarray(f)[::-1]) / mw - 1)
    d = sm(hu) - sref
    m = ndi.binary_fill_holes(hu[96] > -500)
    c = np.argwhere(m).mean(0)
    rows.append({"col_shift": cs, "off_x_mm": ox, "off_y_mm": oy, "rmse_smoothed_HU": float(np.sqrt(np.mean(d[body] ** 2))), "centroid_rc": c.tolist()})
    print(rows[-1], flush=True)
(OUT / "reports/center_test3.json").write_text(json.dumps(rows, indent=1))
