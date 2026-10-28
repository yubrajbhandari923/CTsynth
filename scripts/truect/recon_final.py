"""Full-resolution LEAP helical FBP -> NIfTI in the reference-recon frame (array = slice,row,col; axcodes IPL, PROVISIONAL).

Detector-centre shift -1 channel and volume offset (-0.1,-0.3) mm are EMPIRICAL alignments to the supplied reference recon
(center_test.py: smoothed in-body RMSE 8.8 -> 5.8 HU); no DICOM tag gives them. Sign conventions chosen by recon_search.py against the supplied reference recon (corr 0.955, 0-slice z offset):
rot_sign=-1, channel not flipped, detector rows flipped; slice order reversed to match the reference DICOM order.
Primary output is mu in 1/mm (not HU). If outputs/reports/mu_water_calibration.json exists (from compare_reference.py,
derived from the REFERENCE case only) a clearly named HU estimate is also written; it is not a scanner-calibrated HU.
usage: recon_final.py SERIES OUTNAME NXY VOX_MM DZ_MM [NZ [COL_SHIFT]]
"""
import json
import sys
import time

import numpy as np

from common import OUT
from leap_recon import recon
from niiutil import write_nii

series, name, nxy, vox, dz = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
t0 = time.time()
nz = int(sys.argv[6]) if len(sys.argv) > 6 else None
col_shift = float(sys.argv[7]) if len(sys.argv) > 7 else -1.0  # empirical: see center_test.py (reference only)
f, zphys, info = recon(series, nxy, vox, dz, nz=nz, col_shift=col_shift, off_x_mm=-0.1, off_y_mm=-0.3, rot_sign=-1, ch_flip=False, row_flip=True)
f = np.asarray(f)
print("recon done", f.shape, "s", round(time.time() - t0, 1), "mu min/max", float(f.min()), float(f.max()), flush=True)
mu = np.ascontiguousarray(f[::-1])  # reference DICOM slice order (z descending)
assert np.isfinite(mu).all()
desc = f"{series} mu[1/mm], LEAP helical FBP (parallel-ray derivative + Hilbert filter, default settings); axes IPL PROVISIONAL; vox {vox}mm dz {dz}mm"
r = write_nii(mu, (dz, vox, vox), ("I", "P", "L"), OUT / f"nifti/{name}_mu_per_mm.nii.gz", desc)
cal = OUT / "reports/mu_water_calibration.json"
if cal.exists():
    mw = json.loads(cal.read_text())["mu_water_per_mm"]
    hu = (1000.0 * (mu / mw - 1.0)).astype(np.float32)
    write_nii(hu, (dz, vox, vox), ("I", "P", "L"), OUT / f"nifti/{name}_HUest_refcal.nii.gz",
              f"HU ESTIMATE = 1000*(mu/{mw:.6f}-1); mu_w fit on REFERENCE case only; IPL PROVISIONAL")
info.update(runtime_s=round(time.time() - t0, 1), z_phys_first_last_mm=[float(zphys[0]), float(zphys[-1])], nifti=r["file"],
            scan_total_views=info["n_views"], geometry_note=f"detector centre = middle {col_shift:+.1f} channel and volume offset (-0.1,-0.3) mm: EMPIRICAL (fit to the supplied reference recon, not from tags); LEAP helical-FBP filtering (derivative + Hilbert, not an axial ramp), default settings; input already beam-hardening corrected; frame IPL PROVISIONAL")
(OUT / f"reports/recon_{name}.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
