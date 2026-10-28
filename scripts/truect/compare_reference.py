"""Compare the LEAP reference recon (mu 1/mm) with the supplied reference recon (assumed HU). Not pixel-perfect by design:
different algorithm/filter/FOV details. Fits mu = a + b*HU inside the body (a = mu_water estimate), reports metrics, saves figures."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from common import OUT

ref = np.asarray(nib.load(OUT / "nifti/reference_ct_provisional.nii.gz").dataobj).astype(np.float32)
mu = np.asarray(nib.load(OUT / "nifti/reference_leap_recon_mu_per_mm.nii.gz").dataobj).astype(np.float32)
assert ref.shape == mu.shape, (ref.shape, mu.shape)
body = ndi.binary_erosion(ref > -500, iterations=8)
body[:6] = body[-6:] = False
b, a = np.polyfit(ref[body], mu[body], 1)  # mu = a + b*HU
mw = float(a)
hu = 1000 * (mu / mw - 1)
d = hu - ref
rmse = float(np.sqrt(np.mean(d[body] ** 2)))
sm = lambda x: ndi.gaussian_filter(x, 2.0)
dsm = sm(hu) - sm(ref)
water = body & (np.abs(ref) < 60)
rep = {"mu_water_per_mm": mw, "slope_mu_per_HU": float(b), "expected_slope_if_linear_HU_scale": mw / 1000,
       "corr_mu_vs_refHU_in_body": float(np.corrcoef(ref[body], mu[body])[0, 1]),
       "rmse_HU_after_fit_in_body": rmse, "rmse_HU_smoothed_sigma2vox": float(np.sqrt(np.mean(dsm[body] ** 2))),
       "mean_diff_HU_water_region": float(d[water].mean()), "n_body_voxels": int(body.sum()),
       "noise_sd_HU_water_ref": float(ref[water].std()), "noise_sd_HU_water_leap_est": float(hu[water].std()),
       "caveat": "fit assumes reference DICOM values are HU; different algorithms -> not a pixel-equivalence claim"}
for nm, lab in {"insert 1.19 g/cc": (100, 400), "insert 0.91 g/cc": (-150, -40), "insert 1.52 g/cc": (500, 2000)}.items():
    m = body & (sm(ref) > lab[0]) & (sm(ref) < lab[1])
    rep[f"mean_HU_{nm}"] = {"ref": float(ref[m].mean()), "leap_est": float(hu[m].mean()), "n": int(m.sum())}
(OUT / "reports/mu_water_calibration.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))

k = int(np.argwhere(ref > 500)[:, 0].mean()) if (ref > 500).any() else ref.shape[0] // 2
fig, ax = plt.subplots(3, 3, figsize=(15, 14))
for j, (nm, sl) in enumerate({"axial": (lambda v: v[k]), "coronal": (lambda v: v[:, v.shape[1] // 2]),
                              "sagittal": (lambda v: v[:, :, v.shape[2] // 2])}.items()):
    for i, (t, v, kw) in enumerate([("supplied recon (HU assumed)", ref, dict(vmin=-160, vmax=240)),
                                    ("LEAP recon (HU est, ref-water cal)", hu, dict(vmin=-160, vmax=240)),
                                    ("LEAP - supplied (2-vox smoothed)", sm(hu) - sm(ref), dict(vmin=-60, vmax=60))]):
        a_ = ax[i, j]
        img = sl(v)
        a_.imshow(img, cmap="gray" if i < 2 else "RdBu_r", origin="lower", aspect=1, **kw)
        a_.set_title(f"{t}\n{nm}"); a_.set_xticks([]); a_.set_yticks([])
fig.tight_layout(); fig.savefig(OUT / "previews/reference_leap_vs_supplied.png", dpi=80); plt.close(fig)
print("figure saved")
