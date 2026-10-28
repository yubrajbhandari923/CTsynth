"""Projection previews from outputs/projections/<series>/ (native raw_u16.npy + meta.npz). Values shown are
post-log beam-hardening-corrected line integrals (unitless) = slope*raw + intercept; NOT HU, NOT anatomy."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from common import OUT

series = sys.argv[1]
d = OUT / "projections" / series
raw = np.load(d / "raw_u16.npy", mmap_mode="r")
m = np.load(d / "meta.npz")
n = raw.shape[0]
li = lambda i: raw[i].astype(np.float32) * m["slope"][i] + m["intercept"][i]  # (ch, rows)
pv = OUT / "previews" / "projections"
pv.mkdir(parents=True, exist_ok=True)

views = [0, n // 4, n // 2, n - 1]
fig, ax = plt.subplots(1, 4, figsize=(14, 6))
for a, v in zip(ax, views):
    im = a.imshow(li(v).T, aspect="auto", cmap="gray")  # x = channel (900), y = detector row (64)
    a.set_title(f"view {v + 1}  ang {m['angle_rad'][v]:.2f} rad\nz {m['z_mm'][v]:.1f} mm"); a.set_xlabel("channel"); a.set_ylabel("row")
    plt.colorbar(im, ax=a, fraction=0.046, label="line integral")
fig.suptitle(f"{series}: detector images (NON-anatomical)"); fig.tight_layout(); fig.savefig(pv / f"{series}_detector_views.png", dpi=110); plt.close(fig)

nv = min(n, 2000)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, r in zip(ax, [0, 32, 63]):
    sino = (raw[:nv, :, r].astype(np.float32) * m["slope"][:nv, None] + m["intercept"][:nv, None])
    im = a.imshow(sino, aspect="auto", cmap="gray"); a.set_title(f"sinogram, detector row {r} (views 1..{nv})")
    a.set_xlabel("channel"); a.set_ylabel("view"); plt.colorbar(im, ax=a, fraction=0.046)
fig.tight_layout(); fig.savefig(pv / f"{series}_sinograms.png", dpi=110); plt.close(fig)

mid = n // 2
p = li(mid)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(p[:, 32]); ax[0].set_title(f"channel profile (row 32, view {mid + 1})"); ax[0].set_xlabel("channel")
ax[1].plot(p[450, :]); ax[1].set_title("row profile (channel 450)"); ax[1].set_xlabel("row")
if "mAs" in m.files:
    ax[2].plot(m["mAs"]); ax[2].set_title("tube mAs vs view")
else:
    ax[2].plot(m["z_mm"]); ax[2].set_title("table z (mm) vs view")
fig.tight_layout(); fig.savefig(pv / f"{series}_profiles.png", dpi=110); plt.close(fig)

sat = {"frac_raw_0": float((raw[::10] == 0).mean()), "frac_raw_65535": float((raw[::10] == 65535).mean()),
       "lineint_min": float(min(li(i).min() for i in range(0, n, 50))), "lineint_max": float(max(li(i).max() for i in range(0, n, 50)))}
(d / "stats.json").write_text(json.dumps(sat, indent=1)); print(series, sat)

# downsampled tensor, axes (views/5, channels/3, rows) -- NOT anatomical; 1 mm placeholder spacing
vs, cs = 5, 3
idx = np.arange(0, n - n % vs, vs)
ten = np.stack([li(i)[: 900 - 900 % cs].reshape(900 // cs, cs, 64).mean(1) for i in idx]).astype(np.float32)
img = nib.Nifti1Image(ten, np.eye(4))
img.header["descrip"] = b"NON-ANATOMICAL projection tensor: axes view/5, channel/3, row; line integrals"
nib.save(img, OUT / "projections" / f"{series}_projection_tensor_NONANATOMICAL.nii.gz")
print("tensor", ten.shape)
