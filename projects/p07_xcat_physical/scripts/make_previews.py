#!/usr/bin/env python
"""Programmatic NIfTI check + side-by-side previews (true vs predicted material maps, input CT, reproduced CT, differences).
Asserts that every volume of a patient shares shape and affine. usage: make_previews.py --resim DIR --patients ID ... --arm C_s0 --out DIR"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--resim", required=True)
ap.add_argument("--patients", nargs="+", required=True)
ap.add_argument("--arm", default="C_s0")
ap.add_argument("--out", required=True)
a = ap.parse_args()
R, out = Path(a.resim), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
CN = R / "conc_nifti"
report = {}
for pid in a.patients:
    files = sorted(R.glob(f"p{pid}_*.nii.gz")) + sorted(CN.glob(f"p{pid}_*.nii.gz"))
    ref = nib.load(files[0])
    bad, stats = [], {}
    for f in files:
        im = nib.load(f)
        if im.shape != ref.shape or not np.allclose(im.affine, ref.affine):
            bad.append(f.name)
        arr = np.asarray(im.dataobj)
        stats[f.name] = {
            "finite": bool(np.isfinite(arr).all()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    report[pid] = {
        "n_files": len(files),
        "shape": list(ref.shape),
        "voxel_mm": [float(x) for x in nib.affines.voxel_sizes(ref.affine)],
        "axcodes": "".join(nib.aff2axcodes(ref.affine)),
        "inconsistent_shape_or_affine": bad,
        "all_finite": all(v["finite"] for v in stats.values()),
        "orientation_status": "provisional (not validated against anatomy)",
        "files": stats,
    }
    ld = lambda n: np.asarray(nib.load(R / f"p{pid}_{n}.nii.gz").dataobj)  # noqa: E731
    lc = lambda n: np.asarray(nib.load(CN / f"p{pid}_{n}.nii.gz").dataobj)  # noqa: E731
    dens_t, dens_p = lc("density_true_gcm3"), lc(f"density_{a.arm}_gcm3")
    ct_in, t70, p70 = (
        ld("ct70_input_cached_HU"),
        ld("ct70_recon_true_phantom_HU"),
        ld(f"ct70_recon_pred_{a.arm}_HU"),
    )
    t80, p80 = ld("ct80_recon_true_phantom_HU"), ld(f"ct80_recon_pred_{a.arm}_HU")
    # axial slice through the mid of the body, coronal through the body centre
    z = int(np.argwhere(dens_t.sum((1, 2)) > 0)[:, 0].mean())
    y = int(dens_t.shape[1] // 2)
    leg = json.loads((CN / "legend.json").read_text())["channels"]
    for plane, sl in (("axial", lambda v: v[z]), ("coronal", lambda v: v[:, y])):
        rows = [
            [
                ("density true g/cm3", sl(dens_t), (0, 1.3), "gray"),
                ("density pred", sl(dens_p), (0, 1.3), "gray"),
                ("pred - true g/cm3", sl(dens_p - dens_t), (-0.2, 0.2), "RdBu_r"),
            ],
            [
                ("input CT 70keV HU", sl(ct_in), (-1000, 400), "gray"),
                ("recon true phantom 70", sl(t70), (-1000, 400), "gray"),
                ("recon pred phantom 70", sl(p70), (-1000, 400), "gray"),
            ],
            [
                ("pred-true recon 70 HU", sl(p70 - t70), (-60, 60), "RdBu_r"),
                ("recon true 80", sl(t80), (-1000, 400), "gray"),
                ("pred-true recon 80 HU", sl(p80 - t80), (-60, 60), "RdBu_r"),
            ],
        ]
        for ch in ("conc3_muscle", "conc4_lung", "conc2_sternum", "conc6_skin"):
            if ch in leg:
                tn = [k for k in leg if k == ch][0]
                full = {f.name for f in CN.glob(f"p{pid}_{tn}_*")}
                tf = [n for n in full if "_true_" in n][0]
                pf = [n for n in full if f"_{a.arm}_" in n][0]
                t_, p_ = (
                    np.asarray(nib.load(CN / tf).dataobj),
                    np.asarray(nib.load(CN / pf).dataobj),
                )
                rows.append(
                    [
                        (f"{ch} true", sl(t_), (0, 1.3), "gray"),
                        (f"{ch} pred", sl(p_), (0, 1.3), "gray"),
                        (f"{ch} pred-true", sl(p_ - t_), (-0.3, 0.3), "RdBu_r"),
                    ]
                )
        fig, ax = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)))
        for i, row in enumerate(rows):
            for j, (t, im, (lo, hi), cm) in enumerate(row):
                im = np.flipud(im) if plane == "coronal" else im
                h = ax[i, j].imshow(im, vmin=lo, vmax=hi, cmap=cm)
                ax[i, j].set_title(t, fontsize=8)
                ax[i, j].axis("off")
                fig.colorbar(h, ax=ax[i, j], fraction=0.04)
        fig.suptitle(
            f"p{pid} {plane}, arm {a.arm} (held-out test patient; orientation provisional)",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out / f"preview_p{pid}_{plane}_{a.arm}.png", dpi=110)
        plt.close(fig)
(out / "nifti_consistency.json").write_text(json.dumps(report, indent=1))
print(
    {
        k: (v["n_files"], v["inconsistent_shape_or_affine"], v["all_finite"])
        for k, v in report.items()
    }
)
