#!/usr/bin/env python
"""Held-out test case figure: CT, ground truth and the four arms' predictions (seed 0), axial and sagittal (array axes) through the lung centroid.
usage: make_overlay_figure.py EVALS_DIR OUT_PNG PATIENT VARIANT [VARIANT ...]"""

import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ev, out, pid, variants = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
rng = np.random.default_rng(1)
cols = rng.integers(60, 255, (16, 3)) / 255.0
cols[0] = 0
cmap = matplotlib.colors.ListedColormap(cols)
arms = ["A", "B", "C", "D"]
fig, axs = plt.subplots(2 * len(variants), 6, figsize=(21, 7.2 * len(variants)))
for vi, v in enumerate(variants):
    ct = np.asarray(nib.load(ev / "A_s0" / f"p{pid}_{v}_ct.nii.gz").dataobj)
    gt = np.asarray(nib.load(ev / "A_s0" / f"p{pid}_{v}_truth.nii.gz").dataobj)
    preds = [np.asarray(nib.load(ev / f"{a}_s0" / f"p{pid}_{v}_pred_{a}_s0.nii.gz").dataobj) for a in arms]
    lung = np.argwhere(gt == 10)
    z, y, x = (int(c) for c in lung.mean(0))
    for r, (sl, name) in enumerate(((lambda a: a[z], "axial"), (lambda a: a[:, y], "sagittal"))):
        row = 2 * vi + r
        panels = [(sl(ct), "gray", (-1000, 400), f"CT {v}"), (sl(gt), cmap, (0, 15), "ground truth")] + [(sl(p), cmap, (0, 15), f"arm {a}") for a, p in zip(arms, preds, strict=True)]
        for c, (im, cm, lim, t) in enumerate(panels):
            axs[row, c].imshow(im, cmap=cm, vmin=lim[0], vmax=lim[1], interpolation="nearest", origin="lower" if name == "sagittal" else "upper")
            axs[row, c].set_title(f"{t} ({name})", fontsize=9)
            axs[row, c].axis("off")
            if c >= 2:
                d = (sl(preds[c - 2]) == sl(gt))[sl(gt) > 0].mean()
                axs[row, c].text(0.02, 0.02, f"voxel acc (non-air) {100 * d:.1f}%", transform=axs[row, c].transAxes, color="yellow", fontsize=8)
fig.suptitle(f"test patient {pid} (never seen in training); axes in array order (orientation PROVISIONAL); class colours consistent across panels", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=75)
