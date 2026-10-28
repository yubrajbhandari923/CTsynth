"""Previews + HTML report. Everything is read back from the written NIfTIs (reopen-and-inspect check).
Display = radiological (patient Right on image left) from the canonical-RAS reorientation of each NIfTI's own affine,
so the letters on the edges are the orientation the *file claims* (provisional until confirmed by a human)."""
import html
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

from common import OUT

PV = OUT / "previews"
PV.mkdir(parents=True, exist_ok=True)

def canon(path):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj), img.header.get_zooms()[:3]

def planes(a, zoom, c):
    ax = a[::-1, :, c[2]].T            # axial: x flipped -> R on image left, A up
    co = a[::-1, c[1], :].T            # coronal: S up
    sa = a[c[0], ::-1, :].T            # sagittal: A on image left (y flipped), S up
    asp = [zoom[1] / zoom[0], zoom[2] / zoom[0], zoom[2] / zoom[1]]
    return [(ax, "R", "L", "A", "P", asp[0]), (co, "R", "L", "S", "I", asp[1]), (sa, "A", "P", "S", "I", asp[2])]

def triplanar(path, out, title, cmap, vmin=None, vmax=None, centre=None, cbar=None):
    a, zoom = canon(path)
    if centre is None:
        nz = np.argwhere(a > (a.min() if a.dtype != np.uint8 else 0))
        centre = ((nz.min(0) + nz.max(0)) // 2) if len(nz) else np.array(a.shape) // 2
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax_, (im, l, r, t, b, asp), nm in zip(axs, planes(a, zoom, centre), ["axial", "coronal", "sagittal"]):
        h = ax_.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest", aspect=asp)
        ax_.set_title(nm); ax_.set_xticks([]); ax_.set_yticks([])
        kw = dict(color="yellow", fontsize=14, fontweight="bold", transform=ax_.transAxes, bbox=dict(fc="k", alpha=.5, pad=1))
        ax_.text(.02, .5, l, ha="left", va="center", **kw); ax_.text(.98, .5, r, ha="right", va="center", **kw)
        ax_.text(.5, .97, t, ha="center", va="top", **kw); ax_.text(.5, .03, b, ha="center", va="bottom", **kw)
        if cbar:
            plt.colorbar(h, ax=ax_, fraction=0.046, label=cbar)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)

def label_cmap(leg):
    base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    cols = np.zeros((256, 4)); cols[:, 3] = 1
    for i, k in enumerate(sorted(int(k) for k in leg)):
        cols[k] = base[i % 20] if k else (0, 0, 0, 1)
    return ListedColormap(cols)

def legend_png(case, leg, out):
    ks = sorted(int(k) for k in leg); cm = label_cmap(leg)
    nrow = (len(ks) + 2) // 3
    fig, ax = plt.subplots(figsize=(14, 0.28 * nrow + 1)); ax.axis("off")
    for i, k in enumerate(ks):
        x, y = (i // nrow) * 0.34, 1 - (i % nrow + 1) / (nrow + 1)
        ax.add_patch(plt.Rectangle((x, y), 0.02, 0.8 / (nrow + 1), color=cm(k), transform=ax.transAxes))
        v = leg[str(k)]
        ax.text(x + 0.03, y, f"{k}: {v['name']}  ({v['density_g_cm3']:.3f} g/cm3)", fontsize=8, transform=ax.transAxes)
    ax.set_title(f"{case}: label ID -> name (density from this case's own spreadsheet)")
    fig.savefig(out, dpi=100, bbox_inches="tight"); plt.close(fig)

cases = sys.argv[1:] or ["reference", "copd_22728", "copd_21535"]
items = []
for c in cases:
    leg = json.loads((OUT / f"nifti/{c}_legend.json").read_text())["legend"]
    legend_png(c, leg, PV / f"{c}_legend.png")
    lab = OUT / f"nifti/{c}_material_labels_provisional.nii.gz"
    lab_s = next(iter(sorted((OUT / "nifti").glob(f"{c}_material_labels_s*_provisional.nii.gz"))), lab)
    triplanar(lab_s, PV / f"{c}_labels.png", f"{c}: material labels (PROVISIONAL orientation; letters = file's claim)", label_cmap(leg), 0, 255)
    triplanar(OUT / f"nifti/{c}_density_provisional.nii.gz", PV / f"{c}_density.png",
              f"{c}: density g/cm3 (lookup from case xlsx; PROVISIONAL orientation)", "gray", 0, 1.6, cbar="g/cm3")
    items.append(c)

ct = OUT / "nifti/reference_ct_provisional.nii.gz"
if ct.exists() and "reference" in cases:
    a, z = canon(ct)
    for nm, (lo, hi) in {"lung": (-1400, 200), "soft": (-160, 240), "full": (-1000, 1100)}.items():
        triplanar(ct, PV / f"reference_ct_{nm}.png", f"reference CT recon, window {lo}..{hi} HU (HU ASSUMED; orientation/px PROVISIONAL)",
                  "gray", lo, hi, cbar="HU(assumed)")
    # recon vs phantom at the insert level (both from their own NIfTI affines)
    pa, pz = canon(OUT / "nifti/reference_density_provisional.nii.gz")
    ra, rz = a, z
    pc = np.array(np.where(pa == pa.max())).mean(1).astype(int)   # centre of the densest insert
    rc = np.array(np.where(ra == ra.max())).mean(1).astype(int) if False else np.array(np.argwhere(ra > 500).mean(0), int)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for row, (arr, zm, cc, nm, vmin, vmax) in enumerate([(ra, rz, rc, "recon HU", -200, 900), (pa, pz, pc, "phantom density", 0.8, 1.6)]):
        for ax_, (im, l, r, t, b, asp), pn in zip(axs[row], planes(arr, zm, cc), ["axial", "coronal", "sagittal"]):
            ax_.imshow(im, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", aspect=asp); ax_.set_title(f"{nm} {pn}")
            ax_.set_xticks([]); ax_.set_yticks([])
            for (x, y, s, ha, va) in [(.02, .5, l, "left", "center"), (.98, .5, r, "right", "center"), (.5, .97, t, "center", "top"), (.5, .03, b, "center", "bottom")]:
                ax_.text(x, y, s, color="yellow", fontsize=13, fontweight="bold", ha=ha, va=va, transform=ax_.transAxes)
    fig.suptitle("Reference: recon (top) vs phantom (bottom), both displayed from their own provisional affines"); fig.tight_layout()
    fig.savefig(PV / "reference_recon_vs_phantom.png", dpi=90); plt.close(fig)

for hu in sorted((OUT / "nifti").glob("*_HUest_refcal.nii.gz")):
    for nm, (lo, hi) in {"lung": (-1400, 200), "soft": (-160, 240)}.items():
        triplanar(hu, PV / f"{hu.name[:-7]}_{nm}.png", f"{hu.name}: HU ESTIMATE (mu_water from reference case), window {lo}..{hi}; frame PROVISIONAL",
                  "gray", lo, hi, cbar="HU est")

# HTML
rows, imgs = [], []
for j in sorted((OUT / "nifti").glob("*.json")):
    if j.name.endswith("legend.json"):
        continue
    r = json.loads(j.read_text())
    rows.append(f"<tr><td>{html.escape(j.name[:-5])}</td><td>{r['shape']}</td><td>{[round(v, 4) for v in r['voxel_mm']]}</td>"
                f"<td>{r['orientation']}</td><td>{html.escape(r['note'])}</td></tr>")
for p in sorted(PV.glob("*.png")):
    imgs.append(f"<h3>{p.name}</h3><img src='../previews/{p.name}' style='max-width:100%'>")
for p in sorted((PV / "projections").glob("*.png")):
    imgs.append(f"<h3>projections/{p.name}</h3><img src='../previews/projections/{p.name}' style='max-width:100%'>")
(OUT / "reports").mkdir(exist_ok=True)
(OUT / "reports/index.html").write_text(
    "<html><body style='font-family:sans-serif'><h1>TrueCT pilot - provisional orientation, see README</h1>"
    "<table border=1 cellpadding=4><tr><th>file</th><th>shape</th><th>voxel mm</th><th>aff2axcodes</th><th>note</th></tr>"
    + "".join(rows) + "</table>" + "".join(imgs) + "</body></html>")
print("report done")
