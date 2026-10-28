"""Orientation / scale evidence. Array axes are (z,y,x) for phantoms and (slice,row,col) for the reference recon.

COPD: sternum vs spine centroid -> anterior/posterior axis direction; lung volume per half of x -> right lung is larger
(3 lobes vs 2) so the larger half is patient-right (a population prior, not a proof); lung area profile along z.
Reference: inserts (labels 2,3,4) are located in the phantom and in the recon; the 8 in-plane flip/swap candidates are
scored by how well recon insert positions (relative to body centre, mm) match the phantom's.
"""
import itertools
import json

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from common import OUT, PHANTOMS, array_shape

def load(case, s):
    path, dims, _, _ = PHANTOMS[case]
    return np.asarray(np.memmap(path, dtype=np.uint8, mode="r").reshape(array_shape(dims))[::s, ::s, ::s])

def copd(case):
    s = 4
    v = load(case, s)
    body = ~np.isin(v, [0, 70])
    ids = lambda *r: np.isin(v, [i for a in r for i in (range(a[0], a[1] + 1) if isinstance(a, tuple) else [a])])
    sternum = ids(3, (55, 59)); spine = ids(9, (39, 43)); lung = ids(6, (26, 38), 60, (106, 110))
    cen = lambda m: [float(x) for x in np.argwhere(m).mean(0)]
    bb = np.argwhere(body); ext = (bb.max(0) - bb.min(0) + 1).tolist()
    bc = bb.mean(0)
    xs = np.argwhere(lung)[:, 2]
    area = lung.sum((1, 2)) * s * s
    zc = np.argwhere(lung)[:, 0]
    return {"stride": s, "array_shape": list(v.shape), "body_extent_vox_zyx": [e * s for e in ext],
            "body_centroid_zyx": bc.tolist(), "sternum_centroid_zyx": cen(sternum), "spine_centroid_zyx": cen(spine),
            "lung_centroid_zyx": cen(lung), "lung_vox_x_below_body_centre": int((xs < bc[2]).sum()),
            "lung_vox_x_above_body_centre": int((xs > bc[2]).sum()),
            "lung_z_range_vox": [int(zc.min() * s), int(zc.max() * s)],
            "lung_area_profile_vox2_every_8th_slice": area[::2].tolist(),
            "scapulas_centroid_zyx": cen(ids(6, (50, 54)) & ~lung)}

def reference():
    s = 1
    v = load("reference", 1)
    rec = nib.load(OUT / "nifti/reference_ct_provisional.nii.gz")
    hu = np.asarray(rec.dataobj)  # (slice,row,col)
    out = {}
    bodyp = v > 0
    # phantom inserts
    cen = {k: np.argwhere(v == k).mean(0) * 0.25 for k in (2, 3, 4)}
    ext = {k: (np.argwhere(v == k).max(0) - np.argwhere(v == k).min(0) + 1) * 0.25 for k in (2, 3, 4)}
    bb = np.argwhere(bodyp)
    out["phantom_insert_centroid_mm_zyx"] = {k: c.tolist() for k, c in cen.items()}
    out["phantom_insert_extent_mm_zyx"] = {k: e.tolist() for k, e in ext.items()}
    out["phantom_body_extent_mm_zyx"] = ((bb.max(0) - bb.min(0) + 1) * 0.25).tolist()
    out["phantom_body_centroid_mm_zyx"] = (bb.mean(0) * 0.25).tolist()
    # recon: body = HU>-500, inserts by HU window, within the body's eroded interior
    body = hu > -500
    lab, n = ndi.label(body)
    big = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    inner = ndi.binary_erosion(ndi.binary_fill_holes(big.max(0) if False else big), iterations=6)
    rb = np.argwhere(big)
    out["recon_body_extent_vox_slice_row_col"] = (rb.max(0) - rb.min(0) + 1).tolist()
    out["recon_body_centroid_vox"] = rb.mean(0).tolist()
    res = {}
    for name, (lo, hi) in {"hi(>300HU)": (300, 3000), "mid(100-300HU)": (100, 300), "low(-150..-40HU)": (-150, -40)}.items():
        m = inner & (hu > lo) & (hu < hi)
        m = ndi.binary_opening(m, iterations=2)
        l2, k = ndi.label(m)
        sizes = np.bincount(l2.ravel())[1:]
        if k:
            j = np.argmax(sizes) + 1
            res[name] = {"n_components": int(k), "largest_vox": int(sizes[j - 1]),
                         "centroid_vox": np.argwhere(l2 == j).mean(0).tolist(),
                         "extent_vox": (np.argwhere(l2 == j).max(0) - np.argwhere(l2 == j).min(0) + 1).tolist()}
    out["recon_inserts"] = res
    hist, edges = np.histogram(hu[inner], bins=np.arange(-200, 700, 25))
    out["recon_inner_hu_hist"] = {"edges": edges.tolist(), "counts": hist.tolist()}
    return out

if __name__ == "__main__":
    res = {"reference": reference()}
    for c in ("copd_22728", "copd_21535"):
        res[c] = copd(c)
        print(c, {k: v for k, v in res[c].items() if "profile" not in k}, flush=True)
    print(json.dumps(res["reference"], indent=1))
    (OUT / "reports").mkdir(exist_ok=True)
    (OUT / "reports/orientation_analysis.json").write_text(json.dumps(res, indent=1))
