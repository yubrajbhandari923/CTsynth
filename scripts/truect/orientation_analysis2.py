"""Second-pass orientation evidence.
COPD (array axes z,y,x; stride 4): AP axis found as x (sternum vs spine). Here: L/R via lung volume split along y and the
aorta (ID 25) offset from the spine along y (descending aorta lies to the patient's LEFT of the spine); S/I via z profiles of
lung/scapula/sternum areas. Reference: brute-force registration of recon to phantom over px and 8 in-plane/z flip options.
"""
import itertools
import json

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from common import OUT, PHANTOMS
from orientation_analysis import load

def band(a, k=16):  # sum within bands of k slices
    n = len(a) // k * k
    return a[:n].reshape(-1, k).sum(1).tolist()

def copd(case):
    s = 4
    v = load(case, s)
    isin = lambda *ids: np.isin(v, [i for a in ids for i in (range(a[0], a[1] + 1) if isinstance(a, tuple) else [a])])
    body = ~np.isin(v, [0, 70]); lung = isin(6, (26, 38), 60, (106, 110)); spine = isin(9, (39, 43))
    aorta = isin(25); scap = isin(8, (50, 54)); stern = isin(3, (55, 59)); ribs = isin(44, (45, 49))
    cy = lambda m: float(np.argwhere(m)[:, 1].mean())
    yc = cy(body)
    out = {"body_y_centre": yc, "lung_vox_y_low": int((np.argwhere(lung)[:, 1] < yc).sum()),
           "lung_vox_y_high": int((np.argwhere(lung)[:, 1] > yc).sum()),
           "aorta_y_minus_spine_y": cy(aorta) - cy(spine), "aorta_vox": int(aorta.sum())}
    # restrict spine to the aorta's z-range for a fair offset
    zr = np.argwhere(aorta)[:, 0]
    sp2 = spine.copy(); sp2[: zr.min()] = 0; sp2[zr.max():] = 0
    out["aorta_y_minus_spine_y_same_z"] = cy(aorta) - cy(sp2)
    for nme, m in dict(lung=lung, scapulas=scap, sternum=stern, body=body, aorta=aorta).items():
        out[f"zprofile_{nme}_per16slices(z index increasing)"] = band(m.sum((1, 2)))
    return out

def reference():
    ph = np.asarray(np.memmap(PHANTOMS["reference"][0], dtype=np.uint8, mode="r").reshape(320, 860, 860))
    dens = np.array([0.001205, 1, 1.19, 0.91, 1.52, 1, 1, 1, 1, 1], np.float32)[ph[::2, ::2, ::2]]  # 0.5 mm grid
    hu = np.asarray(nib.load(OUT / "nifti/reference_ct_provisional.nii.gz").dataobj).astype(np.float32)
    rb = np.argwhere(hu > -500).mean(0)  # crude body centre (slice,row,col)
    pb = np.argwhere(ph[::2, ::2, ::2] > 0).mean(0)  # centre in 0.5mm voxels
    # sample recon on a 2 mm grid over a central window; map to phantom via candidate transforms
    res = []
    sl = np.arange(60, 190, 4); rr = np.arange(-170, 171, 6); cc = np.arange(-170, 171, 6)
    S, R, C = np.meshgrid(sl, rr, cc, indexing="ij")
    base = hu[S, np.clip((rb[1] + R).astype(int), 0, 767), np.clip((rb[2] + C).astype(int), 0, 767)]
    for px, sth in itertools.product([0.58, 0.59, 0.6, 0.61, 0.62, 0.651], [0.5476]):
        for swap, fz, fy, fx in itertools.product([0, 1], [1, -1], [1, -1], [1, -1]):
            dz = (S - rb[0]) * sth * fz
            dr, dc = (R * px, C * px) if not swap else (C * px, R * px)
            Z = pb[0] + dz / 0.5; Y = pb[1] + dr * fy / 0.5; X = pb[2] + dc * fx / 0.5
            ok = (Z >= 0) & (Z < dens.shape[0] - 1) & (Y >= 0) & (Y < dens.shape[1] - 1) & (X >= 0) & (X < dens.shape[2] - 1)
            if ok.sum() < 5000:
                continue
            d = dens[np.rint(Z[ok]).astype(int), np.rint(Y[ok]).astype(int), np.rint(X[ok]).astype(int)]
            r = float(np.corrcoef(d, base[ok])[0, 1])
            res.append({"corr": r, "px": px, "swap_rowcol_to_yx": swap, "flip_z": fz, "flip_y": fy, "flip_x": fx, "n": int(ok.sum())})
    res.sort(key=lambda r: -r["corr"])
    return {"top10": res[:10], "rb_centre": rb.tolist(), "pb_centre_0p5mm": pb.tolist()}

if __name__ == "__main__":
    out = {"reference_registration": reference()}
    print(json.dumps(out["reference_registration"]["top10"], indent=0), flush=True)
    for c in ("copd_22728", "copd_21535"):
        out[c] = copd(c)
        print(c, json.dumps(out[c]), flush=True)
    (OUT / "reports/orientation_analysis2.json").write_text(json.dumps(out, indent=1))
