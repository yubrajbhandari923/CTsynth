#!/usr/bin/env python
"""Material recovery vs CT reproducibility, per held-out patient and run (clean input predictions, valid region).

For each patient/run: (1) per-channel total predicted vs true mass (g) and per-channel support MAE; (2) mass confusion
matrix M[true dominant channel, predicted channel] = predicted mass of that channel inside voxels whose TRUE dominant
channel is the row, divided by the true mass of the row (rows of a perfect recovery are one-hot); (3) the support MAE of the
merged soft group {muscle, lung tissue, skin, blood}, which are near-degenerate in attenuation; (4) body mu70/mu80 error and density error for comparison. Units: g, g/cm3, HU.
usage: analyze_materials.py --cache C --preds NAME=DIR ... --patients ID ... --out JSON"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ctsynth.xcat.leap_sim import ScanPlan
from ctsynth.xcat.materials import ElementTable
from ctsynth.xcat.phantom import Grid
from ctsynth.xcat.physical import load_basis, u16_to_conc
from ctsynth.xcat.physical_eval import mu_hu_error, valid_mask

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--patients", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = torch.device("cpu")
basis, et = load_basis(), ElementTable()
K = basis.k
SOFT = [i for i, n in enumerate(basis.names) if n in ("muscle", "lung", "skin", "aorta")]
KAP = {e: torch.from_numpy(basis.kappa_cm2g(e, et)).float() for e in (70.0, 80.0)}
MUW = {e: et.kappa_water(e) / 10.0 for e in (70.0, 80.0)}
preds = dict(p.split("=", 1) for p in a.preds)
res: dict = {"channels": list(basis.names), "soft_group": SOFT, "per_patient": {}}
for pid in a.patients:
    pdir = Path(a.cache) / f"p{pid}"
    meta = json.loads((pdir / "ct_clean.json").read_text())
    ct = torch.from_numpy(u16_to_conc(np.load(pdir / "conc_v1.npy", mmap_mode="r")[:]))
    grid = Grid(tuple(ct.shape[1:]), meta["vox_mm"], tuple(ct.shape[1:]))
    mask = valid_mask(grid, ScanPlan(**meta["scan_plan"]), dev)
    vox_cm3 = (meta["vox_mm"] / 10.0) ** 3
    dom = ct.argmax(0)
    body = mask & (ct.sum(0) > 0.01)
    true_mass = [float(ct[k][mask].sum()) * vox_cm3 for k in range(K)]
    out_p: dict = {"true_mass_g": true_mass, "runs": {}}
    for name, d in preds.items():
        cp = torch.from_numpy(u16_to_conc(np.load(Path(d) / "preds" / f"p{pid}_clean.npy")))
        conf = np.zeros((K, K))
        for r in range(K):
            m = mask & (dom == r)
            denom = float(ct[r][m].sum())
            if denom > 0:
                conf[r] = [float(cp[j][m].sum()) / denom for j in range(K)]
        sup_mae = [
            float((cp[k] - ct[k]).abs()[mask & (ct[k] > 0.05)].mean())
            if bool((mask & (ct[k] > 0.05)).any())
            else None
            for k in range(K)
        ]
        soft_t, soft_p = ct[SOFT].sum(0), cp[SOFT].sum(0)
        grp = mask & (soft_t > 0.05)
        out_p["runs"][name] = {
            "pred_mass_g": [float(cp[k][mask].sum()) * vox_cm3 for k in range(K)],
            "support_mae_gcm3": sup_mae,
            "mass_confusion_true_row_pred_col": conf.tolist(),
            "soft_group_support_mae_gcm3": float((soft_p - soft_t).abs()[grp].mean()),
            "body_density_mae_gcm3": float((cp.sum(0) - ct.sum(0)).abs()[body].mean()),
            "body_mu70_mae_hu": mu_hu_error(cp, ct, KAP[70.0], MUW[70.0], body)["mae_hu"],
            "body_mu80_mae_hu": mu_hu_error(cp, ct, KAP[80.0], MUW[80.0], body)["mae_hu"],
        }
        del cp
    res["per_patient"][pid] = out_p
    print(pid, "done", flush=True)
Path(a.out).write_text(json.dumps(res))
