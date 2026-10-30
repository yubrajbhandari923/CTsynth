#!/usr/bin/env python
"""Physics baselines P0 / P1 / O1 (see ctsynth.xcat.physical_eval) against the physical targets.

The P1 HU threshold is fitted on TRAIN patients' clean CT (minimising body mu80 MAE); nothing is fitted on val/test.
usage: eval_baselines.py --cache CACHE --patients ID ... --variants clean ... --fit-patients ID ... --out OUT.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ctsynth.xcat.leap_sim import ScanPlan
from ctsynth.xcat.materials import ElementTable
from ctsynth.xcat.phantom import Grid
from ctsynth.xcat.physical import load_basis, mu_mm_from_concentration, u16_to_conc
from ctsynth.xcat.physical_eval import (
    baseline_o1,
    baseline_p0,
    baseline_p1,
    physical_metrics,
    valid_mask,
)

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--patients", nargs="+", required=True)
ap.add_argument("--variants", nargs="+", default=["clean"])
ap.add_argument("--fit-patients", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
basis, et = load_basis(), ElementTable()
SOFT, BONE = basis.names.index("muscle"), basis.names.index("sternum")
KAP = {e: torch.from_numpy(basis.kappa_cm2g(e, et)).float().to(dev) for e in (70.0, 80.0)}
MUW = {e: et.kappa_water(e) / 10.0 for e in (70.0, 80.0)}


def load(pid, variant):
    pdir = Path(a.cache) / f"p{pid}"
    meta = json.loads((pdir / f"ct_{variant}.json").read_text())
    hu = torch.from_numpy(
        np.load(pdir / f"ct_{variant}.npy").astype(np.float32) * meta["hu_step"]
    ).to(dev)
    c = torch.from_numpy(u16_to_conc(np.load(pdir / "conc_v1.npy", mmap_mode="r")[:])).to(dev)
    plan = ScanPlan(**meta["scan_plan"])
    mask = valid_mask(Grid(tuple(hu.shape), meta["vox_mm"], tuple(hu.shape)), plan, dev)
    return hu, c, mask, meta


# --- fit the P1 threshold on TRAIN patients (stride-2 subsample of the valid region) -----------------------------
thr_grid = list(range(100, 901, 25))
tot = np.zeros(len(thr_grid))
for pid in a.fit_patients:
    hu, c, mask, _ = load(pid, "clean")
    hu, c, mask = hu[::2, ::2, ::2], c[:, ::2, ::2, ::2], mask[::2, ::2, ::2]
    body = mask & (c.sum(0) > 0.01)
    mu80_true = mu_mm_from_concentration(c, KAP[80.0])
    for i, t in enumerate(thr_grid):
        p = baseline_p1(hu, MUW[70.0], KAP[70.0], SOFT, BONE, float(t))
        tot[i] += float(
            ((mu_mm_from_concentration(p, KAP[80.0]) - mu80_true).abs()[body].mean())
            * 1000.0
            / MUW[80.0]
        )
    del hu, c
best_t = float(thr_grid[int(tot.argmin())])
print("P1 threshold (HU) fitted on", len(a.fit_patients), "train patients:", best_t, flush=True)

res = {
    "p1_threshold_hu": best_t,
    "fit_patients": a.fit_patients,
    "threshold_grid": thr_grid,
    "fit_objective_mean_mu80_body_hu": (tot / len(a.fit_patients)).tolist(),
    "soft_channel": SOFT,
    "bone_channel": BONE,
    "channel_names": list(basis.names),
    "per_patient": {},
}
for variant in a.variants:
    for pid in a.patients:
        hu, c, mask, meta = load(pid, variant)
        makers = {
            "P0": lambda hu=hu: baseline_p0(hu, MUW[70.0], KAP[70.0], SOFT),
            "P1": lambda hu=hu: baseline_p1(hu, MUW[70.0], KAP[70.0], SOFT, BONE, best_t),
            "O1": lambda hu=hu, c=c: baseline_o1(hu, MUW[70.0], KAP[70.0], c),
        }
        for k, make in makers.items():
            p = make()
            res["per_patient"].setdefault(variant, {}).setdefault(k, {})[pid] = physical_metrics(
                p, c, KAP, MUW, mask
            )
        print(
            variant,
            pid,
            {
                k: round(v[pid]["mu80_body"]["mae_hu"], 1)
                for k, v in res["per_patient"][variant].items()
            },
            flush=True,
        )
        del hu, c, p
Path(a.out).write_text(json.dumps(res))
