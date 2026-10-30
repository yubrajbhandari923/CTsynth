#!/usr/bin/env python
"""Predict concentration maps with a trained physical U-Net (best.pt) for held-out patients/variants and score them against the
physical targets (no LEAP here). Saves predictions of --save-variants as uint16 for the closed-loop resimulation.

usage: predict_phys.py --run RUN_DIR --cache CACHE --patients ID ... --variants V ... --save-variants clean --out OUT_DIR
Scores: per-channel concentration MAE (valid region and channel support), density MAE, deterministic mu70 / mu80 error from the PREDICTED
concentrations (mu80 never sees 80 keV truth), in HU units.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.seg.physical import PhysUNet, predict_conc
from ctsynth.xcat.cache import VolumeReader
from ctsynth.xcat.leap_sim import ScanPlan
from ctsynth.xcat.materials import ElementTable
from ctsynth.xcat.phantom import Grid
from ctsynth.xcat.physical import conc_to_u16, load_basis, u16_to_conc
from ctsynth.xcat.physical_eval import physical_metrics, valid_mask

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--patients", nargs="+", required=True)
ap.add_argument("--variants", nargs="+", required=True)
ap.add_argument("--save-variants", nargs="*", default=["clean"])
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=48)
a = ap.parse_args()
dev = torch.device("cuda:0")
out = Path(a.out)
(out / "preds").mkdir(parents=True, exist_ok=True)
ck = torch.load(Path(a.run) / "best.pt", map_location=dev, weights_only=False)
cfg = ck["cfg"]
cond_on = cfg["arm"] == "D"
basis, et = load_basis(), ElementTable()
net = PhysUNet(basis.k, cfg["base"], 3, 7 if cond_on else 0).to(dev)
net.load_state_dict(ck["net"])
KAP = {e: torch.from_numpy(basis.kappa_cm2g(e, et)).float().to(dev) for e in (70.0, 80.0)}
MUW = {e: et.kappa_water(e) / 10.0 for e in (70.0, 80.0)}
res = {"run": a.run, "arm": cfg["arm"], "seed": cfg["seed"], "step": ck["step"], "per_patient": {}}
for v in a.variants:
    for pid in a.patients:
        t0 = time.time()
        pdir = Path(a.cache) / f"p{pid}"
        meta = json.loads((pdir / f"ct_{v}.json").read_text())
        hu = VolumeReader(pdir / f"ct_{v}.npy", meta["fmt"]).read_all()
        cond = torch.tensor([meta["cond"]], dtype=torch.float32) if cond_on else None
        pred = predict_conc(net, hu, 96, a.stride, 4, cond, dev)
        c = torch.from_numpy(u16_to_conc(np.load(pdir / "conc_v1.npy", mmap_mode="r")[:])).to(dev)
        mask = valid_mask(
            Grid(tuple(hu.shape), meta["vox_mm"], tuple(hu.shape)),
            ScanPlan(**meta["scan_plan"]),
            dev,
        )
        res["per_patient"].setdefault(v, {})[pid] = physical_metrics(pred, c, KAP, MUW, mask)
        if v in a.save_variants:
            np.save(out / "preds" / f"p{pid}_{v}.npy", conc_to_u16(pred.cpu().numpy()))
        m = res["per_patient"][v][pid]
        print(
            f"{v} p{pid}: density MAE {m['density_mae_body']:.4f} g/cm3, mu70 {m['mu70_body']['mae_hu']:.1f} HU, mu80 {m['mu80_body']['mae_hu']:.1f} HU ({time.time() - t0:.0f}s)",
            flush=True,
        )
        del pred, c, hu
(out / "metrics.json").write_text(json.dumps(res))
