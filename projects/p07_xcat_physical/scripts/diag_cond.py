#!/usr/bin/env python
"""Diagnose arm D (FiLM) failure on an unseen acquisition value using VALIDATION patients only (no test data).
Frozen checkpoints; each validation CT is predicted with its true conditioning vector and with single entries overridden
(pitch 1.4, energy ratio 80/70, pitch 1.0 explicit). Also reports, per conditioning entry, its training-set range and the
size of its FiLM weights relative to the FiLM bias (a constant training column is learnt as a bias).
usage: diag_cond.py --cache CACHE --runs DIR... --patients ID ... --variants V ... --out JSON"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ctsynth.seg.physical import PhysUNet, predict_conc
from ctsynth.xcat.cache import VolumeReader
from ctsynth.xcat.leap_sim import ScanPlan
from ctsynth.xcat.materials import ElementTable
from ctsynth.xcat.phantom import Grid
from ctsynth.xcat.physical import load_basis, u16_to_conc
from ctsynth.xcat.physical_eval import physical_metrics, valid_mask

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--patients", nargs="+", required=True)
ap.add_argument("--variants", nargs="+", required=True)
ap.add_argument("--train-patients", nargs="+", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = torch.device("cuda:0")
basis, et = load_basis(), ElementTable()
KAP = {e: torch.from_numpy(basis.kappa_cm2g(e, et)).float().to(dev) for e in (70.0, 80.0)}
MUW = {e: et.kappa_water(e) / 10.0 for e in (70.0, 80.0)}
NAMES = ["noisy", "log10_n0", "apod", "lowpass-1", "pitch", "1/decim", "energy/70"]
res: dict = {"cond_names": NAMES}
# training-set range of every conditioning entry (train variants of train patients, from the metadata only)
rng = {}
for v in ("clean", "n0_1e5", "var1"):
    for pid in a.train_patients[:3]:
        c = json.loads((Path(a.cache) / f"p{pid}" / f"ct_{v}.json").read_text())["cond"]
        rng.setdefault(v, c)
res["train_variant_cond"] = rng
OVR = {
    "true": {},
    "pitch=1.4": {4: 1.4},
    "pitch=1.0": {4: 1.0},
    "energy/70=80/70": {6: 80 / 70},
    "pitch=1.4,energy/70=80/70": {4: 1.4, 6: 80 / 70},
}
for rd in a.runs:
    ck = torch.load(Path(rd) / "best.pt", map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    if cfg["arm"] != "D":
        continue
    net = PhysUNet(basis.k, cfg["base"], 3, 7).to(dev)
    net.load_state_dict(ck["net"])
    # FiLM weight column norms vs bias norm (all blocks)
    cols = np.zeros(7)
    bias = 0.0
    for m in net.modules():
        if hasattr(m, "film") and m.film is not None:
            cols += m.film.weight.detach().abs().sum(0).cpu().numpy()
            bias += float(m.film.bias.detach().abs().sum())
    out = {
        "film_abs_weight_sum_per_cond_entry": cols.tolist(),
        "film_abs_bias_sum": bias,
        "cases": {},
    }
    for v in a.variants:
        for pid in a.patients:
            pdir = Path(a.cache) / f"p{pid}"
            meta = json.loads((pdir / f"ct_{v}.json").read_text())
            hu = VolumeReader(pdir / f"ct_{v}.npy", meta["fmt"]).read_all()
            c = torch.from_numpy(u16_to_conc(np.load(pdir / "conc_v1.npy", mmap_mode="r")[:])).to(
                dev
            )
            mask = valid_mask(
                Grid(tuple(hu.shape), meta["vox_mm"], tuple(hu.shape)),
                ScanPlan(**meta["scan_plan"]),
                dev,
            )
            for name, ov in OVR.items():
                cv = list(meta["cond"])
                for i, x in ov.items():
                    cv[i] = x
                pred = predict_conc(
                    net, hu, 96, 48, 4, torch.tensor([cv], dtype=torch.float32), dev
                )
                m = physical_metrics(pred, c, KAP, MUW, mask)
                out["cases"][f"{v}|{pid}|{name}"] = {
                    "density_mae_body": m["density_mae_body"],
                    "mu70_body_hu": m["mu70_body"]["mae_hu"],
                    "mu80_body_hu": m["mu80_body"]["mae_hu"],
                    "density_bias_body": m["density_bias_body"],
                }
                print(rd, v, pid, name, round(m["mu80_body"]["mae_hu"], 2), flush=True)
                del pred
    res[Path(rd).name] = out
Path(a.out).write_text(json.dumps(res, indent=1))
