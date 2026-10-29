#!/usr/bin/env python
"""Dataset plan: which variants each patient gets and which of the 8 GPU workers simulates which patient (greedy longest-first).

Train / val patients: 3 variants = clean, n0_1e5 (noisy baseline), var1 (one varied draw: noise, LEAP kernel, 60/70 keV, decimation).
Test patients: those 3 plus an evaluation panel (other noise levels, both kernels, dec2, e60) and the HELD-OUT acquisition settings e80, pitch1p4.
usage: make_dataset_plan.py SPLIT.json MANIFEST.csv OUT.json [--workers 8] [--seed 0]"""

import argparse
import csv
import json
import math

from ctsynth.xcat.variants import family_variants, sample_training_variant, spec_to_dict

ap = argparse.ArgumentParser()
ap.add_argument("split")
ap.add_argument("manifest")
ap.add_argument("out")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
split = json.load(open(a.split))
man = {r["patient_id"]: r for r in csv.DictReader(open(a.manifest))}
fam = family_variants()
TEST_PANEL = [
    "n0_1e6",
    "n0_3e5",
    "n0_3e4",
    "lowpass_w2",
    "lowpass_w4",
    "dec2",
    "e60",
    "e80",
    "pitch1p4",
]
plan, cost = {}, {}
for role in ("train", "val", "test"):
    for pid in split[role]:
        specs = [fam["clean"], fam["n0_1e5"], sample_training_variant(a.seed, int(pid))]
        if role == "test":
            specs += [fam[k] for k in TEST_PANEL]
        plan[pid] = {"role": role, "variants": [spec_to_dict(s) for s in specs]}
        z, y, x = (int(v) for v in man[pid]["grid_0p8mm_zyx"].split("x"))
        views = int(man[pid]["views_margin40"])
        s = (views / 11000) * math.sqrt(z * y * x / (341 * 452 * 330))
        groups = {sp.group_key() for sp in specs}
        cost[pid] = (
            s * (1.1 + 3.5 * len(groups) + 5.5 * len(specs)) + int(man[pid]["bytes"]) / 270e6
        )
bins = [[] for _ in range(a.workers)]
load = [0.0] * a.workers
for pid in sorted(cost, key=cost.get, reverse=True):
    i = load.index(min(load))
    bins[i].append(pid)
    load[i] += cost[pid]
out = {
    "seed": a.seed,
    "split": a.split,
    "workers": bins,
    "estimated_worker_seconds": load,
    "patients": plan,
}
json.dump(out, open(a.out, "w"), indent=1)
n_vol = sum(len(v["variants"]) for v in plan.values())
print(
    f"{len(plan)} patients, {n_vol} CT volumes; estimated worker seconds min/max {min(load):.0f}/{max(load):.0f} (total {sum(load) / 3600:.2f} GPU-h)"
)
