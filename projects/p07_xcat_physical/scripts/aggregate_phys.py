#!/usr/bin/env python
"""Aggregate baselines, network predictions and closed-loop resimulation into tables with patient-level bootstrap CIs.

usage: aggregate_phys.py --baselines baselines_test.json --preds PREDS_DIR --resim RESIM_DIR --out OUT_DIR
Network arms are averaged over seeds per patient; the seed range of the arm mean is reported next to it. Paired differences are
bootstrapped over patients (4000 draws, seed 0); they reflect patient sampling only, not training randomness.
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--baselines", required=True)
ap.add_argument("--preds", required=True)
ap.add_argument("--resim", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
B = json.loads(Path(a.baselines).read_text())
CH = B["channel_names"]
METRICS = {
    "density_mae_body_gcm3": lambda m: m["density_mae_body"],
    "mu70_body_mae_hu": lambda m: m["mu70_body"]["mae_hu"],
    "mu80_body_mae_hu": lambda m: m["mu80_body"]["mae_hu"],
    **{
        f"support_mae_{n}_gcm3": (lambda m, k=k: m["conc_mae_support"][k]) for k, n in enumerate(CH)
    },
}
# per_patient[model][variant] = {pid: metrics}
data: dict[str, dict[str, dict[str, dict]]] = {}
for v, models in B["per_patient"].items():
    for k, pp in models.items():
        data.setdefault(k, {}).setdefault(v, {}).update({"": pp})  # baselines: single "seed"
seeds: dict[str, dict[str, dict[str, dict]]] = {}
for f in sorted(glob.glob(f"{a.preds}/*/metrics.json")):
    r = json.loads(Path(f).read_text())
    for v, pp in r["per_patient"].items():
        seeds.setdefault(r["arm"], {}).setdefault(v, {})[str(r["seed"])] = pp
pids = sorted(next(iter(next(iter(B["per_patient"].values())).values())))


def vec(model, variant, metric):
    """(n_seeds, n_patients) array."""
    src = data[model][variant] if model in data else seeds[model][variant]
    return np.array([[METRICS[metric](pp[p]) for p in pids] for pp in src.values()])


rng = np.random.default_rng(0)
boot = rng.integers(0, len(pids), size=(4000, len(pids)))
rows, diffs = [], []
models = list(data) + sorted(seeds)
variants = sorted({v for m in models for v in (data.get(m) or seeds.get(m, {}))})
for v in variants:
    for m in models:
        if v not in (data.get(m) or seeds.get(m, {})):
            continue
        for met in METRICS:
            x = vec(m, v, met)
            pm = np.nanmean(x, 0)
            ci = np.nanpercentile(np.nanmean(pm[boot], 1), [2.5, 97.5])
            rows.append(
                {
                    "variant": v,
                    "model": m,
                    "metric": met,
                    "mean": float(np.nanmean(pm)),
                    "ci_lo": ci[0],
                    "ci_hi": ci[1],
                    "seed_min": float(np.nanmean(x, 1).min()),
                    "seed_max": float(np.nanmean(x, 1).max()),
                    "n_patients": len(pids),
                }
            )
PAIRS = [
    ("B", "A"),
    ("C", "B"),
    ("C", "A"),
    ("D", "C"),
    ("A", "P1"),
    ("B", "P1"),
    ("C", "P1"),
    ("D", "P1"),
    ("C", "O1"),
]
for v in variants:
    for new, ref in PAIRS:
        try:
            for met in ("density_mae_body_gcm3", "mu70_body_mae_hu", "mu80_body_mae_hu"):
                d = np.nanmean(vec(new, v, met), 0) - np.nanmean(vec(ref, v, met), 0)
                ci = np.percentile(d[boot].mean(1), [2.5, 97.5])
                diffs.append(
                    {
                        "variant": v,
                        "diff": f"{new}-{ref}",
                        "metric": met,
                        "mean": float(d.mean()),
                        "ci_lo": ci[0],
                        "ci_hi": ci[1],
                        "patients_better": int((d < 0).sum()),
                        "n": len(pids),
                    }
                )
        except KeyError:
            continue
for name, rr in (("summary", rows), ("paired_differences", diffs)):
    if rr:
        with open(out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)
if a.resim:
    files = sorted(glob.glob(f"{a.resim}/resim_p*.json"))
    R = [json.loads(Path(f).read_text()) for f in files]
    agg = {}
    keys = [
        k
        for k in R[0]
        if k not in ("patient", "seconds")
        and isinstance(R[0][k], dict)
        and "mae_hu_body" in R[0][k]
    ]
    for k in keys:  # patient-level shared references (truth pipeline checks)
        agg[k] = {m: float(np.mean([r[k][m] for r in R])) for m in R[0][k]}
    runs = sorted(k for k in R[0] if k[:1] in "ABCD" and k[1:2] == "_")
    for run in runs:
        for k in R[0][run]:
            agg[f"{run}:{k}"] = {m: float(np.mean([r[run][k][m] for r in R])) for m in R[0][run][k]}
    (out / "resim_summary.json").write_text(
        json.dumps(
            {"n_patients": len(R), "patients": [r["patient"] for r in R], "means": agg}, indent=1
        )
    )
print("wrote", out)
