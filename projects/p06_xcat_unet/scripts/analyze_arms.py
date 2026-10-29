#!/usr/bin/env python
"""Final comparison of arms A-D and the intensity baseline from evaluate_unet.py outputs (per-patient confusion matrices).

usage: analyze_arms.py EVAL_ROOT OUT_DIR [--group-names JSON]
EVAL_ROOT contains one directory per run named <arm>_s<seed> (e.g. C_s0) with per_variant.json, and baseline/per_variant.json.
Primary metric: macro Dice over the 14 foreground classes of the POOLED voxels of each test patient (class absent in a patient's truth and
prediction is skipped for that patient), averaged over test patients. CIs: bootstrap over test patients (paired for differences), 4000 draws.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("out")
ap.add_argument("--group-names", default=None)
a = ap.parse_args()
root, out = Path(a.root), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
K = 15
names = {str(i): str(i) for i in range(K)}
if a.group_names:
    names = json.load(open(a.group_names))
IN_DIST = [
    "clean",
    "n0_1e5",
    "var1",
    "n0_1e6",
    "n0_3e5",
    "n0_3e4",
    "lowpass_w2",
    "lowpass_w4",
    "dec2",
    "e60",
]
HELD_OUT = ["e80", "pitch1p4"]
rng = np.random.default_rng(0)


def dice_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), np.nan)


def load(path):
    return json.load(open(path)) if Path(path).exists() else None


runs = {}  # arm -> seed -> per_variant dict
for d in sorted(root.iterdir()):
    if d.is_dir() and (d / "per_variant.json").exists():
        r = load(d / "per_variant.json")
        if d.name == "baseline":
            runs.setdefault("baseline", {})[0] = r["baseline"]
        else:
            arm, seed = d.name.split("_s")
            runs.setdefault(arm, {})[int(seed)] = r["unet"]
variants = sorted(
    {v for arm in runs.values() for s in arm.values() for v in s},
    key=lambda v: (
        v not in IN_DIST + HELD_OUT,
        (IN_DIST + HELD_OUT).index(v) if v in IN_DIST + HELD_OUT else 0,
        v,
    ),
)
patients = sorted(
    {
        p
        for arm in runs.values()
        for s in arm.values()
        for v in s.values()
        for p in v["per_patient_confusion"]
    }
)


def per_patient_macro(run, v):
    """patient -> macro foreground Dice for one run and variant."""
    res = {}
    for p, cm in run[v]["per_patient_confusion"].items():
        d = dice_from_cm(np.array(cm))
        res[p] = float(np.nanmean(d[1:]))
    return res


def arm_scores(arm, v):
    """patient -> macro Dice averaged over the arm's seeds."""
    seeds = [per_patient_macro(r, v) for r in runs[arm].values() if v in r]
    if not seeds:
        return {}
    return {p: float(np.mean([s[p] for s in seeds])) for p in seeds[0]}


def boot_ci(x, n=4000):
    x = np.asarray(x)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


rows, table = [], {}
for arm in runs:
    for v in variants:
        sc = arm_scores(arm, v)
        if sc:
            mean, lo, hi = boot_ci(list(sc.values()))
            seed_means = [
                float(np.mean(list(per_patient_macro(r, v).values())))
                for r in runs[arm].values()
                if v in r
            ]
            table[(arm, v)] = sc
            rows.append(
                {
                    "arm": arm,
                    "variant": v,
                    "set": "held_out" if v in HELD_OUT else "in_distribution",
                    "macro_dice_fg": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_test_patients": len(sc),
                    "seed_min": min(seed_means),
                    "seed_max": max(seed_means),
                    "n_seeds": len(seed_means),
                }
            )
with open(out / "macro_dice_by_arm_variant.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# paired comparisons on the same test patients
comps = [
    ("B", "A"),
    ("C", "B"),
    ("C", "A"),
    ("D", "C"),
    ("D", "B"),
    ("A", "baseline"),
    ("C", "baseline"),
    ("D", "baseline"),
]
crow = []
for x, y in comps:
    if x not in runs or y not in runs:
        continue
    for group, vs in (("in_distribution_mean", IN_DIST), ("held_out_mean", HELD_OUT)) + tuple(
        (v, [v]) for v in IN_DIST + HELD_OUT
    ):
        diffs = []
        for p in patients:
            dx = [table[(x, v)][p] for v in vs if (x, v) in table and p in table[(x, v)]]
            dy = [table[(y, v)][p] for v in vs if (y, v) in table and p in table[(y, v)]]
            if dx and dy:
                diffs.append(np.mean(dx) - np.mean(dy))
        if diffs:
            m, lo, hi = boot_ci(diffs)
            crow.append(
                {
                    "comparison": f"{x} - {y}",
                    "on": group,
                    "mean_diff_macro_dice": m,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "n_patients": len(diffs),
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "patients_x_better": int(sum(d > 0 for d in diffs)),
                }
            )
with open(out / "paired_comparisons.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(crow[0]))
    w.writeheader()
    w.writerows(crow)

# per-class Dice (pooled over test patients, mean over seeds) for selected variants
sel = [
    v
    for v in ("clean", "n0_1e5", "n0_3e4", "lowpass_w4", "dec2", "e80", "pitch1p4")
    if v in variants
]
cls_rows = []
for arm in runs:
    for v in sel:
        ds = []
        for r in runs[arm].values():
            if v in r:
                ds.append(dice_from_cm(np.array(r[v]["confusion"])))
        if ds:
            d = np.nanmean(ds, axis=0)
            for k in range(K):
                cls_rows.append(
                    {
                        "arm": arm,
                        "variant": v,
                        "class_id": k,
                        "class": names.get(str(k), str(k)),
                        "dice": float(d[k]),
                    }
                )
with open(out / "per_class_dice.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(cls_rows[0]))
    w.writeheader()
    w.writerows(cls_rows)

# figures
arms = [x for x in ("baseline", "A", "B", "C", "D") if x in runs]
fig, ax = plt.subplots(figsize=(15, 4.8))
wd = 0.8 / len(arms)
for i, arm in enumerate(arms):
    ys = [
        next((r["macro_dice_fg"] for r in rows if r["arm"] == arm and r["variant"] == v), np.nan)
        for v in variants
    ]
    lo = [
        next(
            (
                r["macro_dice_fg"] - r["ci_lo"]
                for r in rows
                if r["arm"] == arm and r["variant"] == v
            ),
            0,
        )
        for v in variants
    ]
    hi = [
        next(
            (
                r["ci_hi"] - r["macro_dice_fg"]
                for r in rows
                if r["arm"] == arm and r["variant"] == v
            ),
            0,
        )
        for v in variants
    ]
    ax.bar(
        np.arange(len(variants)) + (i - (len(arms) - 1) / 2) * wd,
        ys,
        wd,
        yerr=[lo, hi],
        capsize=1.5,
        label=arm,
    )
ax.set_xticks(range(len(variants)))
ax.set_xticklabels(
    [v + (" (held out)" if v in HELD_OUT else "") for v in variants], rotation=30, ha="right"
)
ax.set_ylabel("macro Dice, 14 foreground classes (test patients, 95% bootstrap CI)")
ax.legend(ncol=len(arms), frameon=False)
fig.tight_layout()
fig.savefig(out / "macro_dice_by_variant.png", dpi=110)
plt.close(fig)
for arm in [x for x in ("C", "D") if x in runs]:
    cls = np.array(
        [
            [
                r["dice"]
                for r in cls_rows
                if r["arm"] == arm and r["variant"] == v and r["class_id"] == k
            ][0]
            for v in sel
            for k in range(1, K)
        ]
    ).reshape(len(sel), K - 1)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(cls, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(K - 1))
    ax.set_xticklabels([names.get(str(k), str(k)) for k in range(1, K)], rotation=60, ha="right")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels(sel)
    fig.colorbar(im, ax=ax, label="Dice")
    ax.set_title(f"arm {arm}: per-class Dice (test patients)")
    fig.tight_layout()
    fig.savefig(out / f"per_class_dice_arm{arm}.png", dpi=110)
    plt.close(fig)
print(
    f"{len(rows)} arm/variant rows, {len(crow)} paired comparisons, {len(patients)} test patients"
)
