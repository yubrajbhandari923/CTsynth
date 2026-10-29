#!/usr/bin/env python
"""Figures + table from the variation-study caches (metrics inside ct_*.json, slices_*.npz). usage: make_variation_report.py CACHE_DIR OUT_DIR"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
metas = {
    json.loads(p.read_text())["variant"]: json.loads(p.read_text())
    for p in sorted(d.glob("ct_*.json"))
}
rows = []
for k, m in metas.items():
    x = m["metrics"]
    nm = x.get("noise_muscle", {})
    rows.append(
        {
            "variant": k,
            "family": m["family"],
            "n0": m["n0"],
            "energy_keV": m["energy_keV"],
            "pitch": m["pitch"],
            "view_decimation": m["view_decimation"],
            "apod_sigma_px": m["apod_sigma_px"],
            "lowpass_w": m.get("lowpass_w", 1.0),
            "n_views": m["n_views"],
            "body_mae_hu": x["body"]["mae_hu"],
            "body_rmse_hu": x["body"]["rmse_hu"],
            "body_bias_hu": x["body"]["bias_hu"],
            "lung_mae_hu": x["lung"]["mae_hu"],
            "lung_rmse_hu": x["lung"]["rmse_hu"],
            "noise_sd_muscle_hu": nm.get("sd_hu"),
            "acf_x_lag1": nm.get("acf_x_lag1"),
            "acf_y_lag1": nm.get("acf_y_lag1"),
            "acf_z_lag1": nm.get("acf_z_lag1"),
            "edge_10_90_mm": x["edge"].get("edge_10_90_mm"),
            "edge_ideal_mm": x["edge_ideal_reference"].get("edge_10_90_mm"),
            "fbp_s": m["times_s"]["fbp"],
            "peak_gpu_MiB_fbp": m["peak_device_MiB_fbp"],
        }
    )
    for g, e in x["per_group"].items():
        rows[-1][f"mae_{g}"] = e["mae_hu"]
with open(out / "variation_metrics.csv", "w", newline="") as fh:
    cols = sorted({c for r in rows for c in r}, key=lambda c: (c not in rows[0], c))
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
order = [
    "clean",
    "n0_1e6",
    "n0_3e5",
    "n0_1e5",
    "n0_3e4",
    "lowpass_w2",
    "lowpass_w4",
    "apod_0p75",
    "e60",
    "e80",
    "dec2",
    "pitch1p4",
]
sl = {k: np.load(d / f"slices_{k}.npz") for k in order}
ideal = sl["clean"]["ideal_axial"].astype(np.float32)
for name, win, crop in (
    ("lung_window", (-1000, 400), None),
    ("soft_tissue_window_zoom", (-160, 240), (90, 270, 120, 330)),
):
    fig, ax = plt.subplots(3, 5, figsize=(20, 12.5))
    ax = ax.ravel()
    panels = [("ideal (noise-free, box-avg)", ideal)] + [
        (k, sl[k]["axial"].astype(np.float32)) for k in order
    ]
    for a_, (t, im) in zip(ax, panels, strict=False):
        if crop:
            im = im[crop[0] : crop[1], crop[2] : crop[3]]
        a_.imshow(im, cmap="gray", vmin=win[0], vmax=win[1], origin="upper")
        r = metrics = rows[[x["variant"] for x in rows].index(t)] if t in metas else None
        sub = f"{t}" + (
            f"\nbody MAE {r['body_mae_hu']:.1f} HU"
            + (f", noise SD {r['noise_sd_muscle_hu']:.1f}" if r["noise_sd_muscle_hu"] else "")
            if r
            else ""
        )
        a_.set_title(sub, fontsize=9)
        a_.axis("off")
    for a_ in ax[len(panels) :]:
        a_.axis("off")
    fig.suptitle(
        f"copd_22728 @0.8 mm, same axial slice (array orientation, provisional): {name} {win}; N0 = incident photons per ray per view (not a calibrated mAs)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / f"variants_same_slice_{name}.png", dpi=90)
    plt.close(fig)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
nz = [r for r in rows if r["noise_sd_muscle_hu"] and r["family"] == "B"]
ax[0].loglog([r["n0"] for r in nz], [r["noise_sd_muscle_hu"] for r in nz], "o-")
ax[0].loglog(
    [r["n0"] for r in nz],
    [nz[1]["noise_sd_muscle_hu"] * (nz[1]["n0"] / r["n0"]) ** 0.5 for r in nz],
    "k--",
    lw=0.8,
    label="1/sqrt(N0)",
)
ax[0].set_xlabel("N0")
ax[0].set_ylabel("muscle noise SD [HU]")
ax[0].legend()
ax[1].bar([r["variant"] for r in rows], [r["body_mae_hu"] for r in rows])
ax[1].tick_params(axis="x", rotation=90)
ax[1].set_ylabel("body MAE vs ideal [HU]")
ax[2].bar([r["variant"] for r in rows], [r["edge_10_90_mm"] or 0 for r in rows])
ax[2].tick_params(axis="x", rotation=90)
ax[2].set_ylabel("edge 10-90% [mm]")
fig.tight_layout()
fig.savefig(out / "variation_summary.png", dpi=90)
print(len(rows), "variants")
