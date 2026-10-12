# ruff: noqa: E501
"""EXP-004 figures from metrics.json (+ val_predictions.npz for the map panel). Not a project dependency:
uv run --no-project --with matplotlib --with numpy python projects/p03_unsupervised/figures.py [results_dir]

Reads ONLY metrics.json and val_predictions.npz. Anything missing in the record is annotated, not
recomputed or guessed.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
m = json.loads((res / "metrics.json").read_text())
rungs = m["rungs"]
ids = list(rungs)
labels = [f"{rid}\n{rungs[rid]['kind'].replace('_', ' ')}" for rid in ids]
x = np.arange(len(ids))


def agg(rid, key):
    a = rungs[rid]["aggregate"][key]
    return (np.nan, []) if a is None else (a["mean"], a.get("values", []))


def panel(ax, key, title, log=False):
    means, pts = [], []
    for i, rid in enumerate(ids):
        mu, vals = agg(rid, key)
        means.append(mu)
        if rungs[rid]["kind"] in ("classical", "classical_tv"):
            ax.scatter([i], [mu], marker="D", color="k", zorder=3)
        else:
            ax.scatter([i] * len(vals), vals, color="tab:blue", s=14, zorder=3)
        pts.append(vals)
    ax.bar(x, np.nan_to_num(means), color=["0.7" if rungs[r]["kind"].startswith("classical") else
                                           ("tab:orange" if r == "B2" else "tab:blue") for r in ids],
           alpha=0.35)  # fmt: skip
    for i, mu in enumerate(means):
        if np.isnan(mu):
            ax.text(i, 0, "not\ncomputed", rotation=90, ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, labels, fontsize=6)
    ax.set_title(title, fontsize=9)
    if log:
        ax.set_yscale("log")
    for i, rid in enumerate(ids):
        if rungs[rid]["kind"].startswith("classical"):
            ax.text(i, ax.get_ylim()[1], "det.", ha="center", va="top", fontsize=6)
        if rid == "B2":
            ax.text(
                i, ax.get_ylim()[1], "labels", ha="center", va="top", fontsize=6, color="tab:orange"
            )


fig, ax = plt.subplots(2, 2, figsize=(12, 8))
panel(ax[0, 0], "rmse_tissue", "tissue-regime RMSE (val; dots = seeds, diamonds = classical)")
panel(ax[0, 1], "null_cos_tissue", "null-direction |cos| on tissue (F-014)")
panel(ax[1, 0], "consistency_val", "validation consistency loss (cm^-1)^2", log=True)
panel(ax[1, 1], "mixed_voxel_frac", "mixed-voxel fraction (tissue, max < thresh)")
fig.suptitle(
    f"{m['experiment_id']} ladder, SYNTHETIC image-domain, val split (n={m['counts']['n_val']})"
)
fig.tight_layout()
fig.savefig(res / "fig_ladder.png", dpi=120)

# Map panel: validation case 0, truth vs a few rungs (seed 0), adipose and fibroglandular rows.
npz_path = res / "val_predictions.npz"
if npz_path.exists():
    d = np.load(npz_path)
    show = [r for r in ("B2", "B3", "B4b", "B4e") if f"{r}_seed0" in d or f"{r}_deterministic" in d]
    show = [r for r in show if f"{r}_seed0" in d] or show
    cols = ["truth"] + show
    fig, axs = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 4.8), squeeze=False)
    for row, mat, name in ((0, 0, "adipose"), (1, 1, "fibroglandular")):
        for col, c in enumerate(cols):
            arr = (
                d["truth_val"] if c == "truth" else d.get(f"{c}_seed0", d.get(f"{c}_deterministic"))
            )
            im = axs[row, col].imshow(arr[0, mat], vmin=0, vmax=1, cmap="viridis")
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            if row == 0:
                axs[row, col].set_title(c, fontsize=9)
            if col == 0:
                axs[row, col].set_ylabel(name, fontsize=8)
    fig.colorbar(im, ax=axs, shrink=0.6)
    fig.suptitle("val case 0: fraction maps (seed 0; B2 uses labels)", fontsize=9)
    fig.savefig(res / "fig_maps.png", dpi=120)
    plt.close(fig)
else:
    print("val_predictions.npz not found: map panel skipped")
