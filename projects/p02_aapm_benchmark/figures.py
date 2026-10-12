"""AAPM benchmark figures from <results_dir>/metrics.json (+ worst_case_<method>.npz). Not a project
dependency (matplotlib):
uv run --no-project --with matplotlib --with numpy python \
  projects/p02_aapm_benchmark/figures.py [results_dir]

Anything not computed is annotated on the figure (null values, methods not requested), never left as
a silent gap. Runtime is in timing.json and is not plotted here (figures read metrics.json only).
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
recs = m["methods"]
names = list(recs)
mats = list(next(iter(recs.values()))["per_material"]) if recs else []
not_computed = m.get("not_computed", {})
note = "; ".join(f"{k}: {v}" for k, v in not_computed.items())
x = np.arange(len(names))
tag = f"split={m['split_name']}  N={m['n_cases']}  seed={m['seed']}"


def _val(v: float | None) -> float:
    return np.nan if v is None else float(v)


def _annotate_null(ax: plt.Axes, xs: np.ndarray, vals: list[float]) -> None:
    for xi, v in zip(xs, vals, strict=True):
        if np.isnan(v):
            ax.text(xi, 0.02, "not\ncomputed", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7, color="firebrick")  # fmt: skip


fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# A: official scalars
a = ax[0, 0]
w = 0.27
for j, (key, label) in enumerate([("s1", "s1 (pooled RMSE)"), ("s2", "s2 (worst ROI)"),
                                  ("aggregate_rmse", "aggregate RMSE")]):  # fmt: skip
    vals = [_val(recs[n][key]) for n in names]
    a.bar(x + (j - 1) * w, vals, w, label=label)
a.set_xticks(x, names, rotation=20, ha="right")
a.set_ylabel("volume fraction (dimensionless)")
a.set_title("Official and aggregate errors")
a.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

# B: per-material RMSE, named by material
b = ax[0, 1]
w = 0.8 / max(len(mats), 1)
for j, mat in enumerate(mats):
    vals = [_val(recs[n]["per_material"][mat]["rmse"]) for n in names]
    b.bar(x + (j - (len(mats) - 1) / 2) * w, vals, w, label=mat)
b.set_xticks(x, names, rotation=20, ha="right")
b.set_ylabel("RMSE (volume fraction)")
b.set_title("Per-material RMSE")
b.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

# C: partition (fraction-conservation) errors, air and tissue SEPARATELY
c = ax[1, 0]
for j, (regime, label) in enumerate([("air", "air: mean |sum(pred)|"),
                                     ("tissue", "tissue: mean |sum(pred)-1|")]):  # fmt: skip
    vals = [_val(recs[n]["partition"][regime]["mean_abs"]) for n in names]
    c.bar(x + (j - 0.5) * 0.36, vals, 0.36, label=label)
    _annotate_null(c, x + (j - 0.5) * 0.36, vals)
c.set_xticks(x, names, rotation=20, ha="right")
c.set_ylabel("mean absolute sum error")
c.set_title("Partition (fraction-conservation) error by regime")
c.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

# D: boundary band RMSE
d = ax[1, 1]
vals = [_val(recs[n]["boundary_rmse"]) for n in names]
d.bar(x, vals, 0.5)
_annotate_null(d, x, vals)
d.set_xticks(x, names, rotation=20, ha="right")
d.set_ylabel("RMSE on boundary-band pixels")
d.set_title("Boundary band (truth sum strictly between air and tissue)")

for axis in ax.ravel():
    axis.grid(axis="y", alpha=0.3)
fig.suptitle(f"AAPM benchmark metrics  |  {tag}", fontsize=11)
footer = f"Not run: {note}" if note else "All registered methods were requested."
fig.text(0.01, 0.005, footer + "  |  provenance: manifest.json, run_meta.json",
         fontsize=7, color="dimgray")  # fmt: skip
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(res / "fig_metrics.png", dpi=130)
plt.close(fig)

# Failure cases: worst case by s1 per method, with the worst s2 ROI drawn when it lies in that case.
for name in names:
    f = res / f"worst_case_{name}.npz"
    fc = recs[name]["failure_cases"]
    if not f.exists():
        print(f"{name}: worst_case npz missing; failure figure not computed")
        continue
    d_ = np.load(f)
    case = int(d_["case"])
    fig, axs = plt.subplots(
        2, len(mats) or 1, figsize=(3.2 * max(len(mats), 1), 6.6), squeeze=False
    )
    roi = fc["worst_roi_s2"]
    for j, mat in enumerate(mats):
        for i, (key, lab) in enumerate([("truth", "truth"), ("pred", "prediction")]):
            im = axs[i, j].imshow(d_[key][j].T, origin="lower", vmin=0, vmax=1, cmap="viridis")
            axs[i, j].set_title(f"{lab}: {mat}", fontsize=8)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if roi is not None and roi["case"] == case:
                s = roi["roi"]
                axs[i, j].add_patch(plt.Rectangle((roi["x0"] - 0.5, roi["y0"] - 0.5), s, s,
                                                  fill=False, ec="red", lw=1.2))  # fmt: skip
    if roi is None or roi["case"] != case:
        note_roi = (
            "worst s2 ROI is in a different case (not drawn)" if roi else "no ROI scored (s2 = 0)"
        )
        fig.text(0.01, 0.01, note_roi, fontsize=8, color="firebrick")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, label="volume fraction")
    fig.suptitle(
        f"{name}: worst case by s1 = case {case}, s1 = {recs[name]['failure_cases']['worst_cases_by_s1'][0]['s1']:.4g}"  # noqa: E501
        f"  |  {tag}",
        fontsize=9,
    )
    fig.savefig(res / f"fig_failure_{name}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_failure_{name}.png")
print(f"wrote {res / 'fig_metrics.png'}")
