"""EXP-002 figures from results/metrics.json (+ recon npz). Not a project dependency (matplotlib):
uv run --no-project --with matplotlib --with numpy python figures.py [results_dir]"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

res = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
m = json.loads((res / "metrics.json").read_text())
cells = [c for c in m["cells"] if c["condition"] == "matched" and c["n_tot"] == 1e6]

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
lab = [f"{c['phantom']} {c['design']} k{c['k']}" for c in cells]
x = np.arange(len(cells))
cond = [c["jac_cond"]["median"] if c["identifiable"] else np.nan for c in cells]
ax[0].bar(x, cond)
for i, c in enumerate(cells):
    if not c["identifiable"]:
        ax[0].text(i, 1, "non-ident.", rotation=90, ha="center", va="bottom")
ax[0].set_yscale("log")
ax[0].set_ylabel("median cond(weighted J)")
ax[0].set_xticks(x, lab, rotation=60, ha="right")
ratio = [
    c["per_material"].get("iodine", {}).get("emp_over_crlb_median", np.nan)
    if c["identifiable"]
    else np.nan
    for c in cells
]
ax[1].bar(x, ratio)
ax[1].axhline(1, c="k")
ax[1].axhspan(0.75, 1.25, alpha=0.15)
ax[1].set_ylabel("iodine emp std / CRLB (N=1e6)")
ax[1].set_xticks(x, lab, rotation=60, ha="right")
mis = [c for c in m["cells"] if c["identifiable"] and c["condition"] != "matched"]
for key in sorted({(c["phantom"], c["design"], c["k"], c["condition"]) for c in mis}):
    pts = sorted(
        (c["n_tot"], c["per_material"]["iodine"]["bias_gcm2_rms_per_ray"])
        for c in mis
        if (c["phantom"], c["design"], c["k"], c["condition"]) == key
    )
    ax[2].plot(*zip(*pts, strict=True), "o-", label=" ".join(map(str, key)))
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].set_xlabel("N_tot")
ax[2].set_ylabel("rms iodine A bias (g/cm^2)")
ax[2].legend(fontsize=5)
fig.tight_layout()
fig.savefig(res / "fig_identifiability.png", dpi=130)

npz = sorted(res.glob("recon_*N1e+06_matched.npz"))
fig, axs = plt.subplots(len(npz), 3, figsize=(9, 3 * len(npz)), squeeze=False)
for r, f in enumerate(npz):
    d = np.load(f)
    for col, (mat, name) in enumerate([(1, "iodine"), (2, "calcium")]):
        if mat < d["rho_hat"].shape[0]:
            im = axs[r, col].imshow(d["rho_hat"][mat], vmin=0, vmax=d["rho_true"][mat].max())
            plt.colorbar(im, ax=axs[r, col])
        axs[r, col].set_title(f"{f.stem[6:]}\n{name} rho_hat (g/cm^3)", fontsize=6)
    axs[r, 2].imshow(d["rho_true"][1])
    axs[r, 2].set_title("iodine truth", fontsize=6)
fig.tight_layout()
fig.savefig(res / "fig_recon.png", dpi=110)
