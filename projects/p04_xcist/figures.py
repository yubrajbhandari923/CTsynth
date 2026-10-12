"""Figure for dataset v0 + mismatch ladder. Run in the XCIST env (has matplotlib):
`.venv-xcist/bin/python projects/p04_xcist/figures.py DATASET_DIR MISMATCH_JSON OUT_PNG`"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ds, mm, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
rec = ds / "G1__kvp120_ref"
truth = np.load(rec / "truth.npy")[:, 0]  # (water, calcium, iodine, ny, nx)
rgb = np.stack([truth[1] / 0.2, truth[0] * 0.5 + 0.05, truth[2] / 0.03], -1).clip(0, 1)
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
ax[0, 0].imshow(rgb, extent=(-128, 128, -128, 128))
ax[0, 0].set(
    title="G1 truth (R=Ca fraction/0.2, G=water, B=I fraction/0.03)", xlabel="x mm", ylabel="y mm"
)
im = ax[0, 1].imshow(
    np.load(rec / "recon.npy"), cmap="gray", vmin=-300, vmax=600, extent=(-128, 128, -128, 128)
)
ax[0, 1].set(title="XCIST FDK recon, 120 kVp (HU, nominal mu=0.02/mm)", xlabel="x mm")
fig.colorbar(im, ax=ax[0, 1], fraction=0.046)
p = np.load(rec / "prep.npy")
im = ax[1, 0].imshow(p, aspect="auto", cmap="magma", extent=(0, 900, 360, 0))
ax[1, 0].set(title="XCIST prep = -ln(T), view x column", xlabel="detector column", ylabel="view")
fig.colorbar(im, ax=ax[1, 0], fraction=0.046)
stages = [
    "M1_analytic_xcistphys",
    "M1v_voxel_chords",
    "M2_our_kappa",
    "M3_ideal_detector",
    "M4_our_spectrum",
]
labels = [
    "M1 XCIST\ningredients",
    "M1v voxel\nchords",
    "M2 our\nkappa",
    "M3 ideal\ndetector",
    "M4 our\nspectrum",
]
w = 0.13
for i, cell in enumerate(mm):
    ax[1, 1].bar(
        np.arange(5) + (i - 2.5) * w,
        [100 * mm[cell][s]["vs_xcist"]["rel_rms_obj"] for s in stages],
        w,
        label=cell,
    )
ax[1, 1].set(
    xticks=range(5),
    xticklabels=labels,
    ylabel="rel. RMS vs XCIST on object rays (%)",
    title="Cumulative-ingredient mismatch, 6 phantom x kVp cells",
)
ax[1, 1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=130)
print("wrote", out)
