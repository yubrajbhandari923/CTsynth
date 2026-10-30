#!/usr/bin/env python
"""Compute detection floors from the VALIDATION-patient baseline spread and append them to PREREGISTRATION.md (section 7).

floor(metric) = 2.8 * SD_patients(best non-oracle baseline) / sqrt(n_patients_test): the difference a two-sided 5 % test with 80 % power could
resolve if paired differences were as variable as the baseline itself (conservative: pairing normally reduces the SD).
usage: freeze_floors.py --baselines baselines_val.json --prereg PREREGISTRATION.md [--n-test 10]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--baselines", required=True)
ap.add_argument("--prereg", required=True)
ap.add_argument("--n-test", type=int, default=10)
a = ap.parse_args()
B = json.loads(Path(a.baselines).read_text())
pp = B["per_patient"]["clean"]
M = {
    "body density MAE (g/cm3)": lambda m: m["density_mae_body"],
    "body mu70 MAE (HU)": lambda m: m["mu70_body"]["mae_hu"],
    "body mu80 MAE (HU)": lambda m: m["mu80_body"]["mae_hu"],
}
lines = [
    f"Frozen from baselines on the {len(next(iter(pp.values())))} VALIDATION patients (clean CT); P1 threshold {B['p1_threshold_hu']:.0f} HU fitted on {len(B['fit_patients'])} train patients.",
    "",
    "| quantity | P0 mean (SD) | P1 mean (SD) | O1 mean (SD) | floor = 2.8*SD(P1)/sqrt(n_test) |",
    "|---|---|---|---|---|",
]
floors = {}
for name, f in M.items():
    cells, sd1 = [], None
    for k in ("P0", "P1", "O1"):
        x = np.array([f(m) for m in pp[k].values()])
        cells.append(f"{x.mean():.4g} ({x.std(ddof=1):.3g})")
        if k == "P1":
            sd1 = x.std(ddof=1)
    floors[name] = 2.8 * sd1 / np.sqrt(a.n_test)
    lines.append(f"| {name} | " + " | ".join(cells) + f" | {floors[name]:.3g} |")
lines += [
    "",
    "Interpretation rule: a network-vs-baseline or arm-vs-arm difference smaller than its floor, or smaller than the seed range of the arm means, is reported as 'not detected', not as 'no effect'.",
]
p = Path(a.prereg)
s = p.read_text()
marker = "_To be appended by `scripts/freeze_floors.py` output; see below._"
assert marker in s, "floors already frozen"
p.write_text(s.replace(marker, "\n".join(lines)))
print("\n".join(lines))
