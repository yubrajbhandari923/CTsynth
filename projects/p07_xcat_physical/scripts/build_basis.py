#!/usr/bin/env python
"""Build and audit the global elemental-composition basis from all patient material tables.

usage: build_basis.py --tables DIR_WITH_xlsx --out data/tables/xcat_copd_basis_v1.json [--report REPORT.md]
Fails if any organ ID has a different composition in any patient (compositions must be global).
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

from ctsynth.xcat.materials import ElementTable, load_materials
from ctsynth.xcat.physical import build_basis

ap = argparse.ArgumentParser()
ap.add_argument("--tables", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--report", default=None)
a = ap.parse_args()
files = sorted(glob.glob(f"{a.tables}/challenge_materials_copd_smeared_*.xlsx"))
tabs = [load_materials(f) for f in files]
basis = build_basis(tabs)
sha = hashlib.sha256(
    "".join(Path(f).name + hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files).encode()
).hexdigest()
d = basis.to_json() | {"n_patient_tables": len(files), "sha256_of_tables": sha}
Path(a.out).write_text(json.dumps(d, indent=1))
print(f"{len(files)} tables -> {basis.k} compositions; wrote {a.out}")
if a.report:
    et = ElementTable()
    E = np.arange(30.0, 141.0)
    kap = np.stack([basis.kappa_cm2g(e, et) for e in E], axis=1)  # (K, nE)
    i70, i80 = list(E).index(70.0), list(E).index(80.0)

    def ang(u, v):
        return float(
            np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        )

    rho = {c: [] for c in range(basis.k)}
    for t in tabs:
        for c, ids in enumerate(basis.organ_ids):
            rho[c] += [t[o].density for o in ids]
    L = [
        "# XCAT COPD material audit (EXP-040)",
        "",
        f"{len(files)} patient tables, sha256 of (name, file hash) list `{sha[:16]}...`. Compositions are identical across patients for every organ ID; only densities differ (smeared per patient).",
        "",
        "| channel | name (first xlsx organ) | organ IDs | density range over all IDs/patients (g/cm3) | kappa 70 keV (cm2/g) | kappa 80 keV |",
        "|---|---|---|---|---|---|",
    ]
    for c in range(basis.k):
        L.append(
            f"| {c} | {basis.names[c]} | {list(basis.organ_ids[c])} | {min(rho[c]):.4f}-{max(rho[c]):.4f} | {kap[c, i70]:.4f} | {kap[c, i80]:.4f} |"
        )
    L += [
        "",
        "## Attenuation-shape degeneracy: angle (deg) between kappa(E) vectors",
        "",
        "Upper triangle: 30-140 keV (111 energies); lower triangle: only {70, 80} keV. Small angle = near-degenerate (cannot be told apart from attenuation shape; separable only through density-times-kappa magnitude AND priors).",
        "",
        "| | " + " | ".join(str(c) for c in range(basis.k)) + " |",
        "|---|" + "---|" * basis.k,
    ]
    for i in range(basis.k):
        row = []
        for j in range(basis.k):
            if i == j:
                row.append("-")
            elif j > i:
                row.append(f"{ang(kap[i], kap[j]):.2f}")
            else:
                row.append(f"{ang(kap[i, [i70, i80]], kap[j, [i70, i80]]):.3f}")
        L.append(f"| {i} | " + " | ".join(row) + " |")
    sv = np.linalg.svd(kap / np.linalg.norm(kap, axis=1, keepdims=True), compute_uv=False)
    sv2 = np.linalg.svd(
        kap[:, [i70, i80]] / np.linalg.norm(kap[:, [i70, i80]], axis=1, keepdims=True),
        compute_uv=False,
    )
    L += [
        "",
        f"Singular values of the normalised kappa(E) curves, 30-140 keV: {np.array2string(sv, precision=5)}; at {{70, 80}} keV: {np.array2string(sv2, precision=5)}.",
        "Structural (arithmetic) consequences: over 30-140 keV the attenuation shapes span about 2 dimensions (photoelectric/Compton-like), so no number of energies can identify more than ~2 independent combinations of the 8 concentrations from attenuation alone; one energy (70 keV) fixes 1 number per voxel.",
    ]
    Path(a.report).write_text("\n".join(L) + "\n")
    print("wrote", a.report)
