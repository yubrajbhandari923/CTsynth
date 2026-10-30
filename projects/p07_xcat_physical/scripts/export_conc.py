#!/usr/bin/env python
"""NIfTI export of true and predicted concentration maps (g/cm^3) for Slicer: total density, per-channel maps, density error.
usage: export_conc.py --cache CACHE --pred-dir DIR(preds/p<id>_clean.npy) --name NAME --patients ID ... --out OUT"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from ctsynth.xcat.phantom import affine_from_axcodes
from ctsynth.xcat.physical import load_basis, u16_to_conc

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--pred-dir", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--patients", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
basis = load_basis()
names = basis.names
(out / "legend.json").write_text(
    json.dumps(
        {
            "units": "g/cm3 (mass concentration of each elemental composition); CT in HU",
            "channels": {
                f"conc{k}_{n}": {
                    "index": k,
                    "first_xlsx_name": n,
                    "organ_ids": list(basis.organ_ids[k]),
                }
                for k, n in enumerate(names)
            },
            "note": "channels are compositions, not anatomical labels; skin, blood, muscle, lung tissue are "
            "near-degenerate in attenuation (see final_report.md). Orientation (L/R) provisional: "
            "affine from affine_from_axcodes, not validated against a clinical atlas.",
        },
        indent=1,
    )
)
for pid in a.patients:
    pdir = Path(a.cache) / f"p{pid}"
    vox = json.loads((pdir / "ct_clean.json").read_text())["vox_mm"]
    ct = u16_to_conc(np.load(pdir / "conc_v1.npy", mmap_mode="r")[:])
    cp = u16_to_conc(np.load(Path(a.pred_dir) / f"p{pid}_clean.npy"))
    aff = affine_from_axcodes(tuple(ct.shape[1:]), vox)

    def save(arr, nm):
        nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), out / f"p{pid}_{nm}.nii.gz")

    save(ct.sum(0), "density_true_gcm3")
    save(cp.sum(0), f"density_{a.name}_gcm3")
    save(cp.sum(0) - ct.sum(0), f"density_error_{a.name}_gcm3")
    for k, n in enumerate(names):
        save(ct[k], f"conc{k}_{n}_true_gcm3")
        save(cp[k], f"conc{k}_{n}_{a.name}_gcm3")
    print(pid, "exported", flush=True)
