#!/usr/bin/env python
"""Write per-patient physical targets `conc_v1.npy` (uint16, step 1/8192 g/cm3, shape (K, nz, ny, nx)) next to the cached CT and validate them.

Per patient: closure sum_k c_k kappa_k(E)/10 vs the simulator's box-averaged mu (70, 80 keV) and density, agreement with the cached
group labels, and an alignment scan against the cached clean CT (mean |dHU| for integer voxel shifts; the minimum must be at 0).
Resumable (skips patients whose conc_v1.json exists). usage: build_targets.py --root DATA --cache CACHE --patients ID ... [--worker i --plan P]
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ctsynth.xcat.cache import VolumeReader
from ctsynth.xcat.cases import find_copd_case
from ctsynth.xcat.materials import build_material_model, load_materials
from ctsynth.xcat.phantom import make_grid, mu_to_hu, open_phantom, resample_phantom
from ctsynth.xcat.physical import (
    CONC_STEP,
    conc_to_u16,
    density_lut,
    load_basis,
    mu_mm_from_concentration,
    resample_concentrations,
    u16_to_conc,
)

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--patients", nargs="+", required=True)
ap.add_argument("--vox", type=float, default=0.8)
a = ap.parse_args()
dev = torch.device("cuda:0")
basis = load_basis()


def run(pid):
    t0 = time.perf_counter()
    pdir = Path(a.cache) / f"p{pid}"
    if (pdir / "conc_v1.json").exists():
        print(f"p{pid}: done, skipping", flush=True)
        return
    case = find_copd_case(a.root, pid)
    mats = load_materials(case.xlsx)
    basis.check(mats)
    vol = open_phantom(case.bin_path, case.shape_zyx)
    grid = make_grid(case.shape_zyx, a.vox)
    c = resample_concentrations(vol, grid, basis, density_lut(mats), dev)
    rec = {
        "patient": pid,
        "grid_zyx": list(grid.shape),
        "vox_mm": a.vox,
        "basis_k": basis.k,
        "conc_step": CONC_STEP,
    }
    assert float(c.min()) >= 0.0
    cq = u16_to_conc(conc_to_u16(c.cpu().numpy()))  # what training will read
    cqt = torch.from_numpy(cq).to(dev)
    clean = VolumeReader(pdir / "ct_clean.npy", "npy_i16q").read_all()
    assert tuple(clean.shape) == grid.shape, (clean.shape, grid.shape)
    for e in (70.0, 80.0):
        m = build_material_model(case.xlsx, e)
        res = resample_phantom(vol, grid, m, dev)
        kap = torch.from_numpy(basis.kappa_cm2g(e)).to(dev)
        mu_f, mu_q = mu_mm_from_concentration(c, kap), mu_mm_from_concentration(cqt, kap)
        ref = res.mu_mm
        rec[f"closure_{int(e)}"] = {
            "max_abs_mu_mm": float((mu_f - ref).abs().max()),
            "rel_to_max": float((mu_f - ref).abs().max() / ref.max()),
            "quantised_max_abs_hu": float(1000 * (mu_q - ref).abs().max() / m.mu_water_mm),
            "quantised_mean_abs_hu": float(1000 * (mu_q - ref).abs().mean() / m.mu_water_mm),
        }
        if e == 70.0:
            rec["density_closure_max_abs"] = float((c.sum(0) - res.density).abs().max())
            ideal = mu_to_hu(ref, m.mu_water_mm)
            labels = np.load(pdir / "labels.npy")
            # group -> channel map from the tables (must be a function)
            g2c: dict[int, set[int]] = {}
            for oid in mats:
                g2c.setdefault(int(m.organ_to_group[oid]), set()).add(
                    int(basis.organ_to_channel[oid])
                )
            assert all(len(v) == 1 for v in g2c.values()), g2c
            lut = np.zeros(256, np.int64)
            for g, v in g2c.items():
                lut[g] = next(iter(v))
            expect = torch.from_numpy(lut[labels]).to(dev)
            dom = c.argmax(0)
            body = torch.from_numpy(labels > 0).to(dev)
            rec["dominant_channel_vs_group_label_agreement"] = float(
                (dom == expect)[body].float().mean()
            )
            # alignment: shift the cached clean CT against the HU of the concentration-derived mu
            hu_c = mu_to_hu(mu_mm_from_concentration(cqt, kap), m.mu_water_mm)
            ct = torch.from_numpy(clean).to(dev)
            inner = (slice(8, -8),) * 3
            scan = {}
            for ax in range(3):
                for s in (-2, -1, 0, 1, 2):
                    sh = torch.roll(ct, s, dims=ax)
                    scan[f"ax{ax}_{s:+d}"] = float((sh - hu_c)[inner].abs().mean())
            rec["align_mean_abs_hu"] = scan
            rec["align_min_at_zero"] = all(
                min((scan[f"ax{ax}_{s:+d}"], s) for s in (-2, -1, 0, 1, 2))[1] == 0
                for ax in range(3)
            )
            rec["clean_ct_vs_phantom_mean_abs_hu"] = float((ct - hu_c)[inner].abs().mean())
            rec["clean_ct_vs_phantom_body_mean_abs_hu"] = float((ct - hu_c).abs()[body].mean())
        del res
    np.save(pdir / "conc_v1.npy", conc_to_u16(c.cpu().numpy()))
    rec["bytes"] = (pdir / "conc_v1.npy").stat().st_size
    rec["mean_conc_gcm3"] = [float(x) for x in c.mean((1, 2, 3))]
    rec["seconds"] = time.perf_counter() - t0
    (pdir / "conc_v1.json").write_text(json.dumps(rec, indent=1))
    print(
        f"p{pid}: closure70 {rec['closure_70']['rel_to_max']:.2e} 80 {rec['closure_80']['rel_to_max']:.2e} quantHU {rec['closure_70']['quantised_max_abs_hu']:.3f} "
        f"agree {rec['dominant_channel_vs_group_label_agreement']:.4f} align0 {rec['align_min_at_zero']} {rec['seconds']:.0f}s",
        flush=True,
    )
    del c, cqt, ct


for pid in a.patients:
    try:
        run(pid)
    except Exception:
        traceback.print_exc()
