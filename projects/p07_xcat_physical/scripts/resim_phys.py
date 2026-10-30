#!/usr/bin/env python
"""Closed loop: predicted phantom -> mu(E) -> LEAP forward projection + helical FBP, compared with the same pipeline on the TRUE phantom.

For each held-out patient: truth CT at 70 and 80 keV from the ground-truth concentrations (identical geometry/FBP as the cache; the
70 keV result is also checked against the cached clean CT), then for every listed prediction directory the predicted phantom at 70 keV
(reproduction of the input) and at 80 keV (cross-energy transfer; the network never saw 80 keV).
Deterministic (noise-free expected response) comparisons are separate from stochastic ones: coupled-noise (same RNG stream applied to truth
and prediction projections) and the independent-noise floor (two realisations of the TRUE phantom).
usage: resim_phys.py --cache CACHE --patients ID ... --preds NAME=DIR ... --out OUT_DIR [--n0 1e5]
Outputs per patient: metrics JSON and (for --export patients) NIfTI of CTs/differences.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.xcat.cache import VolumeReader
from ctsynth.xcat.leap_sim import Acquisition, ScanPlan, fbp, forward_project, make_leap, plan_scan
from ctsynth.xcat.materials import ElementTable
from ctsynth.xcat.noise import apply_poisson_noise, make_generator
from ctsynth.xcat.phantom import Grid, affine_from_axcodes, mu_to_hu
from ctsynth.xcat.physical import load_basis, mu_mm_from_concentration, u16_to_conc
from ctsynth.xcat.physical_eval import ssim_axial, valid_mask

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--patients", nargs="+", required=True)
ap.add_argument(
    "--preds", nargs="+", required=True, help="NAME=DIR (DIR has preds/p<id>_clean.npy)"
)
ap.add_argument("--out", required=True)
ap.add_argument("--n0", type=float, default=1e5)
ap.add_argument("--export", nargs="*", default=[])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda:0")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
basis, et = load_basis(), ElementTable()
E = (70.0, 80.0)
KAP = {e: torch.from_numpy(basis.kappa_cm2g(e, et)).float().to(dev) for e in E}
MUW = {e: et.kappa_water(e) / 10.0 for e in E}
preds = dict(p.split("=", 1) for p in a.preds)


def ct_from_conc(c, e, leap, plan, grid, noise=None):
    """(HU volume, clean projections' tensor) for concentrations c at energy e; noise=(n0, stream) adds Poisson noise."""
    mu = mu_mm_from_concentration(c, KAP[e])
    p = forward_project(leap, mu, plan)
    if noise is not None:
        work = torch.empty_like(p)
        apply_poisson_noise(p, noise[0], make_generator(dev, a.seed, *noise[1]), out=work)
        return mu_to_hu(fbp(leap, work, grid, inplace=True), MUW[e]), p
    return mu_to_hu(fbp(leap, p.clone(), grid), MUW[e]), p


def cmp(x, y, mask, body, zsl):
    d = x - y
    return {
        "mae_hu_valid": float(d[mask].abs().mean()),
        "rmse_hu_valid": float(d[mask].square().mean().sqrt()),
        "mae_hu_body": float(d[body].abs().mean()),
        "bias_hu_body": float(d[body].mean()),
        "ssim_axial": ssim_axial(x, y, zsl),
    }


for pid in a.patients:
    t0 = time.time()
    pdir = Path(a.cache) / f"p{pid}"
    meta = json.loads((pdir / "ct_clean.json").read_text())
    c_true = torch.from_numpy(u16_to_conc(np.load(pdir / "conc_v1.npy", mmap_mode="r")[:])).to(dev)
    grid = Grid(tuple(c_true.shape[1:]), meta["vox_mm"], tuple(c_true.shape[1:]))
    plan = plan_scan(grid, Acquisition())
    assert plan.to_dict() == ScanPlan(**meta["scan_plan"]).to_dict()
    leap = make_leap(grid, plan)
    mask = valid_mask(grid, plan, dev)
    body = mask & (c_true.sum(0) > 0.01)
    zsl = mask.any(2).any(1)
    cached70 = torch.from_numpy(VolumeReader(pdir / "ct_clean.npy", "npy_i16q").read_all()).to(dev)
    R: dict = {"patient": pid}
    truth, ptruth = {}, {}
    for e in E:
        truth[e], ptruth[e] = ct_from_conc(c_true, e, leap, plan, grid)
    R["truth70_vs_cached_clean_ct"] = cmp(
        truth[70.0], cached70, mask, body, zsl
    )  # pipeline reproduces the cache
    nz = (a.n0, (int(pid), 1))
    t_noisy_a, _ = ct_from_conc(c_true, 70.0, leap, plan, grid, noise=(a.n0, (int(pid), 11)))
    t_noisy_b, _ = ct_from_conc(c_true, 70.0, leap, plan, grid, noise=(a.n0, (int(pid), 12)))
    R["noise_floor_independent_realisations_truth70"] = cmp(t_noisy_a, t_noisy_b, mask, body, zsl)
    t_noisy_c, _ = ct_from_conc(c_true, 70.0, leap, plan, grid, noise=(a.n0, (int(pid), 13)))
    del t_noisy_b
    for name, d in preds.items():
        c_pred = torch.from_numpy(u16_to_conc(np.load(Path(d) / "preds" / f"p{pid}_clean.npy"))).to(
            dev
        )
        r = {}
        ct70, pp70 = ct_from_conc(c_pred, 70.0, leap, plan, grid)
        r["ct70_deterministic"] = cmp(ct70, truth[70.0], mask, body, zsl)
        r["ct70_vs_input_clean_ct"] = cmp(ct70, cached70, mask, body, zsl)
        dp = pp70 - ptruth[70.0]
        r["proj70_residual"] = {
            "rmse_line_integral": float(dp.square().mean().sqrt()),
            "mae_line_integral": float(dp.abs().mean()),
            "truth_rms_line_integral": float(ptruth[70.0].square().mean().sqrt()),
        }
        del dp, pp70
        ct80, pp80 = ct_from_conc(c_pred, 80.0, leap, plan, grid)
        r["ct80_deterministic"] = cmp(ct80, truth[80.0], mask, body, zsl)
        dp = pp80 - ptruth[80.0]
        r["proj80_residual"] = {
            "rmse_line_integral": float(dp.square().mean().sqrt()),
            "truth_rms_line_integral": float(ptruth[80.0].square().mean().sqrt()),
        }
        del dp, pp80
        # trivial 70-keV-only cross-energy reference: reuse the 70 keV image as the 80 keV prediction (HU unchanged) and the pure P0 map
        r["ct80_if_70keV_image_reused_as_80"] = cmp(truth[70.0], truth[80.0], mask, body, zsl)
        # coupled noise: same RNG stream on predicted and true clean projections
        pc, _ = ct_from_conc(c_pred, 70.0, leap, plan, grid, noise=(a.n0, (int(pid), 13)))
        r["ct70_coupled_noise"] = cmp(pc, t_noisy_c, mask, body, zsl)
        R[name] = r
        if pid in a.export:
            import nibabel as nib

            aff = affine_from_axcodes(tuple(ct70.shape), meta["vox_mm"])
            for nm, arr in (
                ("ct70_pred_phantom", ct70),
                ("ct80_pred_phantom", ct80),
                ("ct70_diff", ct70 - truth[70.0]),
                ("ct80_diff", ct80 - truth[80.0]),
                ("ct70_true_phantom", truth[70.0]),
                ("ct80_true_phantom", truth[80.0]),
            ):
                nib.save(
                    nib.Nifti1Image(arr.float().cpu().numpy(), aff),
                    out / f"p{pid}_{name}_{nm}.nii.gz",
                )
        del ct70, ct80, pc, c_pred
        gc.collect()
        torch.cuda.empty_cache()
    R["seconds"] = time.time() - t0
    (out / f"resim_p{pid}.json").write_text(json.dumps(R))
    print(f"p{pid} done {R['seconds']:.0f}s", flush=True)
    del leap, truth, ptruth
    gc.collect()
    torch.cuda.empty_cache()
