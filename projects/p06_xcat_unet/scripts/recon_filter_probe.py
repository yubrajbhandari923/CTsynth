#!/usr/bin/env python
"""Does LEAP's helical FBP respond to set_rampFilter / set_FBPlowpass? (source suggests they only act on the axial ramp path)."""

import json
import sys

import torch

from ctsynth.xcat.cases import find_copd_case
from ctsynth.xcat.leap_sim import Acquisition, fbp, forward_project, make_leap, plan_scan
from ctsynth.xcat.materials import build_material_model
from ctsynth.xcat.noise import apply_poisson_noise, make_generator
from ctsynth.xcat.phantom import make_grid, open_phantom, resample_phantom

dev = torch.device("cuda:0")
case = find_copd_case(sys.argv[1], sys.argv[2])
m = build_material_model(case.xlsx, 70.0)
grid = make_grid(case.shape_zyx, 0.8)
res = resample_phantom(open_phantom(case.bin_path, case.shape_zyx), grid, m, dev)
plan = plan_scan(grid, Acquisition())
leap = make_leap(grid, plan)
p = forward_project(leap, res.mu_mm, plan)
noisy = apply_poisson_noise(p, 1e5, make_generator(dev, 0, 1))
base = fbp(leap, noisy, grid)
out = {"default": {"ramp_id": leap.get_rampFilter() if hasattr(leap, "get_rampFilter") else None}}
for name, fn in (
    ("ramp12_RamLak", lambda: leap.set_rampFilter(12)),
    ("ramp0", lambda: leap.set_rampFilter(0)),
    ("lowpass_W2", lambda: leap.set_FBPlowpass(2.0)),
    ("lowpass_W4", lambda: leap.set_FBPlowpass(4.0)),
):
    fn()
    r = fbp(leap, noisy, grid)
    out[name] = {
        "max_abs_diff_mu_per_mm_vs_default": float((r - base).abs().max()),
        "identical": bool(torch.equal(r, base)),
    }
    leap.set_rampFilter(2)
    leap.set_FBPlowpass(1.0)
print(json.dumps(out, indent=1))
