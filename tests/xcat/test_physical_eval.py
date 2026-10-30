"""Baselines and metrics: structural properties (P0/O1 reproduce mu70 exactly from exact HU; P0
cannot get mu80 right for bone), zero error for a perfect prediction, per-channel support."""

import pytest
import torch

from ctsynth.xcat.leap_sim import Acquisition, plan_scan
from ctsynth.xcat.phantom import make_grid
from ctsynth.xcat.physical import mu_mm_from_concentration
from ctsynth.xcat.physical_eval import (
    baseline_o1,
    baseline_p0,
    baseline_p1,
    hu_to_mu_mm,
    physical_metrics,
    valid_mask,
)

MU_W70, MU_W80 = 0.0193, 0.0184
KAP = {
    70.0: torch.tensor([0.175, 0.19, 0.257]),
    80.0: torch.tensor([0.166, 0.182, 0.223]),
}  # air, soft, bone


def _truth():
    c = torch.zeros(3, 6, 7, 8)
    c[1] = 1.05  # soft everywhere ...
    c[1, :, :, :3] = 0.0
    c[2, :, :, :3] = 1.4  # ... bone slab at low x (asymmetric)
    c[0, 0, :, 3:] = 0.0012  # air slice outside the bone slab
    c[1, 0, :, 3:] = 0.0
    return c


def _hu(c):
    mu = mu_mm_from_concentration(c, KAP[70.0])
    return 1000.0 * (mu / MU_W70 - 1.0)


def test_p0_and_o1_reproduce_mu70_exactly_but_p0_misses_bone_at_80():
    c, hu = _truth(), _hu(_truth())
    mask = torch.ones(c.shape[1:], dtype=torch.bool)
    p0 = baseline_p0(hu, MU_W70, KAP[70.0], ref=1)
    o1 = baseline_o1(hu, MU_W70, KAP[70.0], c)
    mu70 = mu_mm_from_concentration(c, KAP[70.0])
    assert torch.allclose(mu_mm_from_concentration(p0, KAP[70.0]), mu70, atol=1e-7)
    assert torch.allclose(o1, c, atol=1e-5)
    m0 = physical_metrics(p0, c, KAP, {70.0: MU_W70, 80.0: MU_W80}, mask)
    mo = physical_metrics(o1, c, KAP, {70.0: MU_W70, 80.0: MU_W80}, mask)
    assert m0["mu70_valid"]["mae_hu"] < 1e-2 and m0["mu80_valid"]["mae_hu"] > 5.0  # type: ignore[index]
    assert mo["mu80_valid"]["mae_hu"] < 1e-2 and mo["conc_mae_valid"][2] < 1e-5  # type: ignore[index]


def test_p1_threshold_and_perfect_prediction_has_zero_error():
    c, hu = _truth(), _hu(_truth())
    p1 = baseline_p1(hu, MU_W70, KAP[70.0], soft=1, bone=2, threshold_hu=300.0)
    assert (
        torch.allclose(p1[2][..., :3], c[2][..., :3], atol=1e-5)
        and float(p1[2][..., 3:].max()) == 0.0
    )
    mask = torch.ones(c.shape[1:], dtype=torch.bool)
    m = physical_metrics(c, c, KAP, {70.0: MU_W70, 80.0: MU_W80}, mask)
    assert m["density_mae_valid"] == 0.0 and m["mu80_body"]["mae_hu"] == 0.0  # type: ignore[index]
    assert m["support_voxels"][2] == 6 * 7 * 3  # type: ignore[index]


def test_valid_mask_excludes_scan_ends_and_corners():
    g = make_grid((40, 50, 60), 0.8)
    m = valid_mask(g, plan_scan(g, Acquisition()))
    assert m.shape == g.shape
    assert bool(m[g.shape[0] // 2, g.shape[1] // 2, g.shape[2] // 2])
    assert m.sum() > 0 and hu_to_mu_mm(torch.tensor(0.0), MU_W70) == pytest.approx(MU_W70)


def test_ssim_axial_identity_and_noise():
    from ctsynth.xcat.physical_eval import ssim_axial

    g = torch.Generator().manual_seed(0)
    a = torch.rand(6, 40, 50, generator=g) * 500
    sl = torch.ones(6, dtype=torch.bool)
    assert ssim_axial(a, a, sl) == pytest.approx(1.0, abs=1e-6)
    assert ssim_axial(a, a + 100 * torch.randn(a.shape, generator=g), sl) < 0.9
    assert ssim_axial(a, a, torch.zeros(6, dtype=torch.bool)) == 0.0
