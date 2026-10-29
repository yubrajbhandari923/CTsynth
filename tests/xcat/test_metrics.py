"""Metrics against analytic cases: erosion, errors, noise autocorrelation, Gaussian edge width."""

import math

import pytest
import torch
import torch.nn.functional as F

from ctsynth.xcat.metrics import edge_width_mm, erode, noise_autocorrelation, region_errors


def test_erode_and_region_errors():
    m = torch.zeros(12, 14, 16, dtype=torch.bool)
    m[2:10, 3:12, 4:14] = True
    e = erode(m, 2)
    assert int(e.sum()) == (8 - 4) * (9 - 4) * (10 - 4) and not e[2, 5, 6] and e[5, 7, 9]
    ideal = torch.zeros(12, 14, 16)
    rec = ideal + 5.0
    r = region_errors(rec, ideal, m)
    assert (
        r["mae_hu"] == r["rmse_hu"] == r["bias_hu"] == pytest.approx(5.0)
        and region_errors(rec, ideal, ~m | m & False)["n_voxels"] > 0
    )


def test_noise_autocorrelation_white_vs_smooth():
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(24, 40, 40, dtype=torch.bool)
    white = torch.randn(24, 40, 40, generator=g)
    smooth = F.avg_pool3d(white[None, None], 3, 1, 1)[0, 0]
    aw, asm = noise_autocorrelation(white, mask), noise_autocorrelation(smooth, mask)
    assert abs(aw["x_lag1"]) < 0.05 and asm["x_lag1"] > 0.4 and asm["z_lag3"] < asm["z_lag1"]


def test_edge_width_matches_gaussian_theory():
    vox, sigma_mm = 1.0, 3.0
    n = 320
    yy, xx = torch.meshgrid(
        torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij"
    )
    disk = ((yy - 160) ** 2 + (xx - 150) ** 2 < 100.0**2).float()
    k = torch.arange(-12, 13, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma_mm) ** 2)
    k /= k.sum()
    img = F.conv2d(
        F.conv2d(disk[None, None], k[None, None, None], padding=(0, 12)),
        k[None, None, :, None],
        padding=(12, 0),
    )[0, 0]
    hu = (40.0 * img - 1000.0 * (1 - img))[None].repeat(3, 1, 1)
    w = edge_width_mm(hu, (1.0, 160.0, 150.0), vox, r_max_mm=150.0)
    assert w["edge_10_90_mm"] == pytest.approx(
        2 * 1.2816 * sigma_mm, rel=0.12
    )  # Gaussian ESF: 10-90% = 2.563 sigma
    assert math.isfinite(w["inside_hu"]) and w["n_profiles"] > 50
