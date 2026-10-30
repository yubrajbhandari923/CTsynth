"""Physics baselines and metrics for physical-phantom (composition concentration) inversion.

All maps are (K, nz, ny, nx) or (nz, ny, nx) torch tensors on the simulation grid. c in g/cm^3,
kappa in cm^2/g, mu in mm^-1; HU-equivalent error = 1000 * |delta mu| / mu_water(E).
Baselines use ONLY the 70 keV image (HU) plus stated assumptions:
  P0 one composition (a reference soft tissue) with continuous density = mu / kappa: reproduces mu70
     by construction (structural, not evidence) and has no energy dependence of its own;
  P1 two compositions (bone vs soft tissue) split by an HU threshold fitted on TRAIN patients;
  O1 ORACLE composition identity (true dominant channel per voxel) with density from mu70: an
     upper-bound diagnostic that uses ground truth, not an inference baseline.
"""

from __future__ import annotations

import math

import torch

from ctsynth.xcat.leap_sim import SOD_MM, ScanPlan
from ctsynth.xcat.phantom import Grid


def valid_mask(grid: Grid, plan: ScanPlan, device: torch.device | str = "cpu") -> torch.Tensor:
    """Voxels inside the helical coverage (>= 20 mm from the scan ends) and the circular FOV
    (95 % of its radius): the same region used for the EXP-039 simulator metrics."""
    nz, ny, nx = grid.shape
    v = grid.vox_mm
    zc = (torch.arange(nz, device=device).float() + 0.5) * v - nz * v / 2
    yy = ((torch.arange(ny, device=device).float() + 0.5) * v - ny * v / 2)[:, None]
    xx = ((torch.arange(nx, device=device).float() + 0.5) * v - nx * v / 2)[None, :]
    fov_r = SOD_MM * math.sin(0.5 * 900 / 1050.0)
    ok_z = zc.abs() <= plan.scan_length_mm / 2 - 20.0
    ok_r = torch.sqrt(yy**2 + xx**2) < 0.95 * fov_r
    return ok_z[:, None, None] & ok_r[None]


def hu_to_mu_mm(hu: torch.Tensor, mu_water_mm: float) -> torch.Tensor:
    return mu_water_mm * (1.0 + hu / 1000.0)


def baseline_p0(
    hu: torch.Tensor, mu_water_mm: float, kappa70: torch.Tensor, ref: int
) -> torch.Tensor:
    mu_cm = (10.0 * hu_to_mu_mm(hu, mu_water_mm)).clamp(min=0)
    out = torch.zeros((kappa70.numel(), *hu.shape), device=hu.device)
    out[ref] = mu_cm / kappa70[ref]
    return out


def baseline_p1(
    hu: torch.Tensor,
    mu_water_mm: float,
    kappa70: torch.Tensor,
    soft: int,
    bone: int,
    threshold_hu: float,
) -> torch.Tensor:
    mu_cm = (10.0 * hu_to_mu_mm(hu, mu_water_mm)).clamp(min=0)
    is_bone = hu > threshold_hu
    out = torch.zeros((kappa70.numel(), *hu.shape), device=hu.device)
    out[soft] = torch.where(is_bone, torch.zeros_like(mu_cm), mu_cm / kappa70[soft])
    out[bone] = torch.where(is_bone, mu_cm / kappa70[bone], torch.zeros_like(mu_cm))
    return out


def baseline_o1(
    hu: torch.Tensor, mu_water_mm: float, kappa70: torch.Tensor, c_true: torch.Tensor
) -> torch.Tensor:
    mu_cm = (10.0 * hu_to_mu_mm(hu, mu_water_mm)).clamp(min=0)
    dom = c_true.argmax(0)
    out = torch.zeros_like(c_true)
    out.scatter_(0, dom[None], (mu_cm / kappa70[dom])[None])
    return out


def mu_hu_error(
    c_pred: torch.Tensor,
    c_true: torch.Tensor,
    kappa_cm2g: torch.Tensor,
    mu_water_mm: float,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Deterministic attenuation error at one energy between two concentration maps (HU units)."""
    d = torch.zeros_like(c_true[0])
    for i in range(c_true.shape[0]):  # per-channel accumulation keeps GPU memory at one volume
        d += (c_pred[i] - c_true[i]) * kappa_cm2g[i].to(c_true)
    d = d.abs() / 10.0 * (1000.0 / mu_water_mm)
    return {"mae_hu": float(d[mask].mean()), "rmse_hu": float(d[mask].square().mean().sqrt())}


def physical_metrics(
    c_pred: torch.Tensor,
    c_true: torch.Tensor,
    kappa: dict[float, torch.Tensor],
    mu_water_mm: dict[float, float],
    mask: torch.Tensor,
    support_min: float = 0.05,
) -> dict[str, object]:
    """Per-channel concentration MAE (g/cm^3) over the valid region and over each channel's
    support (c_true > support_min), total-density MAE, and mu errors at every energy in `kappa`."""
    rho_t, rho_p = c_true.sum(0), c_pred.sum(0)
    body = mask & (rho_t > 0.01)
    sup = [mask & (c_true[k] > support_min) for k in range(c_true.shape[0])]
    mae_valid, mae_sup = [], []
    for k in range(c_true.shape[0]):
        ek = (c_pred[k] - c_true[k]).abs()
        mae_valid.append(float(ek[mask].mean()))
        mae_sup.append(float(ek[sup[k]].mean()) if bool(sup[k].any()) else float("nan"))
    out: dict[str, object] = {
        "conc_mae_valid": mae_valid,
        "conc_mae_support": mae_sup,
        "support_voxels": [int(m.sum()) for m in sup],
        "density_mae_valid": float((rho_p - rho_t).abs()[mask].mean()),
        "density_mae_body": float((rho_p - rho_t).abs()[body].mean()),
        "density_bias_body": float((rho_p - rho_t)[body].mean()),
    }
    for e, kap in kappa.items():
        out[f"mu{int(e)}_valid"] = mu_hu_error(c_pred, c_true, kap, mu_water_mm[e], mask)
        out[f"mu{int(e)}_body"] = mu_hu_error(c_pred, c_true, kap, mu_water_mm[e], body)
    return out


def ssim_axial(
    a: torch.Tensor, b: torch.Tensor, slices: torch.Tensor, data_range: float = 2524.0
) -> float:
    """Mean 2D SSIM (11x11 Gaussian window, sigma 1.5) over the axial slices flagged in `slices`
    (bool, nz). a, b in HU clipped to [-1024, 1500] (data_range 2524)."""
    import torch.nn.functional as F

    x = torch.arange(11, device=a.device, dtype=torch.float32) - 5
    g = torch.exp(-(x**2) / (2 * 1.5**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    out, n = 0.0, 0
    idx = torch.nonzero(slices).flatten()
    for i0 in range(0, len(idx), 32):
        ii = idx[i0 : i0 + 32]
        p = a[ii].clamp(-1024, 1500)[:, None].float()
        q = b[ii].clamp(-1024, 1500)[:, None].float()
        mp, mq = F.conv2d(p, w), F.conv2d(q, w)
        vp = F.conv2d(p * p, w) - mp * mp
        vq = F.conv2d(q * q, w) - mq * mq
        cv = F.conv2d(p * q, w) - mp * mq
        s = ((2 * mp * mq + c1) * (2 * cv + c2)) / ((mp * mp + mq * mq + c1) * (vp + vq + c2))
        out += float(s.mean((1, 2, 3)).sum())
        n += len(ii)
    return out / max(n, 1)
