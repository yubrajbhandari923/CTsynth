"""Image-quality metrics for simulated CT versus the ideal (noise-free, box-averaged) attenuation
image. All on torch tensors.

Regions are eroded to avoid partial-volume voxels. Noise SD and texture are measured on (variant -
same-kernel clean reconstruction),
so the deterministic anatomy cancels. Edge sharpness is the 10-90% width of the averaged edge-
spread function across the body/air boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Boolean 3D erosion with a cubic structuring element of the given radius (voxels)."""
    m = mask[None, None].float()
    return (-F.max_pool3d(-m, 2 * radius + 1, 1, radius))[0, 0] > 0.5


def region_errors(
    recon_hu: torch.Tensor, ideal_hu: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    d = (recon_hu - ideal_hu)[mask]
    if d.numel() == 0:
        return {"n_voxels": 0}
    return {
        "n_voxels": int(d.numel()),
        "mae_hu": float(d.abs().mean()),
        "rmse_hu": float((d**2).mean().sqrt()),
        "bias_hu": float(d.mean()),
    }


def noise_autocorrelation(
    noise: torch.Tensor, mask: torch.Tensor, lags: tuple[int, ...] = (1, 2, 3)
) -> dict[str, float]:
    """Normalised autocovariance of a noise field along z, y, x for the given lags (pairs where
    both voxels are inside the mask).
    White noise gives ~0 at every lag; a smoother reconstruction kernel gives positive values."""
    out: dict[str, float] = {}
    n = noise * mask
    for ax, name in enumerate(("z", "y", "x")):
        for lag in lags:
            a = n.narrow(ax, 0, n.shape[ax] - lag)
            b = n.narrow(ax, lag, n.shape[ax] - lag)
            ma = mask.narrow(ax, 0, n.shape[ax] - lag) & mask.narrow(ax, lag, n.shape[ax] - lag)
            denom = float(((a * ma) ** 2).sum())
            out[f"{name}_lag{lag}"] = (
                float((a * b * ma).sum() / denom) if denom > 0 else float("nan")
            )
    return out


def edge_width_mm(
    hu: torch.Tensor,
    centre_zyx: tuple[float, float, float],
    vox_mm: float,
    n_angles: int = 72,
    r_max_mm: float = 260.0,
    edge_hu: float = -500.0,
) -> dict[str, float]:
    """10-90% width (mm) of the angle-averaged radial edge-spread function across the outer
    body/air boundary on one axial slice.
    Profiles are aligned at the radius where the profile crosses `edge_hu` (outermost crossing),
    then averaged."""
    z = int(round(centre_zyx[0]))
    sl = hu[z]
    ny, nx = sl.shape
    cy, cx = centre_zyx[1], centre_zyx[2]
    n_r = int(r_max_mm / (0.25 * vox_mm))
    r = torch.linspace(0, r_max_mm / vox_mm, n_r, device=hu.device)
    ang = torch.linspace(0, 2 * torch.pi, n_angles + 1, device=hu.device)[:-1]
    yy = cy + r[None] * torch.sin(ang)[:, None]
    xx = cx + r[None] * torch.cos(ang)[:, None]
    grid = torch.stack([(xx / (nx - 1)) * 2 - 1, (yy / (ny - 1)) * 2 - 1], dim=-1)[None]
    prof = F.grid_sample(
        sl[None, None], grid, align_corners=True, mode="bilinear", padding_mode="border"
    )[0, 0]  # (angles, radii)
    half = int(30.0 / (0.25 * vox_mm))
    aligned = []
    for p in prof:
        inside = torch.nonzero(p > edge_hu).flatten()
        if inside.numel() == 0:
            continue
        i = int(inside.max())
        if i - half < 0 or i + half >= p.numel():
            continue
        aligned.append(p[i - half : i + half])
    if not aligned:
        return {"n_profiles": 0}
    esf = torch.stack(aligned).mean(0)
    dx = 0.25 * vox_mm
    # plateau windows 4-8 mm inside / outside the aligned crossing (index `half`)
    k0 = int(4.0 / dx)
    lo = float(esf[half + k0 : half + 2 * k0].mean())  # outside (air side)
    hi = float(esf[half - 2 * k0 : half - k0].mean())  # inside (tissue side)
    # profile runs from inside (high HU) to air (low HU)
    t10, t90 = hi - 0.1 * (hi - lo), lo + 0.1 * (hi - lo)

    def cross(level: float) -> float:
        idx = torch.nonzero(esf < level).flatten()
        if idx.numel() == 0 or int(idx[0]) == 0:
            return float("nan")
        j = int(idx[0])
        f = (esf[j - 1] - level) / (esf[j - 1] - esf[j] + 1e-9)
        return (j - 1 + float(f)) * 0.25 * vox_mm

    return {
        "n_profiles": len(aligned),
        "edge_10_90_mm": abs(cross(t10) - cross(t90)),
        "inside_hu": hi,
        "outside_hu": lo,
    }
