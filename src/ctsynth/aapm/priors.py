"""Simple interpretable priors for the A/F-calcification ambiguity (PLAN-006 Stage 0).

Differentiable torch scalars on `frac` (N,3,H,W) in schema.MATERIALS order, restricted to a bool
`tissue` mask (N,H,W). Unweighted; adipose fraction a = f0/(f0+f1) (the A:F split independent of
total occupancy). `wrong_label` swaps A<->F inside tissue: the control that exposes priors which are
symmetric under the swap (bimodality, TV) and so cannot select the sign. `w1_marginal` uses a
marginal built from TRAIN labels: weak supervision, never label-free.
"""

from __future__ import annotations

import numpy as np
import torch


def _a(frac: torch.Tensor) -> torch.Tensor:
    return (frac[:, 0] / (frac[:, 0] + frac[:, 1]).clamp_min(1e-8)).clamp(0.0, 1.0)


def _mean_over(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def calc_l1(frac: torch.Tensor, tissue: torch.Tensor) -> torch.Tensor:
    """mean over tissue pixels of |calcification fraction|."""
    return _mean_over(frac[:, 2].abs(), tissue)


def calc_l0(frac: torch.Tensor, tissue: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """smooth L0 surrogate mean over tissue of c^2/(c^2+eps^2), in [0, 1)."""
    c2 = frac[:, 2] ** 2
    return _mean_over(c2 / (c2 + eps**2), tissue)


def bimodal(frac: torch.Tensor, tissue: torch.Tensor) -> torch.Tensor:
    """mean over tissue of a(1-a): 0 at a in {0,1}, 0.25 at a=0.5; symmetric under A<->F."""
    a = _a(frac)
    return _mean_over(a * (1 - a), tissue)


def tv_a(frac: torch.Tensor, tissue: torch.Tensor) -> torch.Tensor:
    """mean |da| over 4-neighbour tissue pairs (air edges excluded); symmetric under A<->F."""
    a = _a(frac)
    dx = (a[:, :, 1:] - a[:, :, :-1]).abs()
    mx = tissue[:, :, 1:] & tissue[:, :, :-1]
    dy = (a[:, 1:, :] - a[:, :-1, :]).abs()
    my = tissue[:, 1:, :] & tissue[:, :-1, :]
    n = mx.sum() + my.sum()
    return ((dx * mx).sum() + (dy * my).sum()) / n.clamp_min(1)


def wrong_label(frac: torch.Tensor, tissue: torch.Tensor) -> torch.Tensor:
    """Swap adipose and fibroglandular inside tissue (calcification and air untouched)."""
    out = frac.clone()
    m = tissue.unsqueeze(1)
    swapped = frac[:, [1, 0, 2]]
    return torch.where(m, swapped, out)


def ref_quantiles(a_values: np.ndarray, q: int) -> np.ndarray:
    """q quantiles of a reference adipose-fraction sample at levels (i+0.5)/q (TRAIN labels)."""
    return np.quantile(np.asarray(a_values, dtype=np.float64), (np.arange(q) + 0.5) / q)


def w1_marginal(frac: torch.Tensor, tissue: torch.Tensor, ref_q: np.ndarray) -> torch.Tensor:
    """1-D Wasserstein-1 between the tissue adipose-fraction distribution and a reference."""
    a = _a(frac)[tissue]
    q = len(ref_q)
    levels = (torch.arange(q, dtype=a.dtype, device=a.device) + 0.5) / q
    pred_q = torch.quantile(a, levels) if a.numel() < 16_000_000 else _big_quantile(a, levels)
    ref = torch.as_tensor(ref_q, dtype=a.dtype, device=a.device)
    return (pred_q - ref).abs().mean()


def _big_quantile(a: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    s, _ = torch.sort(a)
    idx = (levels * (s.numel() - 1)).round().long()
    return s[idx]


def argmin_t(t: np.ndarray, mse: np.ndarray, r: np.ndarray, lam: float) -> float:
    """t at the minimum of mse(t) + lam*r(t) on the grid."""
    return float(t[int(np.argmin(mse + lam * r))])


def lambda_window(
    t_n: np.ndarray,
    r_n: np.ndarray,
    t_id: np.ndarray,
    mse_id: np.ndarray,
    r_id: np.ndarray,
    lams: np.ndarray,
    res: float,
    dt: float = 0.1,
    tol: float = 0.02,
) -> tuple[float, float]:
    """(lam_min, lam_max) of the usable weight window.

    lam_min: smallest grid lambda whose worst-side prior margin along the null direction at
    |t|=dt reaches the optimiser residual `res` (MSE units): lam*(R(+-dt)-R(0)) >= res; inf if
    R offers none.
    lam_max: largest grid lambda keeping the argmin along an IDENTIFIABLE direction within `tol`.
    Empty window <=> not lam_min <= lam_max.
    """
    i0 = int(np.argmin(np.abs(t_n)))
    ip, im = int(np.argmin(np.abs(t_n - dt))), int(np.argmin(np.abs(t_n + dt)))
    dr = min(r_n[ip] - r_n[i0], r_n[im] - r_n[i0])
    ok_min = [lam for lam in lams if dr > 0 and lam * dr >= res]
    lo = float(min(ok_min)) if ok_min else float("inf")
    ok_max = [lam for lam in lams if abs(argmin_t(t_id, mse_id, r_id, float(lam))) <= tol]
    hi = float(max(ok_max)) if ok_max else 0.0
    return lo, hi
