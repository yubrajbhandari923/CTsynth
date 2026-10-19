"""Per-case solver helpers for EXP-036 (torch only, device-agnostic, any float dtype).

- `to_fractions`: map a parameter tensor of arm A/B/C to material fractions (N,3,H,W) in
  (adipose, fibroglandular, calcification) order, zero outside the tissue mask.
- `minimize`: chunked L-BFGS with a trace and abs/plateau/max-iteration stop rules.
- `error_bands`: masked RMS of an error map split into low/mid/high radial spatial-frequency bands
  (Parseval: the squared band values sum to the masked mean square).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from ctsynth.aapm.schema import ModelTable


def drop_zero_weight(table: ModelTable) -> ModelTable:
    """Remove energies with spectrum == 0: they add exactly 0 to the model, but exp(+mu.P) of a
    negative path can overflow there (mu ~1e3 cm^-1 at 1-3 keV) and 0*inf = NaN."""
    k = table.spectrum > 0
    return ModelTable(table.kvp, table.energies_kev[k], table.spectrum[k], table.mu_per_cm[:, k])


def to_fractions(x: torch.Tensor, mask: torch.Tensor, arm: str) -> torch.Tensor:
    """Material fractions (N,3,H,W) from parameters x (N,k,H,W); zero outside `mask` (N,H,W)."""
    expected_k = {"A": 1, "B": 2, "C": 3}
    if arm not in expected_k:
        raise ValueError(f"unknown arm {arm!r}; expected one of 'A', 'B', 'C'")
    if x.shape[1] != expected_k[arm]:
        raise ValueError(f"arm {arm} needs k={expected_k[arm]} channels, got {x.shape[1]}")
    if arm == "A":
        a = x[:, 0]
        frac = torch.stack([a, 1 - a, torch.zeros_like(a)], dim=1)
    elif arm == "B":
        a, c = x[:, 0], x[:, 1]
        frac = torch.stack([a, 1 - a - c, c], dim=1)
    else:
        frac = x
    return frac * mask.unsqueeze(1).to(frac.dtype)


def minimize(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    max_iter: int = 1500,
    chunk: int = 25,
    rel_tol: float = 1e-3,
    patience: int = 100,
    abs_tol: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Minimise scalar `loss_fn` from a detached copy of x0 with chunked strong-Wolfe L-BFGS.

    Stops after a chunk with reason "abs_tol" (loss <= abs_tol), "plateau" (relative improvement
    over the last `patience` iterations < rel_tol) or "max_iter". Raises RuntimeError on a
    non-finite loss.
    """
    x = x0.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [x],
        lr=1,
        max_iter=chunk,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=0,
        tolerance_change=0,
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = loss_fn(x)
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    def current_loss() -> float:
        with torch.no_grad():
            value = float(loss_fn(x).detach())
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite loss {value} in minimize")
        return value

    losses = [current_loss()]
    trace: list[tuple[int, float]] = [(0, losses[0])]
    lookback = math.ceil(patience / chunk)
    n_iter = 0
    reason = "max_iter"
    while True:
        opt.step(closure)  # type: ignore[no-untyped-call]
        n_iter += chunk
        losses.append(current_loss())
        trace.append((n_iter, losses[-1]))
        if losses[-1] <= abs_tol:
            reason = "abs_tol"
            break
        if n_iter >= patience:
            past = losses[len(losses) - 1 - lookback]
            if (past - losses[-1]) / past < rel_tol:
                reason = "plateau"
                break
        if n_iter >= max_iter:
            reason = "max_iter"
            break
    return x.detach(), {"trace": trace, "reason": reason, "n_iter": n_iter}


def error_bands(
    err: torch.Tensor, mask: torch.Tensor, edges: tuple[float, float] = (1 / 32, 1 / 8)
) -> dict[str, float]:
    """Masked RMS of `err` (N,H,W) in radial spatial-frequency bands (cycles/pixel) low/mid/high."""
    e = err * mask.to(err.dtype)
    h, w = e.shape[-2], e.shape[-1]
    fy = torch.fft.fftfreq(h, device=e.device, dtype=e.dtype)
    fx = torch.fft.fftfreq(w, device=e.device, dtype=e.dtype)
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bands = {
        "low": r < edges[0],
        "mid": (r >= edges[0]) & (r < edges[1]),
        "high": r >= edges[1],
    }
    spec = torch.fft.fft2(e)
    n_pix = mask.sum().clamp_min(1).to(e.dtype)
    out: dict[str, float] = {}
    for name, band in bands.items():
        e_b = torch.fft.ifft2(spec * band.to(e.dtype)).real
        out[name] = float(torch.sqrt((e_b**2).sum() / n_pix))
    return out
