"""Separately weighted loss terms for label-free decomposition (docs/aapm/unsupervised-spec.md).

Every term is a standalone callable returning a scalar tensor, UNWEIGHTED; weights live only in
`compose`, which skips weight-0 terms entirely so ablation removes their gradient exactly.
Nothing here hard-wires a combination. `frac` is (N,3,H,W) in `schema.MATERIALS` order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from ctsynth.aapm.torch_forward import AapmTorchForward


def consistency_image(
    frac: torch.Tensor, target: torch.Tensor, fwd: AapmTorchForward
) -> torch.Tensor:
    """Image-domain data term. mean over batch, 2 kVp channels and pixels of (A frac - target)^2.

    Divisor: N*2*H*W (per pixel per channel). Units: (cm^-1)^2. `target` is (N,2,H,W).
    """
    return ((fwd.image_forward(frac) - target) ** 2).mean()


def consistency_transmission(
    paths: torch.Tensor, target: torch.Tensor, fwd: AapmTorchForward
) -> torch.Tensor:
    """Transmission-domain data term. mean over all N*2*R rays of (T(paths) - target)^2.

    Divisor: per ray per kVp (R = prod of trailing dims). Units: dimensionless (I0 = 1).
    `paths` are line integrals in cm, (N,3,...); `target` is (N,2,...).
    """
    return ((fwd.path_forward(paths) - target) ** 2).mean()


def nonneg(frac: torch.Tensor) -> torch.Tensor:
    """Linear hinge mean(relu(-frac)) over all N*3*H*W elements; zero iff frac >= 0."""
    return torch.relu(-frac).mean()


def sum_to_one(frac: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """mean over masked pixels of (sum_m frac - 1)^2. mask (N,H,W) bool
    (True = apply), e.g. tissue only.

    Divisor: number of masked pixels (min 1). FALSE for air/band pixels; mask them out.
    """
    err = (frac.sum(dim=1) - 1.0) ** 2
    if mask is None:
        return err.mean()
    m = mask.to(err.dtype)
    return (err * m).sum() / m.sum().clamp_min(1.0)


def tv(frac: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Isotropic TV: mean over N*3*(H-1)*(W-1) of sqrt(dx^2+dy^2+eps) - sqrt(eps) (fwd differences).

    The sqrt(eps) offset makes a constant image score exactly 0 while keeping the gradient finite.
    """
    dx = frac[..., :-1, 1:] - frac[..., :-1, :-1]
    dy = frac[..., 1:, :-1] - frac[..., :-1, :-1]
    return (torch.sqrt(dx**2 + dy**2 + eps) - eps**0.5).mean()  # type: ignore[no-any-return]


def entropy(frac: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean per-pixel Shannon entropy (nats) of p = relu(frac)/sum(relu(frac)).

    Range [0, ln 3]; minimal at one-hot, maximal at uniform. All-zero (air) pixels give 0.
    Minimising it pushes toward pure materials: a TESTABLE HYPOTHESIS, not a known-good prior
    (it can equally reward confident wrong answers).
    """
    r = torch.relu(frac)
    p = r / (r.sum(dim=1, keepdim=True) + eps)
    return -(p * torch.log(p + eps)).sum(dim=1).mean()


REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "consistency_image": consistency_image,
    "consistency_transmission": consistency_transmission,
    "nonneg": nonneg,
    "sum_to_one": sum_to_one,
    "tv": tv,
    "entropy": entropy,
}


def compose(
    weights: Mapping[str, float],
    args: Mapping[str, tuple[Any, ...]],
    kwargs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """sum_t weights[t] * REGISTRY[t](*args[t], **kwargs[t]); weight-0 terms are not evaluated.

    Returns (total, {term: unweighted value}).
    """
    total = torch.zeros(())
    parts: dict[str, float] = {}
    for name, w in weights.items():
        if w == 0:
            continue
        v = REGISTRY[name](*args[name], **(kwargs or {}).get(name, {}))
        parts[name] = float(v.detach())
        total = total + w * v
    return total, parts
