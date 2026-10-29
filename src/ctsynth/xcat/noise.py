"""Projection-domain photon noise for clean line integrals.

Model: counts n ~ Poisson(N0 exp(-p)); measurement p_hat = -ln(max(n, floor) / N0). N0 is an
incident-photon scale per detector reading
(per ray, per view); it is NOT a calibrated clinical mAs. Zero counts are clamped to `floor`
(default 1 count), so the largest measurable
line integral is ln(N0 / floor); this biases only rays with N0 exp(-p) of order one. No electronic
noise, scatter, bowtie or beam hardening.
First-order theory: Var[p_hat] ~ exp(p) / N0 (delta method) and E[p_hat] - p ~ exp(p) / (2 N0).
"""

from __future__ import annotations

import torch

from ctsynth.seeding import rng


def make_generator(device: torch.device | str, seed: int, *stream: int) -> torch.Generator:
    """Deterministic torch generator derived from ctsynth.seeding.rng(seed, *stream)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(rng(seed, *stream).integers(0, 2**62)))
    return g


def apply_poisson_noise(
    p: torch.Tensor,
    n0: float,
    generator: torch.Generator,
    floor: float = 1.0,
    out: torch.Tensor | None = None,
    view_chunk: int = 500,
) -> torch.Tensor:
    """Noisy line integrals for clean `p` (views, rows, channels); processed in view chunks to
    bound temporary memory.
    `out=p` is allowed (in place)."""
    if n0 <= 0 or floor <= 0:
        raise ValueError("n0 and floor must be positive")
    res = out if out is not None else torch.empty_like(p)
    for v0 in range(0, p.shape[0], view_chunk):
        lam = (n0 * torch.exp(-p[v0 : v0 + view_chunk])).to(torch.float32)
        counts = torch.poisson(lam, generator=generator).clamp_(min=floor)
        res[v0 : v0 + view_chunk] = -torch.log(counts / n0)
    return res
