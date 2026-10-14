"""Differentiable torch version of the AAPM observation model (docs/aapm/unsupervised-spec.md).

Two modes, channel order 0=50 kVp (low), 1=80 kVp (high), materials in `schema.MATERIALS` order:

* IMAGE domain (`image_forward`): fractions (N,3,H,W) -> spectrum-weighted attenuation images
  (N,2,H,W), cm^-1, via A[k,m] = sum_E 0.5 s_k(E) mu_m(E). Exact for how the kit builds its images.
* PATH domain (`path_forward`): material LINE INTEGRALS P (N,3,...) in cm -> polychromatic
  transmission T (N,2,...), dimensionless, I0 = 1, noiseless:
  T_k = sum_E 0.5 s_k(E) exp(-sum_m mu_m(E) P_m). The physically correct model.

SCOPE: there is NO fan-beam projector in this repo (ctsynth.physics is parallel beam). The path
model therefore takes SUPPLIED line integrals; sinogram-domain AAPM experiments are impossible
until a fan-beam projector exists. Only image-domain experiments can be run today.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ctsynth.aapm.schema import KVPS, ModelTable


def _cl(x: torch.Tensor, hi: float | None) -> torch.Tensor:
    return x if hi is None else x.clamp(min=-hi)


class AapmTorchForward(nn.Module):
    """Holds the tables as buffers; float64-capable and autograd-clean."""

    def __init__(self, tables: dict[int, ModelTable], dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        for i, k in enumerate(KVPS):
            t = tables[k]
            self.register_buffer(f"w{i}", torch.as_tensor(0.5 * t.spectrum, dtype=dtype))
            self.register_buffer(f"mu{i}", torch.as_tensor(np.asarray(t.mu_per_cm), dtype=dtype))
        a = torch.stack([getattr(self, f"w{i}") @ getattr(self, f"mu{i}").T for i in range(2)])
        self.register_buffer("A", a)  # (2, 3) effective mu, cm^-1

    def image_forward(self, frac: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) fractions -> (N,2,H,W) attenuation images [cm^-1]."""
        return torch.einsum("km,nm...->nk...", self.A.to(frac.dtype), frac)

    def path_forward(
        self,
        paths: torch.Tensor,
        channels: tuple[int, ...] = (0, 1),
        exp_clip: float | None = None,
    ) -> torch.Tensor:
        """(N,3,...) line integrals [cm] -> (N,len(channels),...) transmission in (0,1].

        `exp_clip`: clamp mu.P at >= -exp_clip so exp(-mu.P) cannot overflow (a numerical guard
        only for outputs that can be negative; zero gradient beyond it). None = exact (default).
        """
        n = paths.shape[0]
        flat = paths.reshape(n, 3, -1)
        out = []
        for i in channels:
            mu, w = getattr(self, f"mu{i}").to(paths.dtype), getattr(self, f"w{i}").to(paths.dtype)
            out.append(
                torch.einsum(
                    "e,ner->nr", w, torch.exp(-_cl(torch.einsum("me,nmr->ner", mu, flat), exp_clip))
                )
            )
        return torch.stack(out, dim=1).reshape(n, len(out), *paths.shape[2:])
