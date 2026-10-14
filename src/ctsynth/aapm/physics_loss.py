"""Projection-domain data term for end-to-end training: U-Net fractions -> fan-beam transmissions.

`ref` is the measured (official) transmission, shape (B, 2, V/2, n_det) with channel 0 = 50 kVp,
1 = 80 kVp (schema.KVPS order). `sub` selects the same view positions in `ref` as were simulated.
No ground-truth material enters this function.
"""

from __future__ import annotations

import torch

from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.schema import KVPS


def stack_kvp(t: dict[int, torch.Tensor]) -> torch.Tensor:
    """{50: (B,V,D), 80: (B,V,D)} -> (B, 2, V, D) in KVPS order."""
    return torch.stack([t[k] for k in KVPS], dim=1)


def projection_loss(
    fwd: AapmFanbeamForward,
    frac: torch.Tensor,
    ref: torch.Tensor,
    sub: torch.Tensor | None = None,
    *,
    log: bool = False,
    floor: float = 1e-8,
) -> torch.Tensor:
    """Mean over rays, kVp of (T_pred - T_ref)^2 [dimensionless, I0=1]; `log`: of the log-T gap."""
    pred = stack_kvp(fwd(frac, sub))
    tgt = ref if sub is None else ref[:, :, sub.to(ref.device)]
    if log:
        return ((pred.clamp_min(floor).log() - tgt.clamp_min(floor).log()) ** 2).mean()
    return ((pred - tgt) ** 2).mean()
