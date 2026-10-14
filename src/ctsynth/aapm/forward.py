"""Materials -> kVp-switched polychromatic transmissions (fan-beam projector + path_forward).

Fractions (B, 3, n_pix, n_pix) in canonical `schema.MATERIALS` order -> {50: (B, V/2, n_det),
80: (B, V/2, n_det)} air-normalised transmission in (0, 1]. View parity per F-032: 80 kVp = even
views, 50 kVp = odd views. Same observation model as the kit generator (dualEnergyTransmission.py).
"""

from __future__ import annotations

import torch
from torch import nn

from ctsynth.aapm.fanbeam import fanbeam_project
from ctsynth.aapm.schema import KVP_HIGH, KVP_LOW, REAL_GEOMETRY, GeometrySpec, ModelTable
from ctsynth.aapm.torch_forward import AapmTorchForward


class AapmFanbeamForward(nn.Module):
    def __init__(
        self,
        tables: dict[int, ModelTable],
        geometry: GeometrySpec = REAL_GEOMETRY,
        dtype: torch.dtype = torch.float32,
        view_chunk: int = 32,
    ) -> None:
        super().__init__()
        self.geometry, self.view_chunk = geometry, view_chunk
        self.spectral = AapmTorchForward(tables, dtype=dtype)

    def line_integrals(
        self, frac: torch.Tensor, kvp: int, sub: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(B, 3, n, n) -> (B, 3, V/2, n_det) path lengths [cm], one kVp's views."""
        b, m, n, _ = frac.shape
        start = 0 if kvp == KVP_HIGH else 1
        views = torch.arange(start, self.geometry.n_views, 2, device=frac.device)
        if sub is not None:  # positions within this kVp's view list
            views = views[sub.to(frac.device)]
        return fanbeam_project(frac.reshape(b * m, n, n), self.geometry, views).reshape(
            b, m, views.numel(), -1
        )

    def forward(
        self, frac: torch.Tensor, sub: torch.Tensor | None = None
    ) -> dict[int, torch.Tensor]:
        """`sub`: optional view positions (0..V/2-1), both kVp; output order follows it."""
        out = {}
        for kvp, ch in ((KVP_LOW, 0), (KVP_HIGH, 1)):
            parts = []
            p = self.line_integrals(frac, kvp, sub)
            for lo in range(0, p.shape[2], self.view_chunk):  # bounds the (E x rays) exp buffer
                parts.append(
                    self.spectral.path_forward(p[:, :, lo : lo + self.view_chunk], (ch,))[:, 0]
                )
            out[kvp] = torch.cat(parts, dim=1)
        return out
