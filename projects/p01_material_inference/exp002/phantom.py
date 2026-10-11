# ruff: noqa: E501
"""EXP-002 phantoms P2/P3 (partial mass densities, g/cm^3; idealised additive mixture)."""

from __future__ import annotations

import numpy as np
import torch

from ctsynth.physics import ParallelBeamProjector2D

N_PIX, PIXEL_CM, N_VIEW, N_DET, DET_SPACING_CM = 48, 0.5, 90, 72, 0.4
WATER_RADIUS_CM, INSERT_RADIUS_CM, RING_RADIUS_CM = 9.0, 1.25, 5.0


def make_projector() -> ParallelBeamProjector2D:
    angles = torch.arange(N_VIEW, dtype=torch.float64) * (np.pi / N_VIEW)
    # det_offset avoids rays exactly on pixel edges (unspecified in the spec)
    return ParallelBeamProjector2D(N_PIX, N_PIX, PIXEL_CM, angles, N_DET, DET_SPACING_CM, 0.0137)


def insert_layout(phantom: str) -> list[dict[str, float]]:
    """Insert list: material (1=iodine, 2=calcium), density, centre x/y (cm)."""
    spec = [(1, 0.005), (1, 0.010), (1, 0.020)] + ([(2, 0.1), (2, 0.3)] if phantom == "P3" else [])
    out = []
    for i, (m, d) in enumerate(spec):
        th = 2 * np.pi * (i + 0.37) / len(spec)
        out.append(
            {
                "material": m,
                "density": d,
                "x": RING_RADIUS_CM * np.cos(th),
                "y": RING_RADIUS_CM * np.sin(th),
            }
        )
    return out


def coords() -> tuple[np.ndarray, np.ndarray]:
    c = (np.arange(N_PIX) - (N_PIX - 1) / 2) * PIXEL_CM
    return np.meshgrid(c, c)  # x varies along columns, y along rows


def build(phantom: str) -> tuple[torch.Tensor, list[dict[str, float]], torch.Tensor]:
    """Returns rho (3,48,48) [water,iodine,calcium] g/cm^3, insert list, insert masks (n_ins,48,48)."""
    X, Y = coords()
    rho = np.zeros((3, N_PIX, N_PIX))
    rho[0][X**2 + Y**2 <= WATER_RADIUS_CM**2] = 1.0
    ins = insert_layout(phantom)
    masks = np.zeros((len(ins), N_PIX, N_PIX), dtype=bool)
    for j, it in enumerate(ins):
        masks[j] = (X - it["x"]) ** 2 + (Y - it["y"]) ** 2 <= INSERT_RADIUS_CM**2
        rho[int(it["material"])][masks[j]] = it["density"]
    return torch.from_numpy(rho), ins, torch.from_numpy(masks)
