"""LEAP (LLNL, commit 0c8846f) helical cone-beam forward projection and FBP for the TrueCT/Duke1
geometry.

Geometry (PDF Table 1 / DICOM-CT-PD tags of the TrueCT challenge): SOD 575 mm, SDD 1050 mm, 900 x
64 curved detector, 1 mm pixels,
1000 views per rotation, 35.05 mm table feed per rotation at pitch 1.0, no flying focal spot. The
detector centre is the middle element;
the empirical real-data offsets of the TrueCT pilot are NOT applied (borrowed geometry, not the
phantoms' original DukeSim acquisition).
`FBP` on a helical geometry uses LEAP's derivative + Hilbert filtering and weighted backprojection
(not the axial ramp / FDK path).
Volume arrays are [z, y, x], centred on the isocentre / mid-scan. Projections are (view,
detector_row, detector_channel).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import torch

from ctsynth.xcat.phantom import Grid

SOD_MM, SDD_MM, PIXEL_MM, N_ROWS, N_COLS = 575.0, 1050.0, 1.0, 64, 900
VIEWS_PER_ROTATION, TABLE_MM_PER_ROTATION_PITCH1 = 1000, 35.05
LEAP_SRC = os.environ.get("LEAP_SRC", "/home/yb107/ctsynth_truect/leap/LEAP/src")


@dataclass(frozen=True)
class Acquisition:
    """Physical acquisition settings that change the projections (clean line integrals)."""

    energy_kev: float = 70.0
    pitch: float = 1.0  # table feed per rotation / 35.05 mm
    view_decimation: int = 1  # keep every k-th view: 1000 / k views per rotation
    margin_mm: float = 40.0  # extra scan length beyond the phantom on each side (helical coverage)

    @property
    def views_per_rotation(self) -> int:
        if VIEWS_PER_ROTATION % self.view_decimation:
            raise ValueError("view_decimation must divide 1000")
        return VIEWS_PER_ROTATION // self.view_decimation

    @property
    def table_mm_per_rotation(self) -> float:
        return TABLE_MM_PER_ROTATION_PITCH1 * self.pitch


@dataclass(frozen=True)
class ScanPlan:
    n_rotations: int
    n_views: int
    views_per_rotation: int
    scan_length_mm: float
    table_mm_per_rotation: float
    pitch_mm_per_rad: float
    fov_radius_mm: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def plan_scan(grid: Grid, acq: Acquisition) -> ScanPlan:
    ext_z = grid.shape[0] * grid.vox_mm
    n_rot = int(math.ceil((ext_z + 2 * acq.margin_mm) / acq.table_mm_per_rotation))
    beta = 0.5 * N_COLS * PIXEL_MM / SDD_MM
    return ScanPlan(
        n_rot,
        n_rot * acq.views_per_rotation,
        acq.views_per_rotation,
        n_rot * acq.table_mm_per_rotation,
        acq.table_mm_per_rotation,
        acq.table_mm_per_rotation / (2 * math.pi),
        SOD_MM * math.sin(beta),
    )


def projection_bytes(plan: ScanPlan) -> int:
    return plan.n_views * N_ROWS * N_COLS * 4


def make_leap(grid: Grid, plan: ScanPlan, device: int = 0, leap: Any = None) -> Any:
    """Configure a LEAP model; pass an existing `tomographicModels` (e.g. leaptorch's) to share its
    geometry."""
    import numpy as np

    if LEAP_SRC not in sys.path:
        sys.path.insert(0, LEAP_SRC)
    from leapctype import tomographicModels

    leap = leap if leap is not None else tomographicModels()
    phis = (360.0 * np.arange(plan.n_views) / plan.views_per_rotation).astype(np.float32)
    leap.set_conebeam(
        plan.n_views,
        N_ROWS,
        N_COLS,
        PIXEL_MM,
        PIXEL_MM,
        (N_ROWS - 1) / 2,
        (N_COLS - 1) / 2,
        phis,
        SOD_MM,
        SDD_MM,
    )
    leap.set_curvedDetector()
    leap.set_helicalPitch(plan.pitch_mm_per_rad)
    nz, ny, nx = grid.shape
    leap.set_volume(nx, ny, nz, grid.vox_mm, grid.vox_mm, 0.0, 0.0, 0.0)
    leap.set_gpu(device)
    return leap


def forward_project(leap: Any, mu_mm: torch.Tensor, plan: ScanPlan) -> torch.Tensor:
    """Clean line integrals p = P[mu] (mu in mm^-1, result dimensionless). No scatter, noise or
    spectrum."""
    g = torch.zeros((plan.n_views, N_ROWS, N_COLS), dtype=torch.float32, device=mu_mm.device)
    leap.project(g, mu_mm.contiguous())
    return g


def fbp(leap: Any, g: torch.Tensor, grid: Grid, inplace: bool = False) -> torch.Tensor:
    """LEAP helical FBP -> mu in mm^-1. `inplace=True` overwrites g (saves one projection-sized
    copy). Not differentiable here."""
    f = torch.zeros(grid.shape, dtype=torch.float32, device=g.device)
    leap.FBP(g, f, inplace=inplace)
    return f
