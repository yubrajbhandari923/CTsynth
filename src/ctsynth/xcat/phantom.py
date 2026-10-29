"""Voxelised XCAT phantoms: memory-mapped reading, simulation grid, physically correct resampling.

Phantom files are C-order uint8 [z, y, x] (file name lists (nx, ny, nz)). mu and density are BOX-
AVERAGED onto the simulation grid
(exact overlap weights for any size ratio, so total attenuation is conserved); categorical labels
use nearest neighbour at the voxel
centre; voxels outside the phantom extent are air. Both grids share one centre, which is the world
origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ctsynth.xcat.materials import MaterialModel

NATIVE_VOX_MM = 0.25  # documented for the reference phantom only; ASSUMED for COPD / lung / liver
# array axes (z, y, x) increase toward Superior, Left, Posterior: PROVISIONAL (inferred, not
# metadata)
AXCODES = "SLP"


def open_phantom(path: str | Path, shape_zyx: tuple[int, int, int]) -> NDArray[np.uint8]:
    """Memory-map a phantom, checking that the file size equals the product of the dimensions
    (uint8)."""
    size = Path(path).stat().st_size
    if size != int(np.prod(shape_zyx)):
        raise ValueError(f"{path}: {size} bytes != prod{shape_zyx} for uint8")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=shape_zyx)


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int, int]  # (nz, ny, nx)
    vox_mm: float
    native_shape: tuple[int, int, int]
    native_vox_mm: float = NATIVE_VOX_MM

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return (
            self.shape[0] * self.vox_mm,
            self.shape[1] * self.vox_mm,
            self.shape[2] * self.vox_mm,
        )

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))


def make_grid(
    native_shape: tuple[int, int, int],
    vox_mm: float,
    native_vox_mm: float = NATIVE_VOX_MM,
    pad_vox: int = 2,
) -> Grid:
    n = tuple(int(math.ceil(s * native_vox_mm / vox_mm - 1e-9)) + pad_vox for s in native_shape)
    return Grid((n[0], n[1], n[2]), float(vox_mm), native_shape, float(native_vox_mm))


def box_weights(
    n_out: int, vox_out: float, n_in: int, vox_in: float, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """(n_out, n_in): overlap length of output voxel j with input voxel i divided by vox_out (grids
    share a centre).
    Row sums are the covered fraction of each output voxel: 1 inside the phantom, < 1 on the padded
    border."""
    off = (n_out * vox_out - n_in * vox_in) / 2.0
    lo_o = torch.arange(n_out, dtype=torch.float64) * vox_out - off
    lo_i = torch.arange(n_in, dtype=torch.float64) * vox_in
    ov = (
        torch.minimum(lo_o[:, None] + vox_out, lo_i[None] + vox_in)
        - torch.maximum(lo_o[:, None], lo_i[None])
    ).clamp(min=0) / vox_out
    return ov.float().to(device)


def nn_index(
    n_out: int, vox_out: float, n_in: int, vox_in: float
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Native index under the centre of each output voxel, and whether it lies inside the native
    extent."""
    off = (n_out * vox_out - n_in * vox_in) / 2.0
    c = (np.arange(n_out) + 0.5) * vox_out - off
    i = np.floor(c / vox_in).astype(np.int64)
    return np.clip(i, 0, n_in - 1), (i >= 0) & (i < n_in)


@dataclass
class Resampled:
    labels: torch.Tensor  # uint8, nearest neighbour, original organ IDs
    groups: torch.Tensor  # uint8, organ-group IDs (0 = air)
    density: torch.Tensor  # g/cm^3, box average
    mu_mm: torch.Tensor  # mm^-1, box average
    histogram: NDArray[np.int64]  # native label counts (256,)


def resample_phantom(
    vol: NDArray[np.uint8],
    grid: Grid,
    model: MaterialModel,
    device: torch.device | str,
    chunk: int = 48,
) -> Resampled:
    nz, ny, nx = grid.shape
    nzn, nyn, nxn = grid.native_shape
    wz, wy, wx = (
        box_weights(n, grid.vox_mm, m, grid.native_vox_mm, device)
        for n, m in zip(grid.shape, grid.native_shape, strict=True)
    )
    iz, okz = nn_index(nz, grid.vox_mm, nzn, grid.native_vox_mm)
    iy, oky = nn_index(ny, grid.vox_mm, nyn, grid.native_vox_mm)
    ix, okx = nn_index(nx, grid.vox_mm, nxn, grid.native_vox_mm)
    lut_mu = torch.from_numpy(model.mu_mm).to(device)
    lut_rho = torch.from_numpy(model.rho.astype(np.float32)).to(device)
    lut_grp = torch.from_numpy(model.organ_to_group).to(device)
    mu = torch.zeros(grid.shape, device=device)
    rho = torch.zeros(grid.shape, device=device)
    lab = torch.zeros(grid.shape, dtype=torch.uint8, device=device)
    hist = torch.zeros(256, dtype=torch.int64, device=device)
    iy_t, ix_t = torch.from_numpy(iy).to(device), torch.from_numpy(ix).to(device)
    mask_yx = torch.from_numpy(np.outer(oky, okx)).to(device)
    for z0 in range(0, nzn, chunk):
        z1 = min(z0 + chunk, nzn)
        raw = torch.from_numpy(np.array(vol[z0:z1])).to(device)  # copy: memmaps are read-only
        hist += torch.bincount(raw.flatten().long(), minlength=256)
        li = raw.long()
        for lut, dst in ((lut_mu, mu), (lut_rho, rho)):
            a = torch.einsum("oy,zyx->zox", wy, lut[li])
            a = torch.einsum("px,zox->zop", wx, a)
            dst += torch.einsum("kz,zop->kop", wz[:, z0:z1], a)
        ks = np.flatnonzero((iz >= z0) & (iz < z1) & okz)
        if ks.size:
            sl = raw[torch.from_numpy(iz[ks] - z0).to(device)][:, iy_t][:, :, ix_t]
            lab[torch.from_numpy(ks).to(device)] = torch.where(
                mask_yx[None], sl, torch.zeros_like(sl)
            )
        del raw, li
    cov = wz.sum(1)[:, None, None] * wy.sum(1)[None, :, None] * wx.sum(1)[None, None, :]
    mu += (1 - cov) * float(model.mu_mm[0])  # label 0 = air fills the padded border
    rho += (1 - cov) * float(model.rho[0])
    return Resampled(lab, lut_grp[lab.long()], rho, mu, hist.cpu().numpy())


def mu_to_hu(mu_mm: torch.Tensor, mu_water_mm: float) -> torch.Tensor:
    """HU = 1000 (mu / mu_water - 1)."""
    return 1000.0 * (mu_mm / mu_water_mm - 1.0)


def affine_from_axcodes(
    shape: tuple[int, int, int], vox_mm: float, axcodes: str = AXCODES
) -> NDArray[np.float64]:
    """Voxel -> world (RAS+ mm) affine; `axcodes` name the direction each array index increases
    toward; origin-centred."""
    w = {"R": (0, 1), "L": (0, -1), "A": (1, 1), "P": (1, -1), "S": (2, 1), "I": (2, -1)}
    a = np.eye(4)
    a[:3, :3] = 0
    for i, c in enumerate(axcodes):
        row, sign = w[c]
        a[row, i] = sign * vox_mm
    a[:3, 3] = -a[:3, :3] @ ((np.array(shape) - 1) / 2.0)
    return a
