"""Physical phantom representation: per-voxel mass concentration c_k (g/cm^3) of each distinct
elemental composition k of the XCAT COPD material tables.

mu(r, E) = sum_k c_k(r) kappa_k(E), kappa_k(E) = sum_z w_kz kappa_z(E) (cm^2/g, normalised mass
fractions), so mu is linear in c and a box average of c reproduces the simulator's box-averaged mu
exactly (partial volumes mix by occupancy x density, never by label). Organ IDs are not materials:
the 53 COPD IDs share 8 distinct compositions and differ otherwise only by a patient-specific
density (checked over all 67 tables by `build_basis`). Total density is sum_k c_k.
Units: c in g/cm^3 (stored as uint16 with step CONC_STEP), kappa in cm^2/g, mu in mm^-1 for LEAP.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ctsynth.xcat.materials import ElementTable, Material
from ctsynth.xcat.phantom import Grid, box_weights, nn_index

BASIS_PATH = Path(__file__).resolve().parents[3] / "data" / "tables" / "xcat_copd_basis_v1.json"
CONC_STEP = 1.0 / 8192.0  # g/cm^3; max stored 8.0, max XCAT density 1.4 -> rounding error <= 6.1e-5


def _key(m: Material, decimals: int = 6) -> tuple[tuple[int, float], ...]:
    sw = sum(m.mass_fractions.values())
    return tuple(sorted((z, round(w / sw, decimals)) for z, w in m.mass_fractions.items() if w > 0))


@dataclass(frozen=True)
class CompositionBasis:
    names: tuple[str, ...]  # first xlsx organ name that uses the composition (not an anatomy claim)
    organ_ids: tuple[tuple[int, ...], ...]  # organ IDs per channel
    zs: tuple[int, ...]  # atomic numbers of the fraction columns
    fractions: NDArray[np.float64]  # (K, len(zs)) normalised mass fractions, rows sum to 1
    organ_to_channel: NDArray[np.int16]  # (256,), -1 for organ IDs not in the tables

    @property
    def k(self) -> int:
        return len(self.names)

    def kappa_cm2g(self, e_kev: float, elements: ElementTable | None = None) -> NDArray[np.float64]:
        """(K,) mass attenuation of each composition at one energy."""
        et = elements or ElementTable()
        kz = np.array([et.kappa(z, e_kev) for z in self.zs])
        return np.asarray(self.fractions @ kz, dtype=np.float64)

    def to_json(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "organ_ids": [list(o) for o in self.organ_ids],
            "zs": list(self.zs),
            "fractions": self.fractions.tolist(),
            "organ_to_channel": self.organ_to_channel.tolist(),
        }

    @staticmethod
    def from_json(d: dict[str, object]) -> CompositionBasis:
        return CompositionBasis(
            tuple(d["names"]),  # type: ignore[arg-type]
            tuple(tuple(o) for o in d["organ_ids"]),  # type: ignore[attr-defined]
            tuple(d["zs"]),  # type: ignore[arg-type]
            np.asarray(d["fractions"], dtype=np.float64),
            np.asarray(d["organ_to_channel"], dtype=np.int16),
        )

    def check(self, materials: dict[int, Material]) -> None:
        """Raise unless each organ ID of a patient table has its channel's composition."""
        for oid, m in materials.items():
            ch = int(self.organ_to_channel[oid])
            if ch < 0:
                raise ValueError(f"organ {oid} ({m.name}) is not in the basis")
            row = np.zeros(len(self.zs))
            for z, w in m.mass_fractions.items():
                if w > 0:
                    row[self.zs.index(z)] = w
            if not np.allclose(row / row.sum(), self.fractions[ch], atol=1e-5):
                raise ValueError(f"organ {oid} ({m.name}) composition differs from channel {ch}")


def build_basis(tables: Sequence[dict[int, Material]]) -> CompositionBasis:
    """Distinct normalised compositions over all patient tables; a table that disagrees on the
    composition of any organ ID raises (compositions are global, only densities are patient
    specific)."""
    first = tables[0]
    key_of = {oid: _key(m) for oid, m in first.items()}
    for t in tables[1:]:
        if t.keys() != first.keys() or any(_key(m) != key_of[oid] for oid, m in t.items()):
            raise ValueError("composition of an organ ID differs between patients")
    keys = sorted(set(key_of.values()), key=lambda k: min(o for o, kk in key_of.items() if kk == k))
    zs = tuple(sorted({z for k in keys for z, _ in k}))
    frac = np.zeros((len(keys), len(zs)))
    o2c = np.full(256, -1, np.int16)
    names, ids = [], []
    for c, k in enumerate(keys):
        for z, w in k:
            frac[c, zs.index(z)] = w
        members = sorted(o for o, kk in key_of.items() if kk == k)
        for o in members:
            o2c[o] = c
        names.append(first[members[0]].name)
        ids.append(tuple(members))
    frac /= frac.sum(1, keepdims=True)
    return CompositionBasis(tuple(names), tuple(ids), zs, frac, o2c)


def load_basis(path: str | Path = BASIS_PATH) -> CompositionBasis:
    return CompositionBasis.from_json(json.loads(Path(path).read_text()))


def density_lut(materials: dict[int, Material]) -> NDArray[np.float32]:
    """(256,) patient-specific density (g/cm^3) per organ ID, 0 where unmapped."""
    rho = np.zeros(256, np.float32)
    for oid, m in materials.items():
        rho[oid] = m.density
    return rho


def resample_concentrations(
    vol: NDArray[np.uint8],
    grid: Grid,
    basis: CompositionBasis,
    rho_lut: NDArray[np.float32],
    device: torch.device | str,
    chunk: int = 48,
) -> torch.Tensor:
    """(K, nz, ny, nx) float32 g/cm^3, exact box average of (channel one-hot x density) onto the
    simulation grid, with the same weights and padded-border air fill as `resample_phantom`."""
    nzn, nyn, nxn = grid.native_shape
    wz, wy, wx = (
        box_weights(n, grid.vox_mm, m, grid.native_vox_mm, device)
        for n, m in zip(grid.shape, grid.native_shape, strict=True)
    )
    lut_ch = torch.from_numpy(basis.organ_to_channel.astype(np.int64)).to(device)
    lut_rho = torch.from_numpy(rho_lut).to(device)
    out = torch.zeros((basis.k, *grid.shape), device=device)
    for z0 in range(0, nzn, chunk):
        z1 = min(z0 + chunk, nzn)
        li = torch.from_numpy(np.array(vol[z0:z1])).to(device).long()  # copy: memmaps are read-only
        ch, rho = lut_ch[li], lut_rho[li]
        if bool((ch < 0).any()):
            raise ValueError("phantom contains organ IDs that are not in the basis")
        for k in range(basis.k):
            a = torch.einsum("oy,zyx->zox", wy, torch.where(ch == k, rho, torch.zeros_like(rho)))
            a = torch.einsum("px,zox->zop", wx, a)
            out[k] += torch.einsum("kz,zop->kop", wz[:, z0:z1], a)
        del li, ch, rho
    cov = wz.sum(1)[:, None, None] * wy.sum(1)[None, :, None] * wx.sum(1)[None, None, :]
    out[int(basis.organ_to_channel[0])] += (1 - cov) * float(rho_lut[0])  # padded border = air
    return out


def dominant_labels(
    vol: NDArray[np.uint8], grid: Grid, basis: CompositionBasis, device: torch.device | str
) -> NDArray[np.int16]:
    """Channel index of the native voxel under each simulation-voxel centre (nearest neighbour; -1
    outside the phantom). Diagnostic/visualisation only: concentrations, not these, are the
    target."""
    iz, okz = nn_index(grid.shape[0], grid.vox_mm, grid.native_shape[0], grid.native_vox_mm)
    iy, oky = nn_index(grid.shape[1], grid.vox_mm, grid.native_shape[1], grid.native_vox_mm)
    ix, okx = nn_index(grid.shape[2], grid.vox_mm, grid.native_shape[2], grid.native_vox_mm)
    out = np.full(grid.shape, -1, np.int16)
    for k, z in enumerate(iz):
        if okz[k]:
            plane = basis.organ_to_channel[np.asarray(vol[z])[iy][:, ix]]
            out[k] = np.where(np.outer(oky, okx), plane, -1)
    return out


def mu_mm_from_concentration(
    c: torch.Tensor, kappa_cm2g: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """mu (mm^-1) = sum_k c_k kappa_k / 10 along channel axis `dim` (c in g/cm^3, kappa cm^2/g)."""
    k = kappa_cm2g.to(c)
    mu = torch.zeros_like(c.select(dim, 0))
    for i in range(c.shape[dim]):  # per-channel accumulation: no (K, ...) temporary
        mu += c.select(dim, i) * k[i]
    return mu / 10.0


def conc_to_u16(c: NDArray[np.float32]) -> NDArray[np.uint16]:
    return np.round(np.clip(c, 0.0, 65535 * CONC_STEP) / CONC_STEP).astype(np.uint16)


def u16_to_conc(a: NDArray[np.uint16]) -> NDArray[np.float32]:
    return (a.astype(np.float32) * np.float32(CONC_STEP)).astype(np.float32)
