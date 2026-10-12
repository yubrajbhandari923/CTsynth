"""Synthetic schema-mirror generator.

!!! SYNTHETIC IS NOT AAPM. Data from this module exercise code paths and schema conventions only
and MUST NEVER be reported, plotted, or cited as AAPM results. !!!

The arrays have exactly the AAPM schema (see schema.py) at reduced size. Transmission is produced by
the documented AAPM forward model (docs/aapm/adapter-spec.md section 4) with exact pixel-chord line
integrals; images are NOT FBP reconstructions (see `SyntheticData.image_*`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ctsynth.aapm.schema import (
    KVPS,
    MATERIAL_FILE_NAMES,
    Array,
    GeometrySpec,
    ModelTable,
    view_indices,
)
from ctsynth.seeding import rng


@dataclass(frozen=True)
class SyntheticConfig:
    """Reduced-size generator settings. Defaults: 4 cases of 16x16 px, 16 views, 24 bins."""

    n_cases: int = 4
    n_pix: int = 16
    n_views: int = 16  # total, even
    n_det: int = 24  # even
    seed: int = 0
    fov_cm: float = 18.0

    def geometry(self) -> GeometrySpec:
        """GeometrySpec with this config's sizes and the real fov/R/SDD."""
        return GeometrySpec(
            n_pix=self.n_pix, fov_cm=self.fov_cm, n_views=self.n_views, n_det=self.n_det
        )


@dataclass(frozen=True)
class SyntheticData:
    """phantoms (N,3,n,n) f32 canonical order; transmission/images keyed by kVp {50, 80}.

    transmission[k]: (N, n_views//2, n_det) f32 from the forward model.
    images[k]: (N, n, n) f32 = sum_m phantoms[:, m] * sum_E(0.5*s_k(E)*mu_m(E))
    -- a spectrum-weighted mu map, NOT an FBP image.
    """

    phantoms: Array
    transmission: dict[int, Array]
    images: dict[int, Array]
    geometry: GeometrySpec


def _chord_lengths(g: GeometrySpec) -> Array:
    """(V, D, n, n) float64: exact cm length of each source->bin ray inside each pixel [ix, iy]."""
    rad, sdd, fov, n = g.source_radius_cm, g.sdd_cm, g.fov_cm, g.n_pix
    ang = g.view_angles()
    c, s = np.cos(ang), np.sin(ang)
    src = rad * np.stack([c, s], axis=-1)  # (V, 2)
    cen = (rad - sdd) * np.stack([c, s], axis=-1)
    e_u = np.stack([-s, c], axis=-1)
    half = g.detector_length_cm() / 2.0
    u = -half + (np.arange(g.n_det) + 0.5) * (2 * half) / g.n_det  # (D,)
    d_raw = cen[:, None, :] + u[None, :, None] * e_u[:, None, :] - src[:, None, :]  # (V, D, 2)
    norm = np.hypot(d_raw[..., 0], d_raw[..., 1])
    # shortcut: exact-zero direction component mapped to 1e-300 so the slab limits are +-inf-like
    d = np.where(d_raw == 0.0, 1e-300, d_raw)
    edges = -fov / 2.0 + (fov / n) * np.arange(n + 1)
    lo, hi = edges[None, None, :-1], edges[None, None, 1:]

    def slab(a: int) -> tuple[Array, Array]:
        t1 = (lo - src[:, a][:, None, None]) / d[:, :, a][:, :, None]  # (V, D, n)
        t2 = (hi - src[:, a][:, None, None]) / d[:, :, a][:, :, None]
        return np.minimum(t1, t2), np.maximum(t1, t2)

    x0, x1 = slab(0)
    y0, y1 = slab(1)
    tmin = np.maximum(x0[:, :, :, None], y0[:, :, None, :])
    tmax = np.minimum(x1[:, :, :, None], y1[:, :, None, :])
    return np.asarray(np.maximum(tmax - tmin, 0.0) * norm[:, :, None, None], dtype=np.float64)


def line_integrals(phantoms: Array, geometry: GeometrySpec) -> Array:
    """Path lengths P[case, material, view, bin] in cm, float64, FULL interlaced view axis.

    P = sum over pixels of fraction * (exact length in cm of the source->detector-bin ray inside
    that pixel square). Geometry per docs/aapm/adapter-spec.md section 3.
    """
    ch = _chord_lengths(geometry)
    return np.asarray(
        np.einsum("nmxy,vdxy->nmvd", phantoms.astype(np.float64), ch), dtype=np.float64
    )


def simulate_transmission(
    phantoms: Array, geometry: GeometrySpec, tables: dict[int, ModelTable]
) -> dict[int, Array]:
    """T = sum_E 0.5*s_kVp(E)*exp(-sum_m mu_m(E)*P_m), I0 = 1, noiseless, float32, per kVp.

    kVp 80 uses even views, 50 odd views (schema.view_indices).
    """
    paths = line_integrals(phantoms, geometry)
    out: dict[int, Array] = {}
    for kvp in KVPS:
        tab = tables[kvp]
        p = paths[:, :, view_indices(kvp, geometry.n_views), :]  # (N, 3, Vk, D)
        att = np.exp(-np.einsum("me,nmvd->nevd", tab.mu_per_cm, p))
        out[kvp] = np.einsum("e,nevd->nvd", 0.5 * tab.spectrum, att).astype(np.float32)
    return out


def _disc_mask(n_sub: int, cx: float, cy: float, ax: float, ay: float) -> Array:
    """Boolean (n_sub, n_sub) supersample mask of an axis-aligned ellipse on [-1, 1]^2."""
    xs = -1.0 + 2.0 * (np.arange(n_sub) + 0.5) / n_sub
    x, y = np.meshgrid(xs, xs, indexing="ij")  # axis 0 = x (ix), axis 1 = y (iy)
    return ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2 <= 1.0


def _coverage(mask: Array, n: int, k: int) -> Array:
    """Mean of the k x k supersamples inside each of the n x n pixels."""
    return mask.reshape(n, k, n, k).mean(axis=(1, 3))


def generate(config: SyntheticConfig, tables: dict[int, ModelTable]) -> SyntheticData:
    """Generate a self-consistent synthetic dataset (deterministic given config.seed).

    Per case: an off-centre asymmetric (semi-axes and centre differ in x and y) elliptical
    'breast' of tissue on air; an inner fibroglandular ellipse, remainder adipose;
    boundary pixels carry partial coverage c in (0,1) (supersampled),
    so tissue sum == c there; 1..3 single-pixel calcification
    inclusions (fraction 0.3..0.8) inside full tissue, taken from the local tissue so the sum is
    conserved. Guarantees air (sum==0), tissue (sum==1) and band pixels all exist.
    """
    n, k = config.n_pix, 8  # 8x8 supersampling: partial coverages stay >= 1/64 from 0 and 1
    phantoms = np.zeros((config.n_cases, 3, n, n), dtype=np.float32)
    for i in range(config.n_cases):
        g = rng(config.seed, i)
        cx, cy = g.uniform(-0.08, 0.08, size=2)
        ax, ay = g.uniform(0.82, 0.93), g.uniform(0.80, 0.92)
        outer = _disc_mask(n * k, cx, cy, ax, ay)
        ix_, iy_ = cx + g.uniform(-0.1, 0.1), cy + g.uniform(-0.1, 0.1)
        inner = _disc_mask(
            n * k, ix_, iy_, ax * g.uniform(0.45, 0.7), ay * g.uniform(0.45, 0.7)
        )  # sample-level mask, so fibroglandular is always inside the breast
        c_out = _coverage(outer, n, k)
        fg = _coverage(outer & inner, n, k)
        adi = c_out - fg
        calc = np.zeros((n, n))
        cand = np.argwhere(c_out == 1.0)  # full-tissue pixels only
        for ix, iy in cand[g.choice(len(cand), size=int(g.integers(1, 4)), replace=False)]:
            f = g.uniform(0.3, 0.8)
            calc[ix, iy] = f
            adi[ix, iy] *= 1.0 - f
            fg[ix, iy] *= 1.0 - f
        phantoms[i] = np.stack([adi, fg, calc])
    geometry = config.geometry()
    transmission = simulate_transmission(phantoms, geometry, tables)
    images = {
        kvp: np.einsum(
            "nmxy,m->nxy",
            phantoms.astype(np.float64),
            (0.5 * tables[kvp].spectrum * tables[kvp].mu_per_cm).sum(axis=1),
        ).astype(np.float32)
        for kvp in KVPS
    }
    return SyntheticData(phantoms, transmission, images, geometry)


def write_synthetic_dir(data: SyntheticData, root: str | Path) -> None:
    """Write the on-disk layout the loader reads (uncompressed .npy), see adapter-spec section 6:
    Phantom_{Adipose,Fibroglandular,Calcification}.npy (N,n,n), {low,high}kVpTransmission.npy,
    {low,high}kVpImages.npy.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for j, name in enumerate(MATERIAL_FILE_NAMES):
        np.save(root / f"Phantom_{name}.npy", np.ascontiguousarray(data.phantoms[:, j]))
    for kvp, prefix in ((50, "low"), (80, "high")):
        np.save(root / f"{prefix}kVpTransmission.npy", data.transmission[kvp])
        np.save(root / f"{prefix}kVpImages.npy", data.images[kvp])
