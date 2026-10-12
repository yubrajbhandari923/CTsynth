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

from ctsynth.aapm.schema import Array, GeometrySpec, ModelTable


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
        raise NotImplementedError


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


def line_integrals(phantoms: Array, geometry: GeometrySpec) -> Array:
    """Path lengths P[case, material, view, bin] in cm, float64, FULL interlaced view axis.

    P = sum over pixels of fraction * (exact length in cm of the source->detector-bin ray inside
    that pixel square). Geometry per docs/aapm/adapter-spec.md section 3.
    """
    raise NotImplementedError


def simulate_transmission(
    phantoms: Array, geometry: GeometrySpec, tables: dict[int, ModelTable]
) -> dict[int, Array]:
    """T = sum_E 0.5*s_kVp(E)*exp(-sum_m mu_m(E)*P_m), I0 = 1, noiseless, float32, per kVp.

    kVp 80 uses even views, 50 odd views (schema.view_indices).
    """
    raise NotImplementedError


def generate(config: SyntheticConfig, tables: dict[int, ModelTable]) -> SyntheticData:
    """Generate a self-consistent synthetic dataset (deterministic given config.seed).

    Per case: an off-centre asymmetric (semi-axes and centre differ in x and y) elliptical
    'breast' of tissue on air; an inner fibroglandular ellipse, remainder adipose;
    boundary pixels carry partial coverage c in (0,1) (supersampled),
    so tissue sum == c there; 1..3 single-pixel calcification
    inclusions (fraction 0.3..0.8) inside full tissue, taken from the local tissue so the sum is
    conserved. Guarantees air (sum==0), tissue (sum==1) and band pixels all exist.
    """
    raise NotImplementedError


def write_synthetic_dir(data: SyntheticData, root: str | Path) -> None:
    """Write the on-disk layout the loader reads (uncompressed .npy), see adapter-spec section 6:
    Phantom_{Adipose,Fibroglandular,Calcification}.npy (N,n,n), {low,high}kVpTransmission.npy,
    {low,high}kVpImages.npy.
    """
    raise NotImplementedError
