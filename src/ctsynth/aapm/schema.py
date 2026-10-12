"""Frozen AAPM DL-Spectral-CT schema conventions (spec: docs/aapm/adapter-spec.md).

Constants and the `GeometrySpec` container are final; every function body is a stub for Haiku.
Every violation raises `SchemaError` whose `.convention` is one of the CONV_* names below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[Any]

# --- convention names (value of SchemaError.convention) ---
CONV_SHAPE = "shape"
CONV_DTYPE = "dtype"
CONV_RANGE = "range"
CONV_FINITE = "finite"
CONV_MATERIAL_ORDER = "material_order"
CONV_KVP_PARITY = "kvp_parity"
CONV_PARTITION = "partition"
CONV_TABLE = "table"

# --- frozen conventions ---
# the ONE channel order, everywhere
MATERIALS: tuple[str, ...] = ("adipose", "fibroglandular", "calcification")
MATERIAL_FILE_NAMES: tuple[str, ...] = (
    "Adipose",
    "Fibroglandular",
    "Calcification",
)  # Phantom_X.npy
KVP_HIGH = 80  # EVEN views (0, 2, 4, ...)
KVP_LOW = 50  # ODD views (1, 3, 5, ...)
KVPS: tuple[int, int] = (KVP_LOW, KVP_HIGH)
DTYPE = np.dtype("float32")
# in memory; on disk one (case, ix, iy) file per material
PHANTOM_AXES = ("case", "material", "ix", "iy")
IMAGE_AXES = ("case", "ix", "iy")  # image[ix, iy]; NOT row/col
TRANSMISSION_AXES = ("case", "view_within_kvp", "bin")
MAP_UNITS = "volume fraction (dimensionless; inferred, not stated by the kit)"
MU_UNITS = "cm^-1"
PARTITION_TOL = 1e-3  # |sum-1| / |sum| tolerance defining tissue / air pixels (F-013 used 1e-3)
SUM_MAX = 1.001  # maps never sum above this (kit max 1.0000001)
MAP_EPS = 1e-6  # float32 slack on the [0, 1] per-map range
CALC_SUPPORT_THRESH = 1e-3  # calcification pixel = value above this
CALC_SUPPORT_MAX_FRAC = 0.05  # calcification support must be sparser than this (permutation guard)


class SchemaError(ValueError):
    """Violation of a frozen convention; `.convention` names which (CONV_*)."""

    def __init__(self, convention: str, message: str) -> None:
        super().__init__(f"[{convention}] {message}")
        self.convention = convention


@dataclass(frozen=True)
class GeometrySpec:
    """Circular fan-beam, flat detector. Real AAPM: REAL_GEOMETRY. Reduced sizes keep fov/R/SDD."""

    n_pix: int
    fov_cm: float
    n_views: int  # total over both kVp; must be even
    n_det: int
    source_radius_cm: float = 50.0
    sdd_cm: float = 100.0

    def pixel_cm(self) -> float:
        """fov_cm / n_pix (0.03515625 cm for the real geometry)."""
        raise NotImplementedError

    def detector_length_cm(self) -> float:
        """2 * tan(asin((fov_cm/2) / source_radius_cm)) * sdd_cm (36.598 cm real)."""
        raise NotImplementedError

    def view_angles(self) -> Array:
        """float64 (n_views,): angle_k = k * 2*pi / n_views, counter-clockwise from 0."""
        raise NotImplementedError


REAL_GEOMETRY = GeometrySpec(n_pix=512, fov_cm=18.0, n_views=512, n_det=1024)


@dataclass(frozen=True)
class ModelTable:
    """One committed model_data_{kVp}kVp.npy table."""

    kvp: int
    energies_kev: Array  # (E,) float64
    spectrum: Array  # (E,) float64, sum(0.5 * spectrum) == 1
    mu_per_cm: Array  # (3, E) float64, canonical MATERIALS order


def load_model_table(path: str | Path, kvp: int) -> ModelTable:
    """Load a (5, E) table: row0 energies keV, row1 spectrum, rows 2-4 mu (adipose, fibro, calc).

    Raises SchemaError(CONV_TABLE) if the shape is not (5, E) or sum(0.5*s) != 1 within 1e-4.
    """
    raise NotImplementedError


def kvp_of_view(view: int) -> int:
    """80 for even `view`, 50 for odd."""
    raise NotImplementedError


def view_indices(kvp: int, n_views: int) -> Array:
    """int64 indices into the full interlaced view axis: 80 -> 0,2,4,...; 50 -> 1,3,5,...

    Raises SchemaError(CONV_KVP_PARITY) for kvp not in (50, 80).
    """
    raise NotImplementedError


def deinterlace(full: Array) -> dict[int, Array]:
    """Split a full (N, n_views, n_det) array into {80: full[:, 0::2], 50: full[:, 1::2]}."""
    raise NotImplementedError


def check_material_order(names: Sequence[str]) -> None:
    """Raise SchemaError(CONV_MATERIAL_ORDER) unless tuple(names) == MATERIALS exactly."""
    raise NotImplementedError


def validate_phantoms(
    phantoms: Array, geometry: GeometrySpec, *, material_names: Sequence[str] | None = None
) -> None:
    """Validate (N, 3, n_pix, n_pix) float32 volume fractions in canonical order.

    Checks, each raising its own convention: ndim/shape (CONV_SHAPE), float32 (CONV_DTYPE), finite
    (CONV_FINITE), each map in [-MAP_EPS, 1+MAP_EPS] (CONV_RANGE), sum over materials <= SUM_MAX
    (CONV_PARTITION), `material_names` (if given) == MATERIALS (CONV_MATERIAL_ORDER), and
    calcification channel support fraction (> CALC_SUPPORT_THRESH) < CALC_SUPPORT_MAX_FRAC
    over the batch
    (CONV_MATERIAL_ORDER; detects calcification swapped with a bulk channel, NOT adipose<->fibro).
    """
    raise NotImplementedError


def validate_images(images: Array, geometry: GeometrySpec) -> None:
    """Validate (N, n_pix, n_pix) float32 finite image[ix, iy] arrays (shape/dtype/finite)."""
    raise NotImplementedError


def validate_transmission(transmission: Array, geometry: GeometrySpec) -> None:
    """Validate (N, n_views//2, n_det) float32, finite, values in (0, 1] (CONV_RANGE)."""
    raise NotImplementedError


def validate_dataset(
    phantoms: Array,
    transmission: Mapping[int, Array],
    images: Mapping[int, Array],
    geometry: GeometrySpec,
) -> None:
    """Validate all arrays plus cross-array conventions.

    `transmission` / `images` must be keyed by exactly {50, 80} (else CONV_KVP_PARITY), have equal N
    to `phantoms` (CONV_SHAPE), and mean(transmission[50]) < mean(transmission[80]) over the batch
    (CONV_KVP_PARITY: lower kVp attenuates more; detects swapped labels).
    """
    raise NotImplementedError
