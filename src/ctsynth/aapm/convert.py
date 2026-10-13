"""Axis conversion between the AAPM kit layout and the paired-ct truth layout.

AAPM (`schema.IMAGE_AXES`): a map is `a[ix, iy]` (x first); a case of materials is `(M, ix, iy)`.
Paired (`paired.schema`): truth is `(M, nz, ny, nx)` = `(iz, iy, ix)` per material; 2D slices have
nz == 1.

These functions only RELABEL INDICES (transpose + singleton z). They say nothing about physical
orientation: the kit does not state voxel size or the sign of its axes relative to a scanner
frame, so no affine can be derived here and none is invented. A leading batch axis (cases) is
passed through.
"""

import numpy as np


def aapm_to_paired(maps: np.ndarray) -> np.ndarray:
    """`(..., M, ix, iy)` -> `(..., M, 1, iy, ix)`. Returns a contiguous copy."""
    if maps.ndim < 3:
        raise ValueError(f"expected (..., M, ix, iy), got shape {maps.shape}")
    return np.ascontiguousarray(np.swapaxes(maps, -1, -2)[..., None, :, :])


def paired_to_aapm(truth: np.ndarray) -> np.ndarray:
    """`(..., M, 1, iy, ix)` -> `(..., M, ix, iy)`; the exact inverse of `aapm_to_paired`."""
    if truth.ndim < 4 or truth.shape[-3] != 1:
        raise ValueError(f"expected (..., M, 1, iy, ix) with nz == 1, got shape {truth.shape}")
    return np.ascontiguousarray(np.swapaxes(truth[..., 0, :, :], -1, -2))
