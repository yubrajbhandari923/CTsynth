"""Analytic disk phantoms: supersampled volume-fraction truth and exact ray chords.

World frame is the XCIST scanner frame (x right, y up, mm). Arrays are `(iy, ix)` with
`x = (ix-(n-1)/2)*dx`, `y = -(iy-(n-1)/2)*dx` (row 0 is +y, verified against XCIST in
tests/paired/test_xcist.py). Disks add `delta` to the material volume fractions inside them, so
an insert is a disk with delta = (-f, +f, ...) inside a body disk.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Disk:
    cx_mm: float
    cy_mm: float
    r_mm: float
    delta: tuple[float, ...]  # per-material volume-fraction change inside the disk


def fractions(disks: list[Disk], n: int, dx_mm: float, ss: int = 4) -> np.ndarray:
    """(n_material, 1, n, n) area-coverage volume fractions (ss x ss subsamples per pixel)."""
    n_mat = len(disks[0].delta)
    sub = (np.arange(n * ss) + 0.5) / ss - 0.5  # subsample centres in pixel-index units
    ix = (sub - (n - 1) / 2) * dx_mm
    x, y = np.meshgrid(ix, -ix, indexing="xy")  # y[row] = -(row coordinate)
    out = np.zeros((n_mat, n * ss, n * ss))
    for d in disks:
        inside = (x - d.cx_mm) ** 2 + (y - d.cy_mm) ** 2 <= d.r_mm**2
        out += np.asarray(d.delta)[:, None, None] * inside
    out = out.reshape(n_mat, n, ss, n, ss).mean(axis=(2, 4))
    return out[:, None]


def chords(disks: list[Disk], src: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Path-length-weighted line integrals (mm): (n_material, n_view, n_det).

    `src` is (n_view, 2), `det` is (n_view, n_det, 2): infinite lines through each pair.
    """
    n_mat = len(disks[0].delta)
    u = det - src[:, None, :]
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    out = np.zeros((n_mat, *det.shape[:2]))
    for d in disks:
        w = np.array([d.cx_mm, d.cy_mm]) - src[:, None, :]
        dist = np.abs(w[..., 0] * u[..., 1] - w[..., 1] * u[..., 0])
        chord = 2 * np.sqrt(np.clip(d.r_mm**2 - dist**2, 0, None))
        out += np.asarray(d.delta)[:, None, None] * chord
    return out
