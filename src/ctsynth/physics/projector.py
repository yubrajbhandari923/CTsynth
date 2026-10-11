"""2D parallel-beam projector, exact pixel-constant (Siddon) chord lengths, matrix-free."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ctsynth.physics.materials import _all_finite


def _slab(
    p: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Parameter interval t in [t_lo, t_hi] where p + t*d lies in lo <= . <= hi (per axis)."""
    par = np.abs(d) < 1e-12  # shortcut: near-parallel to this axis, decided by position only
    ds = np.where(par, 1.0, d)
    ta, tb = (lo - p) / ds, (hi - p) / ds
    inside = (p > lo) & (p < hi)
    t_lo = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(ta, tb))
    t_hi = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(ta, tb))
    return t_lo, t_hi


def _chord_matrix(
    nx: int,
    ny: int,
    dx: float,
    angles: np.ndarray,
    n_det: int,
    det_spacing: float,
    det_offset: float,
) -> np.ndarray:
    """Exact pixel/ray chord lengths, float64, shape (n_view*n_det, ny*nx). Rows: view-major."""
    s = (np.arange(n_det) - (n_det - 1) / 2.0) * det_spacing + det_offset  # (D,)
    c, sn = np.cos(angles)[:, None], np.sin(angles)[:, None]  # (V, 1)
    # line p(t) = s*n + t*d with d = (cos, sin), n = (-sin, cos)
    px = (-s[None, :] * sn)[:, :, None, None]
    py = (s[None, :] * c)[:, :, None, None]
    dxv, dyv = c[:, :, None, None], sn[:, :, None, None]
    xc = ((np.arange(nx) - (nx - 1) / 2.0) * dx)[None, None, None, :]
    yc = ((np.arange(ny) - (ny - 1) / 2.0) * dx)[None, None, :, None]
    h = dx / 2.0
    tx0, tx1 = _slab(px, dxv, xc - h, xc + h)
    ty0, ty1 = _slab(py, dyv, yc - h, yc + h)
    length = np.maximum(0.0, np.minimum(tx1, ty1) - np.maximum(tx0, ty0))
    return np.asarray(length.reshape(len(angles) * n_det, ny * nx), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class ParallelBeamProjector2D:
    """Linear operator P: images (..., ny, nx) -> sinograms (..., n_view, n_det).

    Geometry (all lengths cm, angles radians; full definition in the spec):
        pixel (iy, ix) is the square centred at x = (ix-(nx-1)/2)*pixel_size,
        y = (iy-(ny-1)/2)*pixel_size, side pixel_size, constant value.
        View angle th: direction d = (cos th, sin th), normal n = (-sin th, cos th).
        Detector j of view v is the line {p : p.n = s_j},
        s_j = (j-(n_det-1)/2)*det_spacing + det_offset.
        (P x)[v, j] = sum_pixels x[pixel] * length(line_vj intersect pixel square).
    Rays lying exactly on a pixel edge are unspecified; callers/tests avoid them.

    Attributes:
        nx, ny: image width/height in pixels (>= 1).
        pixel_size: cm, > 0.
        angles: (n_view,) radians, n_view >= 1.
        n_det: detectors per view (>= 1).
        det_spacing: cm, > 0.
        det_offset: cm, shift of the detector array along n.
    """

    nx: int
    ny: int
    pixel_size: float
    angles: torch.Tensor
    n_det: int
    det_spacing: float
    det_offset: float = 0.0
    _matrix64: torch.Tensor = field(init=False, repr=False, compare=False)
    _cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1 or self.n_det < 1:
            raise ValueError("nx, ny and n_det must be >= 1")
        if not self.pixel_size > 0 or not self.det_spacing > 0:
            raise ValueError("pixel_size and det_spacing must be > 0")
        a = self.angles
        if not a.is_floating_point() or a.dim() != 1 or a.numel() < 1 or not _all_finite(a):
            raise ValueError("angles must be a finite floating 1-D tensor with >= 1 entry")
        geom = (self.nx, self.ny, float(self.pixel_size), self.angles.detach().cpu().double())
        mat = _chord_matrix(
            geom[0],
            geom[1],
            geom[2],
            geom[3].numpy(),
            self.n_det,
            float(self.det_spacing),
            float(self.det_offset),
        )
        object.__setattr__(self, "_matrix64", torch.from_numpy(mat))
        object.__setattr__(self, "_cache", {})

    @property
    def n_view(self) -> int:
        """Number of views."""
        return int(self.angles.shape[0])

    def _matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key not in self._cache:
            self._cache[key] = self._matrix64.to(device=device, dtype=dtype)
        return self._cache[key]

    def _check_image(self, image: torch.Tensor) -> None:
        if image.dim() < 2 or tuple(image.shape[-2:]) != (self.ny, self.nx):
            raise ValueError(f"image must end with shape ({self.ny}, {self.nx})")
        if not image.is_floating_point():
            raise ValueError("image must be floating point")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """(..., ny, nx) -> (..., n_view, n_det); linear, differentiable, float dtype preserved."""
        self._check_image(image)
        m = self._matrix(image.device, image.dtype)  # (n_view*n_det, ny*nx)
        flat = image.reshape(*image.shape[:-2], self.ny * self.nx)
        out = flat @ m.T
        return out.reshape(*out.shape[:-1], self.n_view, self.n_det)

    def adjoint(self, sino: torch.Tensor) -> torch.Tensor:
        """(..., n_view, n_det) -> (..., ny, nx); exact transpose of `forward`."""
        if sino.dim() < 2 or tuple(sino.shape[-2:]) != (self.n_view, self.n_det):
            raise ValueError(f"sinogram must end with shape ({self.n_view}, {self.n_det})")
        m = self._matrix(sino.device, sino.dtype)
        flat = sino.reshape(*sino.shape[:-2], self.n_view * self.n_det)
        out = flat @ m
        return out.reshape(*out.shape[:-1], self.ny, self.nx)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> ParallelBeamProjector2D:
        """Copy with `angles` moved/cast."""
        return ParallelBeamProjector2D(
            self.nx,
            self.ny,
            self.pixel_size,
            self.angles.to(device=device, dtype=dtype),
            self.n_det,
            self.det_spacing,
            self.det_offset,
        )
