"""Projector tests: tests 3 (analytic chords) and 11 (adjoint) plus dense brute-force agreement."""

import math
from typing import Any

import numpy as np
import physics_reference as ref
import pytest
import torch

from ctsynth.physics import ParallelBeamProjector2D

F64 = torch.float64
# (nx, ny, dx, angles, n_det, ds, off): non-square, odd/even, offset, single view, 1x1
GEOMS = [
    (7, 5, 0.3, (0.0, 0.4, 1.3, 2.9), 9, 0.25, 0.0),
    (4, 9, 0.1, (0.9, 2.2, 4.0), 3, 0.2, 0.07),
    (16, 16, 0.05, (0.0, math.pi / 4, math.pi / 2, 2.0), 20, 0.04, 0.013),
    (5, 5, 0.2, (1.1,), 6, 0.3, 0.0),
    (1, 1, 0.5, (0.3, 1.0), 2, 0.2, 0.01),
]


def _proj(g: tuple[Any, ...]) -> ParallelBeamProjector2D:
    nx, ny, dx, ang, nd, ds, off = g
    return ParallelBeamProjector2D(nx, ny, dx, torch.tensor(ang, dtype=F64), nd, ds, off)


@pytest.mark.parametrize("g", GEOMS)
def test_t11_adjoint_inner_product(g: tuple[Any, ...], gen: torch.Generator) -> None:
    """Law: <Px,y> = <x,P^T y> (exact adjoint pair). Reference: the two inner products computed
    with float64 sums from seeded random tensors. Tolerance 1e-12 relative to |Px||y|: float64
    rounding is ~sqrt(N)*1e-16 ~ 1e-14, leaving two orders of margin."""
    p = _proj(g)
    x = torch.randn(2, p.ny, p.nx, generator=gen, dtype=F64)
    y = torch.randn(2, p.n_view, p.n_det, generator=gen, dtype=F64)
    lhs = (p.forward(x) * y).sum()
    rhs = (x * p.adjoint(y)).sum()
    scale = p.forward(x).norm() * y.norm()
    assert abs(float(lhs - rhs)) <= 1e-12 * float(scale) + 1e-300


@pytest.mark.parametrize("g", GEOMS)
def test_dense_matrix_agreement(g: tuple[Any, ...], gen: torch.Generator) -> None:
    """forward == brute-force dense matrix of exact pixel/ray chord lengths (reference:
    physics_reference.dense_matrix, python loops + slab clipping), adjoint == its transpose.
    atol 1e-12 * max|result|: only float64 summation-order differences. Geometries avoid rays on
    pixel edges, where the definition is unspecified."""
    nx, ny, dx, ang, nd, ds, off = g
    mat = ref.dense_matrix(nx, ny, dx, np.array(ang), nd, ds, off)
    p = _proj(g)
    x = torch.randn(ny, nx, generator=gen, dtype=F64)
    y = torch.randn(len(ang), nd, generator=gen, dtype=F64)
    want_f = (mat @ x.numpy().ravel()).reshape(len(ang), nd)
    want_a = (mat.T @ y.numpy().ravel()).reshape(ny, nx)
    assert np.allclose(
        p.forward(x).numpy(), want_f, rtol=0, atol=1e-12 * max(1.0, abs(want_f).max())
    )
    assert np.allclose(
        p.adjoint(y).numpy(), want_a, rtol=0, atol=1e-12 * max(1.0, abs(want_a).max())
    )


def test_batch_independence_and_linearity(gen: torch.Generator) -> None:
    """Channels are independent (P(stack) == stack of P each) and P is linear."""
    p = _proj(GEOMS[0])
    x = torch.randn(3, 5, 7, generator=gen, dtype=F64)
    out = p.forward(x)
    for m in range(3):
        assert torch.allclose(out[m], p.forward(x[m]), rtol=0, atol=1e-13)
    assert torch.allclose(p.forward(2.0 * x[0] - x[1]), 2.0 * out[0] - out[1], rtol=0, atol=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.234, 2.5, 3.0, math.pi / 4])
def test_t03_uniform_field_equals_analytic_rectangle_chord(theta: float) -> None:
    """Law: line integral of uniform rho through a rectangle = rho * chord length (exact, because
    pixel chords partition the rectangle chord). Reference: analytic slab-clipped chord of the
    whole rectangle. rtol 1e-11 (float64 rounding of ~20 summed pieces). Offsets avoid
    pixel edges."""
    nx, ny, dx, rho = 8, 5, 0.3, 1.7
    p = ParallelBeamProjector2D(nx, ny, dx, torch.tensor([theta], dtype=F64), 11, 0.21, 0.017)
    got = p.forward(torch.full((ny, nx), rho, dtype=F64))[0].numpy()
    s = ref.det_positions(11, 0.21, 0.017)
    want = rho * np.array(
        [
            ref.chord_rect(sj, theta, -nx * dx / 2, nx * dx / 2, -ny * dx / 2, ny * dx / 2)
            for sj in s
        ]
    )
    assert np.allclose(got, want, rtol=1e-11, atol=1e-13)


def test_t03_axis_aligned_exact() -> None:
    """Axis-aligned views through pixel centres in a uniform field: A = rho*N*dx exactly
    (theta=0 -> sum along x: nx*dx; theta=pi/2 -> ny*dx). rtol 1e-12 (cos(pi/2)=6e-17 tilt)."""
    nx, ny, dx, rho = 6, 4, 0.25, 0.9
    p = ParallelBeamProjector2D(nx, ny, dx, torch.tensor([0.0, math.pi / 2], dtype=F64), 6, dx)
    # detectors at s=(j-2.5)dx; rows centred at (i-1.5)dx -> dets 1..4 hit rows
    out = p.forward(torch.full((ny, nx), rho, dtype=F64)).numpy()
    assert np.allclose(out[0, 1:5], rho * nx * dx, rtol=1e-12)
    assert np.allclose(out[0, [0, 5]], 0.0, atol=1e-13)  # beyond image rows: s=+-2.5dx > 2dx
    # theta=pi/2: ray at x=-s_j; columns at (i-2.5)dx, s_j=(j-2.5)dx -> all 6 hit, each ny*dx
    assert np.allclose(out[1, :], rho * ny * dx, rtol=1e-12)


def test_t03_square_diagonal() -> None:
    """45 degree central ray through an n x n uniform square (through pixel corners) has length
    n*dx*sqrt(2). Reference: geometry. rtol 1e-10 tolerates corner-touch rounding."""
    n, dx = 6, 0.2
    p = ParallelBeamProjector2D(n, n, dx, torch.tensor([math.pi / 4], dtype=F64), 1, dx)
    got = float(p.forward(torch.ones(n, n, dtype=F64))[0, 0])
    assert math.isclose(got, n * dx * math.sqrt(2), rel_tol=1e-10)


@pytest.mark.parametrize("theta", [0.0, 0.6, 1.5707963267948966, 2.2])
def test_t03_disk_vs_analytic_chord(theta: float) -> None:
    """Pixel-centre-rasterised uniform disk vs analytic chord 2 sqrt(R^2-(s-s_c)^2) (reference:
    closed form), for rays with |s-s_c| <= 0.8 R. Tolerance per ray 6*dx*rho: each of 2 boundary
    crossings can mis-assign at most 2 pixels along the ray (non-grazing: incidence cosine >= 0.6)
    each contributing <= sqrt(2)*dx -> <= 5.7*dx. Error is O(dx), as pixel size drives it."""
    nx = ny = 40
    dx, rho, radius, cx, cy = 0.05, 1.3, 0.7, 0.1, -0.05
    img = torch.tensor(ref.disk_image(nx, ny, dx, radius, cx, cy, rho))
    p = ParallelBeamProjector2D(nx, ny, dx, torch.tensor([theta], dtype=F64), 61, 0.0237, 0.0011)
    got = p.forward(img)[0].numpy()
    s = ref.det_positions(61, 0.0237, 0.0011)
    sc = -cx * math.sin(theta) + cy * math.cos(theta)
    sel = np.abs(s - sc) <= 0.8 * radius
    want = rho * ref.disk_chord(s, theta, radius, cx, cy)
    assert sel.sum() > 20
    assert np.max(np.abs(got - want)[sel]) <= 6 * dx * rho
