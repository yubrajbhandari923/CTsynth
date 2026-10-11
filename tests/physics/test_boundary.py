"""T14 boundary cases."""

import math

import numpy as np
import physics_reference as ref
import torch

from ctsynth.physics import ParallelBeamProjector2D, SpectralForwardModel, Spectrum

F64 = torch.float64
NX, NY, DX = 6, 5, 0.2


def _model(
    scatter: float | None = None, off: float = 0.0, angles: tuple[float, ...] = (0.0,)
) -> SpectralForwardModel:
    p = ref.build_projector(NX, NY, DX, angles, NY, DX, off)
    return SpectralForwardModel(ref.build_basis(), ref.build_spectrum(), p, scatter)


def _uni(a: float, b: float) -> torch.Tensor:
    return torch.stack([torch.full((NY, NX), a, dtype=F64), torch.full((NY, NX), b, dtype=F64)])


def test_t14_vacuum_and_tiny_density() -> None:
    """rho=0 gives exactly the unattenuated counts; rho=1e-12 g/cm^3 gives them to first order:
    1 - counts/unatt <= max_kappa * L * rho = 6.4*1.2*1e-12*2 < 2e-11, so rtol 1e-10 (reference:
    numpy unattenuated sums)."""
    unatt = ref.toy_arrays()["w"].sum(axis=1)[:, None, None]
    m = _model()
    assert np.allclose(m.expected_counts(_uni(0.0, 0.0)).numpy(), unatt, rtol=1e-13)
    tiny = m.expected_counts(_uni(1e-12, 1e-12)).numpy()
    assert np.allclose(tiny, unatt, rtol=1e-10) and (tiny <= unatt * (1 + 1e-15)).all()


def test_t14_extreme_attenuation_underflow_is_guarded() -> None:
    """rho=1e4 g/cm^3 of both materials: every exponent > 2.6e3 > 745 (float64 exp underflow),
    so counts must be exactly 0 (no NaN/inf), gradients finite (and 0); with scatter fraction
    0.01 the counts equal exactly 0.01*unattenuated (rtol 1e-12)."""
    rho = _uni(1e4, 1e4).requires_grad_(True)
    c = _model().expected_counts(rho)
    assert torch.isfinite(c).all() and (c == 0).all()
    c.sum().backward()
    assert rho.grad is not None and torch.isfinite(rho.grad).all()
    unatt = ref.toy_arrays()["w"].sum(axis=1)[:, None, None]
    cs = _model(0.01).expected_counts(_uni(1e4, 1e4)).numpy()
    assert np.allclose(cs, 0.01 * unatt, rtol=1e-12)


def test_t14_gradients_finite_at_zero_density() -> None:
    """Gradient at rho=0 is finite and negative-definite-signed: d counts/d rho <= 0 for all
    pixels (more material, fewer photons); reference: sign of -kappa*W*exp(.) <= 0."""
    rho = torch.zeros(2, NY, NX, dtype=F64, requires_grad=True)
    _model().expected_counts(rho).sum().backward()
    assert rho.grad is not None and torch.isfinite(rho.grad).all() and (rho.grad <= 0).all()


def test_t14_thin_one_pixel_line_no_partial_volume_blur() -> None:
    """Documented behaviour: pixels are piecewise constant, so a 1-pixel-wide vertical line
    (column ix=2 of a 5x5 grid, rho=c) is seen with its full value or not at all (no linear
    interpolation spreading). View theta=pi/2 (rays along y, ray x = -s) with det_offset=0.25 dx:
    only detector j=2 (x=-0.25dx inside column 2) integrates it: A = c*ny*dx; all other detectors
    are 0 (atol 1e-13). View theta=0 crosses the line once per row: A = c*dx for every detector
    (rays at row centres). rtol 1e-12."""
    n, dx, c = 5, 0.2, 1.4
    img = torch.zeros(n, n, dtype=F64)
    img[:, 2] = c
    p = ParallelBeamProjector2D(n, n, dx, torch.tensor([math.pi / 2], dtype=F64), n, dx, 0.25 * dx)
    out = p.forward(img)[0].numpy()
    want = np.zeros(n)
    want[2] = c * n * dx
    assert np.allclose(out, want, rtol=1e-12, atol=1e-13)
    p0 = ParallelBeamProjector2D(n, n, dx, torch.tensor([0.0], dtype=F64), n, dx)
    assert np.allclose(p0.forward(img)[0].numpy(), c * dx, rtol=1e-12)


def test_t14_single_view_single_detector_single_bin() -> None:
    """Degenerate sizes: 3x3 uniform image, 1 view (theta=0), 1 detector at s=0 (middle row
    centre), 1 energy and 1 bin -> output shape (1,1,1) equal to I0 exp(-kappa . rho L), L=3dx
    (reference: closed form power law at E=61.7). rtol 1e-12."""
    dx, rho = 0.2, (0.4, 0.8)
    p = ParallelBeamProjector2D(3, 3, dx, torch.tensor([0.0], dtype=F64), 1, dx)
    spec = Spectrum(*(torch.tensor(v, dtype=F64) for v in ([61.7], [1e5], [1.0], [[1.0]])))
    m = SpectralForwardModel(ref.build_basis(), spec, p)
    img = torch.stack([torch.full((3, 3), r, dtype=F64) for r in rho])
    got = m.expected_counts(img)
    assert got.shape == (1, 1, 1)
    expo = 3 * dx * (rho[0] * ref.kappa_pe(np.array(61.7)) + rho[1] * ref.kappa_cs(np.array(61.7)))
    assert math.isclose(float(got), 1e5 * math.exp(-expo), rel_tol=1e-12)
