"""Self-tests of the independent references and hand-computed constants (pass now)."""

import math

import aapm_reference as ref
import numpy as np

R, SDD, FOV = 50.0, 100.0, 18.0


def test_detector_length_formula() -> None:
    u = ref.det_u(1024, FOV, R, SDD)
    length = 2 * (u[-1] - u[0]) * 1024 / (2 * 1023)  # (u1-u0) = L(1023/1024)
    assert math.isclose(length, 36.598, abs_tol=2e-3)  # audit value 36.598 cm


def test_full_square_central_chord() -> None:
    """All-adipose phantom, view 0, bin nearest 0: chord = 18 * sqrt(100^2+u^2)/100 (exact)."""
    n = 16
    ph = np.zeros((1, 3, n, n))
    ph[0, 0] = 1.0
    p = ref.ref_line_integrals(ph, n, FOV, 8, 24, R, SDD)
    u = ref.det_u(24, FOV, R, SDD)[12]  # |u| ~ 0.76 cm, ray stays inside the square
    assert math.isclose(
        p[0, 0, 0, 12], 0.18 * math.hypot(100.0, u), rel_tol=1e-12
    )  # exact geometry
    assert p[0, 1:].max() == 0.0


def test_reference_index_order_is_ix_iy() -> None:
    """A pixel at large x (ix=n-1) and y<0 (iy=n//4) is hit from view 0 only by bins with u<0."""
    n = 16
    ph = np.zeros((1, 3, n, n))
    ph[0, 0, n - 1, n // 4] = 1.0
    p = ref.ref_line_integrals(ph, n, FOV, 8, 24, R, SDD)[0, 0, 0]
    assert p[:12].sum() > 0 and p[12:].sum() == 0.0


def test_reference_transmission_monoenergetic() -> None:
    p = np.full((1, 3, 4, 2), 2.0)
    mu = np.array([[0.1], [0.2], [0.3]])  # one energy bin, s such that 0.5*s = 1
    t = ref.ref_transmission(p, np.array([2.0]), mu, 80)
    assert np.allclose(t, math.exp(-1.2), rtol=1e-14)  # exp(-(0.1+0.2+0.3)*2)
    assert t.shape == (1, 2, 2)


def test_ref_s1_s2_hand_values() -> None:
    t = np.zeros((2, 3, 2, 2))
    r = np.zeros_like(t)
    r[0, 0] = 0.1  # case 0: 4 px * 0.01 / 12
    r[1, 0, 0, 0] = 0.3  # case 1: 0.09 / 12
    want = (math.sqrt(1 / 300) + math.sqrt(0.0075)) / 2
    assert math.isclose(ref.ref_s1(t, r), want, rel_tol=1e-12)
    assert math.isclose(want, 0.0721688, abs_tol=1e-7)  # hand arithmetic
    assert ref.ref_s2(t, r, 2) == 0.0  # truth all zero -> every ROI skipped
