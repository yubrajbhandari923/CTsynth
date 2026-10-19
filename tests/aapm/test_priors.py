"""Stage-0 prior functions and lambda-window logic (EXP-035). FROZEN before implementation.

SYNTHETIC ARRAYS ONLY: these tests pin definitions and symmetries (e.g. bimodality and TV cannot
tell adipose from fibroglandular), never AAPM performance. `frac` is (N,3,H,W), tissue (N,H,W) bool.
"""

import numpy as np
import pytest
import torch

from ctsynth.aapm import priors as P

pytestmark = pytest.mark.unit


def _frac(a: np.ndarray, c: np.ndarray | None = None) -> torch.Tensor:
    """Tissue map from adipose fraction a (H,W) and calcification c: adipose a, fibro 1-a-c."""
    c = np.zeros_like(a) if c is None else c
    f = np.stack([a, 1 - a - c, c])[None]
    return torch.tensor(f, dtype=torch.float64)


def test_calc_l1_is_mean_abs_over_tissue_only() -> None:
    a = np.full((2, 2), 0.5)
    c = np.array([[0.1, -0.2], [0.0, 5.0]])
    tissue = torch.tensor([[[True, True], [True, False]]])  # the 5.0 pixel is air
    assert P.calc_l1(_frac(a, c), tissue).item() == pytest.approx((0.1 + 0.2 + 0.0) / 3)
    assert P.calc_l1(_frac(a), tissue).item() == 0.0


def test_calc_l0_surrogate_bounds() -> None:
    a = np.full((2, 2), 0.5)
    tissue = torch.ones(1, 2, 2, dtype=torch.bool)
    assert P.calc_l0(_frac(a), tissue).item() == 0.0
    big = P.calc_l0(_frac(a, np.full((2, 2), 0.5)), tissue, eps=1e-3).item()
    assert 0.999 < big < 1.0
    eps = 1e-2
    c = np.full((2, 2), eps)  # c = eps => c^2/(c^2+eps^2) = 1/2 exactly
    assert P.calc_l0(_frac(a, c), tissue, eps=eps).item() == pytest.approx(0.5)


def test_bimodal_zero_at_vertices_max_at_half_and_swap_symmetric() -> None:
    tissue = torch.ones(1, 2, 2, dtype=torch.bool)
    onehot = _frac(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert P.bimodal(onehot, tissue).item() == pytest.approx(0.0, abs=1e-7)
    assert P.bimodal(_frac(np.full((2, 2), 0.5)), tissue).item() == pytest.approx(0.25)
    skew = _frac(np.array([[0.1, 0.9], [0.8, 0.3]]))
    assert P.bimodal(skew, tissue).item() == pytest.approx(
        P.bimodal(P.wrong_label(skew, tissue), tissue).item()
    )


def test_tv_a_step_edge_value_and_ignores_air_edges_and_swap_symmetric() -> None:
    a = np.zeros((4, 4))
    a[:, 2:] = 1.0
    tissue = torch.ones(1, 4, 4, dtype=torch.bool)
    # 24 neighbouring pairs (12 horizontal + 12 vertical), 4 across the edge with |da|=1
    assert P.tv_a(_frac(a), tissue).item() == pytest.approx(4 / 24)
    half = tissue.clone()
    half[:, :, 2:] = False  # right half is air: the step edge no longer joins two tissue pixels
    assert P.tv_a(_frac(a), half).item() == 0.0
    assert P.tv_a(_frac(np.full((4, 4), 0.37)), tissue).item() == 0.0
    f = _frac(a)
    assert P.tv_a(P.wrong_label(f, tissue), tissue).item() == pytest.approx(
        P.tv_a(f, tissue).item()
    )


def test_wrong_label_is_involution_and_leaves_air_and_calcification() -> None:
    a = np.array([[0.2, 0.7], [0.5, 0.9]])
    c = np.array([[0.0, 0.1], [0.0, 0.0]])
    f = _frac(a, c)
    tissue = torch.tensor([[[True, True], [True, False]]])
    w = P.wrong_label(f, tissue)
    assert torch.allclose(w[0, 0, 0, 0], f[0, 1, 0, 0])  # adipose <-> fibro inside tissue
    assert torch.equal(w[:, :, 1, 1], f[:, :, 1, 1])  # air pixel untouched
    assert torch.equal(w[:, 2], f[:, 2])  # calcification untouched
    assert torch.equal(P.wrong_label(w, tissue), f)


def test_w1_marginal_zero_at_reference_and_detects_label_swap_only_when_skewed() -> None:
    tissue = torch.ones(1, 8, 8, dtype=torch.bool)
    rng = np.random.default_rng(0)
    skewed = np.clip(rng.beta(0.3, 0.1, size=(8, 8)), 0, 1)  # mostly near 1: a skewed marginal
    f = _frac(skewed)
    ref = P.ref_quantiles(skewed.ravel(), 64)
    assert P.w1_marginal(f, tissue, ref).item() == pytest.approx(0.0, abs=1e-12)
    assert P.w1_marginal(P.wrong_label(f, tissue), tissue, ref).item() > 0.1  # skew breaks symmetry
    sym = np.concatenate([np.full(32, 0.2), np.full(32, 0.8)]).reshape(8, 8)
    ref_sym = P.ref_quantiles(sym.ravel(), 64)
    fs = _frac(sym)
    assert P.w1_marginal(P.wrong_label(fs, tissue), tissue, ref_sym).item() == pytest.approx(
        P.w1_marginal(fs, tissue, ref_sym).item(), abs=1e-12
    )


def test_argmin_t_picks_the_minimum_of_mse_plus_lambda_r() -> None:
    t = np.linspace(-0.4, 0.4, 41)
    assert P.argmin_t(t, 1e-9 * t**2, np.abs(t - 0.1), 0.0) == pytest.approx(0.0)
    # lam*(t-0.1)^2 + 1e-9 t^2 -> t* = 0.1*lam/(lam+1e-9) ~ 0.1 for lam >> 1e-9
    assert P.argmin_t(t, 1e-9 * t**2, (t - 0.1) ** 2, 1.0) == pytest.approx(0.1)


def test_lambda_window_hand_computed() -> None:
    t = np.linspace(-0.4, 0.4, 81)
    lams = np.array([1e-8, 1e-6, 1e-4, 1e-2])
    r_n = np.abs(t)  # minimised at the truth along n
    mse_id, r_id = (
        1e-2 * t**2,
        (t - 0.05) ** 2,
    )  # prior biased by 0.05 along an identifiable direction
    lo, hi = P.lambda_window(t, r_n, t, mse_id, r_id, lams, res=2e-8, dt=0.1, tol=0.02)
    # lam_min = res / (R(0.1)-R(0)) = 2e-7 -> first grid value >= it is 1e-6.
    # |argmin_id| = 0.05 lam/(1e-2+lam) <= 0.02 <=> lam <= 6.67e-3 -> largest grid value is 1e-4.
    assert (lo, hi) == (1e-6, 1e-4)


def test_lambda_window_empty_for_swap_symmetric_prior() -> None:
    t = np.linspace(-0.4, 0.4, 81)
    lams = np.logspace(-10, 0, 11)
    lo, hi = P.lambda_window(
        t, np.zeros_like(t), t, 1e-2 * t**2, np.zeros_like(t), lams, res=2e-8, dt=0.1, tol=0.02
    )
    assert not lo <= hi  # a prior flat along n offers no margin at any weight: empty window
