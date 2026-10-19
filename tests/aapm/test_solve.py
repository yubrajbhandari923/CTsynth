"""Per-case solver helpers for EXP-036 (spec: src/ctsynth/aapm/solve.py docstring). FROZEN before
implementation. Pure-numerics tests on SYNTHETIC problems: they say nothing about AAPM recovery."""

import math

import pytest
import torch

from ctsynth.aapm import solve as V

pytestmark = pytest.mark.unit
D = torch.float64


def _quad(d: torch.Tensor):
    return lambda x: 0.5 * (d * x * x).sum()


def test_minimize_reaches_tiny_loss_on_ill_conditioned_quadratic() -> None:
    d = torch.logspace(0, 4, 50, dtype=D)  # condition number 1e4
    x0 = torch.ones(50, dtype=D)
    x, info = V.minimize(_quad(d), x0, max_iter=1500, chunk=25)
    assert _quad(d)(x).item() <= 1e-10
    assert info["reason"] in {"abs_tol", "plateau"}
    assert x.shape == x0.shape and torch.equal(x0, torch.ones(50, dtype=D))  # x0 untouched


def test_minimize_trace_is_nonincreasing_and_starts_at_initial_loss() -> None:
    d = torch.logspace(0, 2, 20, dtype=D)
    x0 = torch.full((20,), 2.0, dtype=D)
    _, info = V.minimize(_quad(d), x0, max_iter=200, chunk=10)
    losses = [v for _, v in info["trace"]]
    assert info["trace"][0][0] == 0 and losses[0] == pytest.approx(_quad(d)(x0).item())
    assert all(b <= a * (1 + 1e-12) for a, b in zip(losses, losses[1:], strict=False))


def test_minimize_stops_on_plateau_when_loss_floor_is_above_abs_tol() -> None:
    def f(x: torch.Tensor) -> torch.Tensor:
        return 1.0 + 1e-9 * (x * x).sum()  # floor 1.0: relative improvement vanishes immediately

    x, info = V.minimize(f, torch.ones(5, dtype=D), max_iter=1500, chunk=25, patience=100)
    assert info["reason"] == "plateau" and info["n_iter"] <= 100 + 3 * 25


def test_minimize_respects_max_iter() -> None:
    def rosen(x: torch.Tensor) -> torch.Tensor:
        return (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()

    _, info = V.minimize(
        rosen, torch.full((8,), -1.0, dtype=D), max_iter=50, chunk=25, patience=10_000
    )
    assert info["reason"] == "max_iter" and info["n_iter"] <= 50 + 25


def test_to_fractions_arm_a_sums_to_one_in_mask_zero_outside_and_no_calcification() -> None:
    mask = torch.tensor([[[True, True], [False, True]]])
    a = torch.tensor([[[[0.2, 0.9], [0.5, 0.0]]]], dtype=D)  # (N,1,H,W)
    f = V.to_fractions(a, mask, "A")
    assert f.shape == (1, 3, 2, 2)
    assert torch.allclose(f.sum(1)[mask], torch.ones(3, dtype=D))
    assert (f.permute(0, 2, 3, 1)[~mask] == 0).all() and (f[:, 2] == 0).all()
    assert f[0, 0, 0, 1].item() == pytest.approx(0.9) and f[0, 1, 0, 1].item() == pytest.approx(0.1)


def test_to_fractions_arm_b_calcification_takes_from_fibroglandular() -> None:
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    x = torch.tensor([[[[0.3, 0.4]], [[0.1, 0.0]]]], dtype=D)  # channels (a, c)
    f = V.to_fractions(x, mask, "B")
    assert torch.allclose(f[0, :, 0, 0], torch.tensor([0.3, 0.6, 0.1], dtype=D))
    assert torch.allclose(f.sum(1), torch.ones(1, 1, 2, dtype=D))


def test_to_fractions_arm_c_is_identity_inside_mask_only() -> None:
    mask = torch.tensor([[[True, False]]])
    x = torch.arange(6, dtype=D).reshape(1, 3, 1, 2)
    f = V.to_fractions(x, mask, "C")
    assert torch.equal(f[0, :, 0, 0], x[0, :, 0, 0]) and (f[0, :, 0, 1] == 0).all()


def test_to_fractions_rejects_unknown_arm_and_wrong_channels() -> None:
    mask = torch.ones(1, 2, 2, dtype=torch.bool)
    with pytest.raises(ValueError):
        V.to_fractions(torch.zeros(1, 1, 2, 2, dtype=D), mask, "Z")
    with pytest.raises(ValueError):
        V.to_fractions(torch.zeros(1, 2, 2, 2, dtype=D), mask, "A")


def test_error_bands_parseval_partition_of_masked_rms() -> None:
    g = torch.Generator().manual_seed(0)
    err = torch.randn(1, 64, 64, generator=g, dtype=D)
    mask = torch.ones(1, 64, 64, dtype=torch.bool)
    b = V.error_bands(err, mask)
    total = math.sqrt((err**2).mean().item())
    assert math.sqrt(b["low"] ** 2 + b["mid"] ** 2 + b["high"] ** 2) == pytest.approx(
        total, rel=1e-9
    )


def test_error_bands_puts_a_pure_sinusoid_in_the_right_band() -> None:
    n = 64
    x = torch.arange(n, dtype=D)
    mask = torch.ones(1, n, n, dtype=torch.bool)
    for cycles, band in ((1, "low"), (6, "mid"), (24, "high")):  # cycles/pixel = cycles/64
        err = torch.sin(2 * math.pi * cycles * x / n)[None, :].expand(n, n)[None]
        b = V.error_bands(err.contiguous(), mask)
        assert b[band] == pytest.approx(math.sqrt(0.5), rel=1e-6)
        assert sum(v for k, v in b.items() if k != band) < 1e-6
