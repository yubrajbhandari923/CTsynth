"""Poisson projection noise: mean/variance against first-order theory, determinism, zero counts."""

import pytest
import torch

from ctsynth.xcat.noise import apply_poisson_noise, make_generator


def test_variance_and_bias_follow_the_delta_method():
    p = torch.full((40, 16, 64), 1.0)
    n0 = 2e4
    out = apply_poisson_noise(p, n0, make_generator("cpu", 0, 1))
    assert float(out.var()) == pytest.approx(torch.exp(torch.tensor(1.0)).item() / n0, rel=0.05)
    assert float(out.mean() - 1.0) == pytest.approx(
        torch.exp(torch.tensor(1.0)).item() / (2 * n0), abs=1e-4
    )


def test_deterministic_given_seed_and_different_streams_differ():
    p = torch.rand(10, 8, 8) * 3
    a = apply_poisson_noise(p, 1e3, make_generator("cpu", 5, 0))
    b = apply_poisson_noise(p, 1e3, make_generator("cpu", 5, 0))
    c = apply_poisson_noise(p, 1e3, make_generator("cpu", 5, 1))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_zero_counts_are_clamped_and_finite():
    p = torch.full((4, 4, 4), 30.0)  # exp(-30) * N0 ~ 0 counts
    out = apply_poisson_noise(p, 1e3, make_generator("cpu", 0), floor=1.0)
    assert torch.isfinite(out).all() and float(out.max()) == pytest.approx(
        torch.log(torch.tensor(1e3)).item(), rel=1e-6
    )


def test_in_place_and_chunking_do_not_change_values():
    p = torch.rand(30, 4, 4) * 2
    a = apply_poisson_noise(p, 5e3, make_generator("cpu", 2), view_chunk=7)
    q = p.clone()
    b = apply_poisson_noise(q, 5e3, make_generator("cpu", 2), out=q, view_chunk=7)
    assert torch.equal(a, b) and b.data_ptr() == q.data_ptr()
    with pytest.raises(ValueError):
        apply_poisson_noise(p, 0.0, make_generator("cpu", 0))
