"""Torch AAPM forward vs an independent numpy implementation, plus gradcheck."""

import numpy as np
import pytest
import torch

from ctsynth.aapm.torch_forward import AapmTorchForward


def np_transmission(tables, paths):  # paths (3,)
    return np.array(
        [
            (0.5 * tables[k].spectrum * np.exp(-(tables[k].mu_per_cm.T @ paths))).sum()
            for k in (50, 80)
        ]
    )


@pytest.fixture
def fwd(tables):
    return AapmTorchForward(tables, torch.float64)


def test_path_matches_numpy(tables, fwd):
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 6, size=(2, 3, 4))
    out = fwd.path_forward(torch.as_tensor(p)).numpy()
    ref = np.array([[np_transmission(tables, p[n, :, j]) for j in range(4)] for n in range(2)])
    # 1e-12: same float64 sums, only summation order differs
    np.testing.assert_allclose(out, ref.transpose(0, 2, 1), rtol=1e-12)


def test_image_matches_numpy(tables, fwd):
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(2, 3, 5, 5))
    a = np.stack([(0.5 * tables[k].spectrum * tables[k].mu_per_cm).sum(1) for k in (50, 80)])
    ref = np.einsum("km,nmhw->nkhw", a, x)
    np.testing.assert_allclose(fwd.image_forward(torch.as_tensor(x)).numpy(), ref, rtol=1e-12)


def test_gradcheck_both_modes(fwd):
    p = torch.rand(2, 3, 3, dtype=torch.float64) * 3 + 0.1
    x = torch.rand(1, 3, 3, 3, dtype=torch.float64)
    p.requires_grad_(), x.requires_grad_()
    # eps 1e-6 / atol 1e-5 are torch's float64 defaults, adequate for smooth exp / linear maps
    assert torch.autograd.gradcheck(fwd.path_forward, (p,), eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(fwd.image_forward, (x,), eps=1e-6, atol=1e-5)


def test_transmission_range(fwd):
    t = fwd.path_forward(torch.zeros(1, 3, 2, dtype=torch.float64))
    torch.testing.assert_close(t, torch.ones_like(t), rtol=1e-4, atol=1e-4)  # sum 0.5 s = 1 (1e-4)


def test_exp_clip_prevents_overflow_for_negative_paths(tables):
    fwd = AapmTorchForward(tables, torch.float32)
    p = torch.full((1, 3, 2), -5.0)
    assert not torch.isfinite(fwd.path_forward(p)).all()
    assert torch.isfinite(fwd.path_forward(p, exp_clip=20.0)).all()
    ok = torch.rand(1, 3, 2)
    assert torch.equal(fwd.path_forward(ok), fwd.path_forward(ok, exp_clip=1e9))
