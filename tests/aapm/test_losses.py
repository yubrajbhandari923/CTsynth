"""Loss terms, ablation, and the F-018/F-019 decoy test (headline)."""

import math

import numpy as np
import pytest
import torch

from ctsynth.aapm import losses as L
from ctsynth.aapm.torch_forward import AapmTorchForward

D = torch.float64


@pytest.fixture
def fwd(tables):
    return AapmTorchForward(tables, D)


def np_T(tables, p):
    return np.array(
        [(0.5 * tables[k].spectrum * np.exp(-(tables[k].mu_per_cm.T @ p))).sum() for k in (50, 80)]
    )


def decoy(tables, p, c):
    """Independent numpy Newton: hold calcification at c, re-solve adipose/fibro to match T(p)."""
    x = p[:2].copy()
    for _ in range(100):
        q = np.array([x[0], x[1], c])
        r = np_T(tables, q) - np_T(tables, p)
        if np.abs(r).max() < 1e-17:
            break
        jac = np.array(
            [
                [
                    -(
                        0.5
                        * tables[k].spectrum
                        * tables[k].mu_per_cm[m]
                        * np.exp(-(tables[k].mu_per_cm.T @ q))
                    ).sum()
                    for m in range(2)
                ]
                for k in (50, 80)
            ]
        )
        x = x - np.linalg.solve(jac, r[:2])
    return np.array([x[0], x[1], c])


def t(a):
    return torch.as_tensor(a, dtype=D)


def test_f018_truth_vs_decoy_headline(tables, fwd, capsys):
    truth = np.array([8.0, 4.0, 0.05])
    dec = decoy(tables, truth, 0.10)
    assert np.linalg.norm(dec - truth) > 4.0  # F-018: ||dx|| ~ 6.5 cm; material error is large
    meas = fwd.path_forward(t(truth).reshape(1, 3))
    l_true = L.consistency_transmission(t(truth).reshape(1, 3), meas, fwd).item()
    l_dec = L.consistency_transmission(t(dec).reshape(1, 3), meas, fwd).item()
    with capsys.disabled():
        print(
            f"\nHEADLINE truth={l_true:.3e} decoy={l_dec:.3e}  dx={np.linalg.norm(dec - truth):.3f}"
        )
    # T ~ 0.04-0.07, rel. error 1.8e-16 (F-018) => squared err < 1e-30; 1e-28 leaves slack
    assert l_true == 0.0 and l_dec < 1e-28
    # nothing in the data term prefers truth; the decoy even violates nonnegativity (f=-0.51), but
    # that is the nonneg term, not consistency
    assert L.nonneg(t(dec)).item() > 0 and L.nonneg(t(truth)).item() == 0


def test_nonneg_decoy_not_separated_by_nonneg_but_by_sum_to_one(tables, fwd):
    # F-019 style tissue pixel (sum 1, 1 cm path so fractions == path integrals), small step
    # keeps the decoy fully nonnegative
    truth = np.array([0.6, 0.39, 0.01])
    dec = decoy(tables, truth, 0.013)
    assert (dec >= 0).all() and np.abs(dec - truth).max() > 0.1
    xt, xd = t(truth).reshape(1, 3, 1, 1), t(dec).reshape(1, 3, 1, 1)
    meas = fwd.path_forward(t(truth).reshape(1, 3))
    c = [L.consistency_transmission(x.reshape(1, 3), meas, fwd).item() for x in (xt, xd)]
    assert max(c) < 1e-28  # same bound as above
    assert L.nonneg(xt).item() == L.nonneg(xd).item() == 0.0  # nonneg does NOT separate
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    s_t, s_d = L.sum_to_one(xt, mask).item(), L.sum_to_one(xd, mask).item()
    assert s_t < 1e-28 and s_d > 1e-6  # sum = 1.0 vs ~1.015 => (0.015)^2 ~ 2e-4


def test_image_domain_null_direction(tables, fwd):
    a = fwd.A.numpy()
    null = np.linalg.svd(a)[2][-1]
    x = np.array([0.6, 0.39, 0.01])
    meas = fwd.image_forward(t(x).reshape(1, 3, 1, 1))
    xd = t(x + 0.5 * null).reshape(1, 3, 1, 1)
    assert L.consistency_image(xd, meas, fwd).item() < 1e-28  # exact null of the 2x3 matrix
    assert L.sum_to_one(xd).item() > 1e-6  # null direction changes the sum (F-019)


def test_each_term_zero_at_optimum_positive_otherwise_gradient_sign(fwd):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, 4, 4, generator=g, dtype=D) * 0.3
    x = x / x.sum(1, keepdim=True)
    meas = fwd.image_forward(x)
    cases = {
        "consistency_image": ((x, meas, fwd), (x + 0.1, meas, fwd)),
        "nonneg": ((x,), (x - 0.5,)),
        "sum_to_one": ((x,), (x * 1.2,)),
        "tv": ((torch.ones(1, 3, 4, 4, dtype=D),), (x,)),
        "entropy": ((torch.eye(3, dtype=D).reshape(1, 3, 3, 1).expand(1, 3, 3, 3),), (x,)),
    }
    for name, (opt, off) in cases.items():
        f = L.REGISTRY[name]
        assert f(*opt).item() == pytest.approx(0.0, abs=1e-6), name  # entropy eps ~1e-8 floor
        assert f(*off).item() > 1e-6, name
    # gradient sign: descent step lowers each loss
    for name in ("consistency_image", "nonneg", "sum_to_one", "tv"):
        a = list(cases[name][1])
        v = a[0].clone().requires_grad_()
        a[0] = v
        L.REGISTRY[name](*a).backward()
        a[0] = (v - 1e-3 * v.grad).detach()
        assert L.REGISTRY[name](*a).item() < L.REGISTRY[name](*cases[name][1]).item(), name


def test_sum_to_one_mask(fwd):
    x = torch.zeros(1, 3, 2, 2, dtype=D)  # air everywhere: sum 0
    x[:, 0, 0, :] = 1.0  # top row tissue
    mask = torch.tensor([[[True, True], [False, False]]])
    assert L.sum_to_one(x, mask).item() == 0.0
    assert L.sum_to_one(x).item() == pytest.approx(0.5)


def test_entropy_hypothesis_testable():
    one = torch.tensor([1.0, 0, 0], dtype=D).reshape(1, 3, 1, 1)
    uni = torch.full((1, 3, 1, 1), 1 / 3, dtype=D)
    mid = torch.tensor([0.7, 0.2, 0.1], dtype=D).reshape(1, 3, 1, 1)
    e = [L.entropy(v).item() for v in (one, mid, uni)]
    assert e[0] < e[1] < e[2]
    assert e[0] == pytest.approx(0.0, abs=1e-6)
    assert e[2] == pytest.approx(math.log(3), abs=1e-6)  # eps=1e-8 perturbs ~1e-8


def test_weight_zero_ablates_exactly(fwd):
    x = (torch.rand(1, 3, 4, 4, dtype=D) - 0.2).requires_grad_()
    meas = torch.zeros(1, 2, 4, 4, dtype=D)
    args = {"consistency_image": (x, meas, fwd), "tv": (x,), "entropy": (x,), "nonneg": (x,)}
    full, _ = L.compose({"consistency_image": 1.0, "tv": 0.5, "entropy": 0.0, "nonneg": 0.0}, args)
    g1 = torch.autograd.grad(full, x)[0]
    only, parts = L.compose({"consistency_image": 1.0, "tv": 0.5}, args)
    g2 = torch.autograd.grad(only, x)[0]
    assert torch.equal(g1, g2) and set(parts) == {"consistency_image", "tv"}
    tv_only, _ = L.compose({"tv": 0.5}, args)
    ref = torch.autograd.grad(0.5 * L.tv(x), x)[0]
    assert torch.equal(torch.autograd.grad(tv_only, x)[0], ref)
