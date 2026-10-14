"""Fan-beam projector (kit port): invariants, gradients, kit reproduction."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ctsynth.aapm.fanbeam import fanbeam_project
from ctsynth.aapm.schema import GeometrySpec

G = GeometrySpec(n_pix=16, fov_cm=18.0, n_views=16, n_det=24)


def _img(seed: int = 0) -> torch.Tensor:
    return torch.tensor(np.random.default_rng(seed).random((2, 16, 16)))


def test_zero_image_zero_integral() -> None:
    assert fanbeam_project(torch.zeros(1, 16, 16, dtype=torch.float64), G).abs().max() == 0


def test_constant_image_central_chord() -> None:
    # Central ray (u ~ 0) through an all-ones image crosses the 18 cm FOV (views 0, 4).
    p = fanbeam_project(torch.ones(1, 16, 16, dtype=torch.float64), G)
    centre = G.n_det // 2
    for v in (0, 4):
        assert p[0, v, centre - 1 : centre + 1].mean().item() == pytest.approx(18.0, rel=0.03)


def test_chunk_and_view_subset_invariance() -> None:
    x = _img()
    full = fanbeam_project(x, G)
    assert torch.equal(full, fanbeam_project(x, G, ray_chunk=7))
    odd = fanbeam_project(x, G, views=torch.arange(1, 16, 2))
    assert torch.equal(odd, full[:, 1::2])  # parity: odd views = 50 kVp (F-032)
    assert not torch.allclose(full[:, ::2], full[:, 1::2])


def test_gradcheck_finite_difference() -> None:
    x = _img()[:1].clone().requires_grad_(True)
    v = torch.tensor([0, 3, 5])
    assert torch.autograd.gradcheck(lambda t: fanbeam_project(t, G, views=v), (x,), eps=1e-6)


def test_adjoint_consistency() -> None:
    # <A x, y> == <x, A^T y>, with A^T from autograd.
    x = _img()[:1].clone().requires_grad_(True)
    y = torch.tensor(np.random.default_rng(1).random((1, 16, 24)))
    (fanbeam_project(x, G) * y).sum().backward()
    z = _img(2)[:1]
    assert (fanbeam_project(z, G) * y).sum().item() == pytest.approx((z * x.grad).sum().item())


KIT = os.environ.get("CTSYNTH_AAPM_KIT")


@pytest.mark.skipif(not KIT, reason="set CTSYNTH_AAPM_KIT to an extracted starting_kit/ directory")
def test_reproduces_kit_sinogram() -> None:
    k = Path(KIT or "")
    ph = np.load(k / "Phantom_Fibroglandular.npy")[3]
    ref = np.load(k / "model" / "Sinogram_Fibroglandular.npy")[3]
    p = fanbeam_project(torch.tensor(ph)[None], views=torch.arange(0, 512, 32))[0].numpy()
    assert np.abs(p - ref[::32]).max() < 1e-5 * ref.max()  # float32 rounding of the kit sinogram
