"""Torch port of the official FBP image formation: gradients, linearity, kit reproduction."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ctsynth.aapm.fbp import AapmFBP
from ctsynth.aapm.schema import GeometrySpec

G = GeometrySpec(n_pix=16, fov_cm=18.0, n_views=16, n_det=24)


def _t(seed=0):
    return torch.tensor(np.random.default_rng(seed).uniform(0.2, 1.0, (1, 8, 24)))


def test_gradcheck_through_log_filter_backprojection():
    op = AapmFBP(G)
    t = _t().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: op(x, 80), (t,), eps=1e-6)


def test_unit_transmission_gives_zero_image_and_parity_changes_result():
    op = AapmFBP(G)
    assert op(torch.ones(1, 8, 24, dtype=torch.float64), 50).abs().max() == 0
    t = _t()
    assert not torch.allclose(op(t, 50), op(t, 80))


KIT = os.environ.get("CTSYNTH_AAPM_KIT")


@pytest.mark.skipif(not KIT, reason="set CTSYNTH_AAPM_KIT to an extracted starting_kit/ directory")
def test_reproduces_kit_images():
    k = Path(KIT or "")
    op = AapmFBP()
    for kvp, stem in ((80, "high"), (50, "low")):
        t = torch.tensor(np.load(k / f"{stem}kVpTransmission.npy")[3:4]).double()
        ref = np.load(k / f"{stem}kVpImages.npy")[3]
        # kit images are float32; ref range is ~1.4, so 1e-4 is ~7e-5 relative
        assert np.abs(op(t, kvp)[0].numpy() - ref).max() < 1e-4
