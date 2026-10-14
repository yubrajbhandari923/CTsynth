"""Fan-beam polychromatic forward: shapes, parity, agreement with the numpy reference chain."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ctsynth.aapm.fanbeam import fanbeam_project
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.schema import GeometrySpec, load_model_table

G = GeometrySpec(n_pix=16, fov_cm=18.0, n_views=16, n_det=24)


def test_empty_object_is_unit_transmission(tables):
    t = AapmFanbeamForward(tables, G, torch.float64)(torch.zeros(1, 3, 16, 16, dtype=torch.float64))
    for k in (50, 80):
        assert t[k].shape == (1, 8, 24) and torch.allclose(
            t[k], torch.ones_like(t[k]), atol=1e-8
        )  # sum(0.5 s) = 1 + 7e-9


def test_matches_numpy_chain_and_parity(tables):
    x = torch.tensor(np.random.default_rng(0).random((1, 3, 16, 16)) / 3)
    t = AapmFanbeamForward(tables, G, torch.float64, view_chunk=3)(x)
    p = fanbeam_project(x[0], G).numpy()  # (3, V, D) path lengths for all views
    for kvp, sl in ((80, slice(0, None, 2)), (50, slice(1, None, 2))):
        tb = tables[kvp]
        pk = p[:, sl]
        ref = (
            0.5 * tb.spectrum[:, None, None] * np.exp(-np.einsum("me,mvd->evd", tb.mu_per_cm, pk))
        ).sum(0)
        np.testing.assert_allclose(t[kvp][0].numpy(), ref, rtol=1e-12)
    assert not torch.allclose(t[50], t[80])  # channel swap is detectable


def test_gradient_flows_to_all_materials(tables):
    x = torch.full((1, 3, 16, 16), 0.2, dtype=torch.float64, requires_grad=True)
    t = AapmFanbeamForward(tables, G, torch.float64)(x)
    (t[50].log().sum() + t[80].log().sum()).backward()
    g = x.grad
    assert g is not None and torch.isfinite(g).all() and (g.abs().sum(dim=(0, 2, 3)) > 0).all()


KIT = os.environ.get("CTSYNTH_AAPM_KIT")


@pytest.mark.skipif(not KIT, reason="set CTSYNTH_AAPM_KIT to an extracted starting_kit/ directory")
def test_reproduces_kit_transmission():
    k = Path(KIT or "")
    names = ("Adipose", "Fibroglandular", "Calcification")
    x = torch.tensor(np.stack([np.load(k / f"Phantom_{m}.npy")[3] for m in names]))[None]
    tables = {v: load_model_table(k / "model" / f"model_data_{v}kVp.npy", v) for v in (50, 80)}
    fwd = AapmFanbeamForward(tables)
    for kvp, stem, sl in ((80, "high", slice(0, 256, 32)), (50, "low", slice(0, 256, 32))):
        ref = np.load(k / f"{stem}kVpTransmission.npy")[3][sl]
        views = torch.arange(0 if kvp == 80 else 1, 512, 2)[sl]
        p = fanbeam_project(x[0], views=views)[None]
        got = fwd.spectral.path_forward(p, (0 if kvp == 50 else 1,))[0, 0].numpy()
        assert np.abs(got - ref).max() < 1e-5


def test_view_subset_matches_full_and_loss_zero_on_truth(tables):
    from ctsynth.aapm.physics_loss import projection_loss, stack_kvp

    fwd = AapmFanbeamForward(tables, G, torch.float64)
    x = torch.tensor(np.random.default_rng(3).random((1, 3, 16, 16)) / 3)
    full, sub = fwd(x), torch.tensor([1, 4, 6])
    part = fwd(x, sub)
    for k in (50, 80):
        assert torch.equal(part[k], full[k][:, sub])
    ref = stack_kvp(full)
    assert projection_loss(fwd, x, ref, sub).item() == 0.0
    assert projection_loss(fwd, x * 0.9, ref, sub, log=True).item() > 0
