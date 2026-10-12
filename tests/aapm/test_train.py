import inspect

import pytest
import torch

from ctsynth.aapm import train as tr
from ctsynth.aapm.torch_forward import AapmTorchForward
from ctsynth.aapm.unet import UNet

F32 = torch.float32


@pytest.fixture
def setup(tables):
    fwd = AapmTorchForward(tables, F32)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, 3, 8, 8, generator=g)
    x = x / x.sum(1, keepdim=True)
    return fwd, fwd.image_forward(x), x


def factory():
    return UNet(1, 4, "softmax")


def run(setup, seed=0, **kw):
    fwd, m, _ = setup
    return tr.train_label_free(
        factory, fwd, m[:6], m[6:], {"consistency_image": 1.0}, epochs=15, lr=1e-2, seed=seed, **kw
    )


def test_loss_decreases(setup):
    _, h = run(setup)
    assert h["train_total"][-1] < 0.5 * h["train_total"][0]
    assert len(h["val_consistency"]) == 15


def test_deterministic(setup):
    m1, h1 = run(setup)
    m2, h2 = run(setup)
    assert h1 == h2
    for a, b in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.equal(a, b)
    _, h3 = run(setup, seed=1)
    assert h3 != h1


def test_label_free_cannot_see_truth(setup):
    params = set(inspect.signature(tr.train_label_free).parameters)
    assert not params & {"truth", "labels", "truth_train", "truth_val", "y", "phantoms"}
    with pytest.raises(TypeError):
        tr.train_label_free(  # type: ignore[call-arg]
            factory, setup[0], setup[1], setup[1], {"consistency_image": 1.0}, truth=setup[2]
        )
    assert "truth_train" in inspect.signature(tr.train_supervised).parameters


def test_early_stopping_and_supervised(setup):
    _, h = run(setup, patience=1)
    assert len(h["val_consistency"]) <= 15
    fwd, m, x = setup
    _, hs = tr.train_supervised(factory, m[:6], x[:6], m[6:], x[6:], epochs=3, seed=0)
    assert len(hs["train_total"]) == 3
