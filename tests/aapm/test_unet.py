import pytest
import torch

from ctsynth.aapm.unet import UNet


def x():
    return torch.randn(2, 2, 8, 8, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("output", ["linear", "softplus", "softmax", "softmax_with_air"])
def test_shape_and_params(output):
    m = UNet(2, 4, output)
    assert m(x()).shape == (2, 3, 8, 8)
    assert sum(p.numel() for p in m.parameters()) < 20_000  # CPU-seconds budget


def test_constraints_exact():
    # 1e-6: float32 softmax sums
    s = UNet(2, 4, "softmax")(x()).sum(1)
    torch.testing.assert_close(s, torch.ones_like(s), atol=1e-6, rtol=0)
    o = UNet(2, 4, "softmax_with_air")(x())
    assert (o >= 0).all() and (o.sum(1) <= 1 + 1e-6).all()
    assert (UNet(2, 4, "softplus")(x()) >= 0).all()
    assert (UNet(2, 4, "linear")(x()) < 0).any()


def test_air_representable():
    m = UNet(1, 2, "softmax_with_air")
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    m.head.bias.data = torch.tensor([-30.0, -30, -30, 30])  # air logit dominates
    assert m(torch.zeros(1, 2, 4, 4)).max() < 1e-12  # exp(-60): true air pixel, all fractions ~0
    m2 = UNet(1, 2, "softmax")  # plain softmax cannot: sums to 1
    assert m2(torch.zeros(1, 2, 4, 4)).sum(1).min() > 0.999


def test_bad_args():
    with pytest.raises(ValueError):
        UNet(output="relu")
    with pytest.raises(ValueError):
        UNet(2, 4)(torch.zeros(1, 2, 6, 6))
