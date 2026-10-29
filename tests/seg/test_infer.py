"""Sliding-window inference covers every voxel and reproduces a deterministic per-voxel rule
(identity-like network)."""

import numpy as np
import pytest
import torch

from ctsynth.seg.infer import _starts, predict_volume
from ctsynth.seg.unet3d import UNet3D


def test_window_starts_cover_the_axis():
    for n, p, s in [(10, 4, 3), (11, 4, 4), (4, 4, 2), (3, 4, 2)]:
        st = _starts(n, p, s)
        assert st[0] == 0 and (st[-1] + p >= n or n <= p)


class Thresh(UNet3D):
    """Stub with the UNet interface: class 1 where the normalised input exceeds 0, else class 0
    (checks tiling logic only)."""

    def __init__(self) -> None:
        super().__init__(1, 2, base=2, depth=1)

    def forward(self, x, cond=None):  # type: ignore[no-untyped-def]
        return torch.cat([-x * 50, x * 50], dim=1)


def test_prediction_matches_voxelwise_rule_on_asymmetric_volume():
    hu = np.full((20, 26, 30), -500.0, np.float32)
    hu[5:12, 8:20, 10:25] = 400.0
    pred = predict_volume(Thresh(), hu, patch=8, stride=5, batch=3, device="cpu", amp=False)
    assert pred.shape == hu.shape and np.array_equal(pred, (hu > 0).astype(np.uint8))
    with pytest.raises(ValueError):
        predict_volume(Thresh(), hu[:6], patch=8, device="cpu", amp=False)
