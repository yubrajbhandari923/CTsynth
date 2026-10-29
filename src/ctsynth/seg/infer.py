"""Sliding-window whole-volume inference with a patch network; probabilities are averaged over
overlapping windows."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ctsynth.seg.unet3d import UNet3D, normalise_hu


def _starts(n: int, patch: int, stride: int) -> list[int]:
    if n <= patch:
        return [0]
    s = list(range(0, n - patch + 1, stride))
    if s[-1] != n - patch:
        s.append(n - patch)
    return s


@torch.no_grad()
def predict_volume(
    net: UNet3D,
    hu: NDArray[np.float32],
    patch: int = 96,
    stride: int = 64,
    batch: int = 4,
    cond: torch.Tensor | None = None,
    device: torch.device | str = "cuda",
    amp: bool = True,
) -> NDArray[np.uint8]:
    """Argmax class volume for a HU volume [z,y,x]; every dimension must be >= patch (pad
    beforehand otherwise)."""
    if min(hu.shape) < patch:
        raise ValueError(f"volume {hu.shape} smaller than the patch {patch}")
    net.eval()
    n_classes = net.head.out_channels
    prob = torch.zeros((n_classes, *hu.shape), dtype=torch.float16, device=device)
    cnt = torch.zeros(hu.shape, dtype=torch.float16, device=device)
    x_all = torch.from_numpy(hu).to(device)
    origins = [
        (z, y, x)
        for z in _starts(hu.shape[0], patch, stride)
        for y in _starts(hu.shape[1], patch, stride)
        for x in _starts(hu.shape[2], patch, stride)
    ]
    for i in range(0, len(origins), batch):
        chunk = origins[i : i + batch]
        xb = torch.stack(
            [normalise_hu(x_all[z : z + patch, y : y + patch, x : x + patch]) for z, y, x in chunk]
        )[:, None]
        with torch.autocast(str(device).split(":")[0], dtype=torch.bfloat16, enabled=amp):
            lg = net(xb, None if cond is None else cond.to(device).expand(len(chunk), -1))
        pb = torch.softmax(lg.float(), dim=1).half()
        for (z, y, x), p in zip(chunk, pb, strict=True):
            prob[:, z : z + patch, y : y + patch, x : x + patch] += p
            cnt[z : z + patch, y : y + patch, x : x + patch] += 1
    return (prob / cnt.clamp(min=1)).argmax(0).to(torch.uint8).cpu().numpy()
