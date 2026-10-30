"""Physical-phantom U-Net: CT (HU) -> nonnegative composition concentrations c_k (g/cm^3).

Reuses `UNet3D` (same backbone, optional FiLM conditioning) with a softplus head, so c >= 0 by
construction. Loss (weights fixed a priori, nothing tuned): per-channel Huber on c (divided by
delta so large errors have unit slope), plus L1 on total density (g/cm^3) and L1 on mu70 in
water units (= |dHU|/1000), where mu70 = sum_k c_k kappa_k(70 keV) is computed from the
prediction. No Dice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ctsynth.seeding import rng
from ctsynth.seg.infer import _starts
from ctsynth.seg.unet3d import UNet3D, normalise_hu
from ctsynth.xcat.cache import VolumeReader
from ctsynth.xcat.physical import u16_to_conc

HUBER_DELTA = 0.02  # g/cm^3


class PhysUNet(torch.nn.Module):
    def __init__(self, k: int = 8, base: int = 16, depth: int = 3, cond_dim: int = 0) -> None:
        super().__init__()
        self.net = UNet3D(1, k, base, depth, cond_dim)
        assert self.net.head.bias is not None
        torch.nn.init.constant_(self.net.head.bias, -3.0)  # softplus(-3) ~ 0.05 g/cm^3 at start

    @property
    def k(self) -> int:
        return int(self.net.head.out_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return F.softplus(self.net(x, cond).float())


def phys_loss(
    pred: torch.Tensor, target: torch.Tensor, kappa70: torch.Tensor, mu_water_cm: float
) -> dict[str, torch.Tensor]:
    """pred/target (B, K, D, H, W) g/cm^3; kappa70 (K,) cm^2/g; mu_water_cm = mu_water(70 keV) in
    cm^-1."""
    hub = F.huber_loss(pred, target, delta=HUBER_DELTA, reduction="none") / HUBER_DELTA
    l_c = hub.mean((0, 2, 3, 4)).mean()  # mean over channels of per-channel voxel means
    l_rho = (pred.sum(1) - target.sum(1)).abs().mean()
    k = kappa70.to(pred).reshape(1, -1, 1, 1, 1)
    l_mu = (((pred - target) * k).sum(1)).abs().mean() / mu_water_cm
    return {"total": l_c + l_rho + l_mu, "conc": l_c, "density": l_rho, "mu70": l_mu}


@dataclass
class PhysRecord:
    patient_id: str
    ct_path: str
    ct_fmt: str
    conc_path: str  # (K, nz, ny, nx) uint16 .npy
    cond: tuple[float, ...] = ()
    class_samples: dict[int, NDArray[np.int32]] = field(default_factory=dict)  # voxel centres


class PhysPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Item = (CT patch (1,D,D,D) normalised, concentrations (K,D,D,D) g/cm^3, cond). Patch
    centres follow the EXP-039 sampler (class_samples of the organ-group labels, 70 % foreground);
    deterministic in (seed, epoch, index)."""

    def __init__(
        self,
        records: list[PhysRecord],
        patch: int,
        length: int,
        seed: int = 0,
        p_foreground: float = 0.7,
    ) -> None:
        self.records, self.patch, self.length, self.seed, self.p_fg = (
            records,
            patch,
            length,
            seed,
            p_foreground,
        )
        self.epoch = 0
        self._readers: dict[int, tuple[VolumeReader, Any]] = {}

    def __len__(self) -> int:
        return self.length

    def _open(self, i: int) -> tuple[VolumeReader, Any]:
        if i not in self._readers:
            rec = self.records[i]
            self._readers[i] = (
                VolumeReader(rec.ct_path, rec.ct_fmt),
                np.load(rec.conc_path, mmap_mode="r"),
            )
        return self._readers[i]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = rng(self.seed, self.epoch, idx)
        ri = int(r.integers(len(self.records)))
        rec = self.records[ri]
        ct, conc = self._open(ri)
        classes = sorted(rec.class_samples)
        fg = [k for k in classes if k != 0] or classes
        k = int(r.choice(fg if r.random() < self.p_fg else classes))
        centre = rec.class_samples[k][int(r.integers(len(rec.class_samples[k])))]
        o = [
            int(
                np.clip(
                    c - self.patch // 2 + r.integers(-self.patch // 4, self.patch // 4 + 1),
                    0,
                    s - self.patch,
                )
            )
            for c, s in zip(centre, ct.shape, strict=True)
        ]
        sl = tuple(slice(v, v + self.patch) for v in o)
        x = torch.from_numpy(ct.read_patch((o[0], o[1], o[2]), (self.patch,) * 3))[None]
        y = torch.from_numpy(u16_to_conc(np.asarray(conc[(slice(None), *sl)])))
        return normalise_hu(x), y, torch.tensor(rec.cond, dtype=torch.float32)


@torch.no_grad()
def predict_conc(
    net: PhysUNet,
    hu: NDArray[np.float32],
    patch: int = 96,
    stride: int = 64,
    batch: int = 4,
    cond: torch.Tensor | None = None,
    device: torch.device | str = "cuda",
    amp: bool = True,
) -> torch.Tensor:
    """(K, nz, ny, nx) float32 concentrations on `device`, mean over overlapping windows."""
    if min(hu.shape) < patch:
        raise ValueError(f"volume {hu.shape} smaller than the patch {patch}")
    net.eval()
    acc = torch.zeros((net.k, *hu.shape), dtype=torch.float32, device=device)
    cnt = torch.zeros(hu.shape, dtype=torch.float32, device=device)
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
            out = net(xb, None if cond is None else cond.to(device).expand(len(chunk), -1))
        for (z, y, x), p in zip(chunk, out, strict=True):
            acc[:, z : z + patch, y : y + patch, x : x + patch] += p
            cnt[z : z + patch, y : y + patch, x : x + patch] += 1
    return acc / cnt.clamp(min=1)
