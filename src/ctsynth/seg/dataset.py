"""Class-balanced random 3D patches from cached CT volumes (HU) and organ-group label volumes.

Records point at cache files written by ctsynth.xcat.cache; label volumes are uint8 .npy shared by
all CT variants of a patient.
`class_samples` holds up to a few thousand voxel coordinates per class so balanced sampling needs
no scan of the volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from ctsynth.seeding import rng
from ctsynth.seg.unet3d import normalise_hu
from ctsynth.xcat.cache import VolumeReader


@dataclass
class VolumeRecord:
    patient_id: str
    ct_path: str
    ct_fmt: str
    label_path: str
    cond: tuple[float, ...] = ()  # acquisition metadata for the conditioned arm
    class_samples: dict[int, NDArray[np.int32]] = field(
        default_factory=dict
    )  # class -> (n, 3) voxel coordinates


def class_voxel_samples(
    labels: NDArray[np.uint8], n_per_class: int = 4000, seed: int = 0
) -> dict[int, NDArray[np.int32]]:
    r = rng(seed, 7)
    out = {}
    for k in np.unique(labels):
        idx = np.flatnonzero(labels == k)
        pick = r.choice(idx, size=min(n_per_class, idx.size), replace=False)
        out[int(k)] = np.stack(np.unravel_index(pick, labels.shape), axis=1).astype(np.int32)
    return out


class PatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Item = (CT patch (1,D,D,D) normalised, labels (D,D,D) int64, cond vector). Deterministic in
    (seed, epoch, index)."""

    def __init__(
        self,
        records: list[VolumeRecord],
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
        if (
            i not in self._readers
        ):  # opened lazily in each dataloader worker (memory-mapped / lazy zarr)
            rec = self.records[i]
            self._readers[i] = (
                VolumeReader(rec.ct_path, rec.ct_fmt),
                np.load(rec.label_path, mmap_mode="r"),
            )
        return self._readers[i]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = rng(self.seed, self.epoch, idx)
        ri = int(r.integers(len(self.records)))
        rec = self.records[ri]
        ct, lab = self._open(ri)
        shape = ct.shape
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
            for c, s in zip(centre, shape, strict=True)
        ]
        origin = (o[0], o[1], o[2])
        sl = tuple(slice(v, v + self.patch) for v in origin)
        x = torch.from_numpy(ct.read_patch(origin, (self.patch,) * 3))[None]
        y = torch.from_numpy(np.asarray(lab[sl]).astype(np.int64))
        return normalise_hu(x), y, torch.tensor(rec.cond, dtype=torch.float32)
