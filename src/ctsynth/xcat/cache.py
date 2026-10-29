"""Compact training-array cache for CT volumes (HU) and label volumes. Never compressed NIfTI.

HU formats: `npy_f32` (reference), `npy_f16`, `npy_i16q` (int16 = round(HU / 0.25), clipped to
+-8191 HU; max quantisation error 0.125 HU,
SD 0.072 HU), and chunked `zarr_*` variants of the int16 form (needs zarr). Labels: uint8 `.npy`.
Readers return float32 HU.
Random 3D patches are read with `read_patch`; `.npy` files are memory-mapped (page cache), zarr
reads whole chunks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

HU_STEP = 0.25
HU_LIMIT = 32767 * HU_STEP
FORMATS = ("npy_f32", "npy_f16", "npy_i16q", "zarr_raw_i16q", "zarr_lz4_i16q", "zarr_zstd1_i16q")


def hu_to_i16(hu: NDArray[np.float32]) -> NDArray[np.int16]:

    return np.round(np.clip(hu, -HU_LIMIT, HU_LIMIT) / HU_STEP).astype(np.int16)


def i16_to_hu(a: NDArray[np.int16]) -> NDArray[np.float32]:
    return (a.astype(np.float32) * HU_STEP).astype(np.float32)


@dataclass
class WriteResult:
    path: str
    fmt: str
    bytes: int
    seconds: float


def _dir_bytes(p: Path) -> int:
    return (
        sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        if p.is_dir()
        else p.stat().st_size
    )


def write_hu(
    path: str | Path, hu: NDArray[np.float32], fmt: str, chunks: tuple[int, int, int] = (64, 64, 64)
) -> WriteResult:
    """Write a HU volume [z,y,x]; `path` gets the right suffix (`.npy` or `.zarr`)."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt}; choose from {FORMATS}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if fmt.startswith("npy"):
        p = p.with_suffix(".npy")
        arr = (
            hu.astype(np.float32)
            if fmt == "npy_f32"
            else hu.astype(np.float16)
            if fmt == "npy_f16"
            else hu_to_i16(hu)
        )
        np.save(p, arr)
    else:
        import zarr

        p = p.with_suffix(".zarr")
        comp: Any = None
        if fmt != "zarr_raw_i16q":
            import numcodecs

            comp = numcodecs.Blosc(
                cname="lz4" if "lz4" in fmt else "zstd",
                clevel=1 if "zstd" in fmt else 5,
                shuffle=numcodecs.Blosc.SHUFFLE,
            )
        z = zarr.open(str(p), mode="w", shape=hu.shape, chunks=chunks, dtype="i2", compressor=comp)
        z[:] = hu_to_i16(hu)
    return WriteResult(str(p), fmt, _dir_bytes(p), time.perf_counter() - t0)


class VolumeReader:
    """Lazy reader returning float32 HU for slices/patches. `fmt` is inferred from the dtype/suffix
    and the sidecar-free layout."""

    def __init__(self, path: str | Path, fmt: str) -> None:
        self.path, self.fmt = Path(path), fmt
        if fmt.startswith("npy"):
            self._a: Any = np.load(self.path, mmap_mode="r")
        else:
            import zarr

            self._a = zarr.open(str(self.path), mode="r")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._a.shape)

    def read_patch(
        self, origin: tuple[int, int, int], size: tuple[int, int, int]
    ) -> NDArray[np.float32]:
        sl = tuple(slice(o, o + s) for o, s in zip(origin, size, strict=True))
        raw = np.asarray(self._a[sl])

        return i16_to_hu(raw) if raw.dtype == np.int16 else raw.astype(np.float32)

    def read_all(self) -> NDArray[np.float32]:
        return self.read_patch((0, 0, 0), self.shape)  # type: ignore[arg-type]


def write_meta(path: str | Path, meta: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(meta, indent=1, sort_keys=True, default=str))
