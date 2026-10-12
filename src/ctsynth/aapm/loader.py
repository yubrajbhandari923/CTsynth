"""Streaming per-case access. Peak memory is O(one case): every file is opened with
numpy.load(..., mmap_mode='r') and only the requested case's slices are copied out. The full arrays
are never read into RAM (OS page cache aside).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ctsynth.aapm import schema as S
from ctsynth.aapm.schema import Array, GeometrySpec

# file stem -> (kind, kVp or None); kinds: phantom material j, transmission, image
_FILES: dict[str, str] = {
    "Phantom_Adipose": "phantom",
    "Phantom_Fibroglandular": "phantom",
    "Phantom_Calcification": "phantom",
    "lowkVpTransmission": "transmission",
    "highkVpTransmission": "transmission",
    "lowkVpImages": "image",
    "highkVpImages": "image",
}
_KVP_OF_STEM = {"low": 50, "high": 80}


@dataclass(frozen=True)
class Case:
    """One case, in-RAM copies (not views of the memmaps)."""

    index: int
    phantom: Array  # (3, n, n) float32, canonical MATERIALS order, [material, ix, iy]
    transmission: dict[int, Array]  # {50: (views//2, bins), 80: (...)} float32
    image: dict[int, Array]  # {50: (n, n), 80: (n, n)} float32, image[ix, iy]


class CaseStore:
    """Lazy case store over an uncompressed .npy directory (layout: adapter-spec section 6).

    Opening validates header shapes/dtypes against `geometry` (SchemaError) but reads no pixel data.
    `.gz` files must be decompressed beforehand; a directory holding only .gz raises
    FileNotFoundError naming the missing .npy.
    """

    def __init__(self, root: str | Path, geometry: GeometrySpec | None = None) -> None:
        """geometry defaults to schema.REAL_GEOMETRY."""
        self.root = Path(root)
        self.geometry = geometry or S.REAL_GEOMETRY
        self._arrays: dict[str, np.memmap] = {}
        for stem in _FILES:
            path = self.root / f"{stem}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path.name} missing in {self.root} (decompress .gz first)"
                )
            self._arrays[stem] = np.load(path, mmap_mode="r")
        self._check_headers()

    def _check_headers(self) -> None:
        g = self.geometry
        n, v2 = g.n_pix, g.n_views // 2
        expected = {"phantom": (n, n), "image": (n, n), "transmission": (v2, g.n_det)}
        counts = set()
        for stem, kind in _FILES.items():
            arr = self._arrays[stem]
            if arr.ndim != 3 or arr.shape[1:] != expected[kind]:
                raise S.SchemaError(
                    S.CONV_SHAPE, f"{stem}.npy {arr.shape}, expected (N, {expected[kind]})"
                )
            if arr.dtype != S.DTYPE:
                raise S.SchemaError(S.CONV_DTYPE, f"{stem}.npy dtype {arr.dtype}, expected float32")
            counts.add(arr.shape[0])
        if len(counts) != 1:
            raise S.SchemaError(S.CONV_SHAPE, f"case counts differ across files: {sorted(counts)}")

    @property
    def arrays(self) -> dict[str, np.memmap]:
        """The open read-only memmaps, keyed by file stem (e.g. 'Phantom_Adipose')."""
        return self._arrays

    def __len__(self) -> int:
        """Number of cases N."""
        return int(self._arrays["Phantom_Adipose"].shape[0])

    def __getitem__(self, index: int) -> Case:
        """Copy out case `index` (negative indices rejected with IndexError). Validates the phantom
        with schema.validate_phantoms (so on-disk material files swapped by name are detected).
        """
        if not 0 <= index < len(self):
            raise IndexError(f"case index {index} out of range [0, {len(self)})")
        a = self._arrays
        phantom = np.stack(
            [
                np.array(a[s][index], dtype=S.DTYPE)
                for s in ("Phantom_Adipose", "Phantom_Fibroglandular", "Phantom_Calcification")
            ]
        )
        S.validate_phantoms(phantom[None], self.geometry, material_names=S.MATERIALS)
        transmission = {
            kvp: np.array(a[f"{p}kVpTransmission"][index]) for p, kvp in _KVP_OF_STEM.items()
        }
        image = {kvp: np.array(a[f"{p}kVpImages"][index]) for p, kvp in _KVP_OF_STEM.items()}
        return Case(index, phantom, transmission, image)

    def iter_cases(self, indices: Iterable[int] | None = None) -> Iterator[Case]:
        """Yield cases lazily, one at a time, in `indices` order (default: all, ascending)."""
        for i in range(len(self)) if indices is None else indices:
            yield self[i]
