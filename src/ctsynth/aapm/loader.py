"""Streaming per-case access. Peak memory is O(one case): every file is opened with
numpy.load(..., mmap_mode='r') and only the requested case's slices are copied out. The full arrays
are never read into RAM (OS page cache aside).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ctsynth.aapm.schema import Array, GeometrySpec


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
        raise NotImplementedError

    @property
    def arrays(self) -> dict[str, np.memmap]:
        """The open read-only memmaps, keyed by file stem (e.g. 'Phantom_Adipose')."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of cases N."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Case:
        """Copy out case `index` (negative indices rejected with IndexError). Validates the phantom
        with schema.validate_phantoms (so on-disk material files swapped by name are detected).
        """
        raise NotImplementedError

    def iter_cases(self, indices: Iterable[int] | None = None) -> Iterator[Case]:
        """Yield cases lazily, one at a time, in `indices` order (default: all, ascending)."""
        raise NotImplementedError
