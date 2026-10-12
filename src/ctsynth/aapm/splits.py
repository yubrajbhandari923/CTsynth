"""Deterministic by-case train/val/test splits with a held-out guard on the test split.

Stability property (frozen): a case's split depends ONLY on (seed, case_index, fractions) via
u = ctsynth.seeding.rng(seed, case_index).random(); train if u < f_train, val if
u < f_train + f_val, else test. Hence the assignment of case i is identical for every n_cases > i
(prefix-stable
under changing N) and across re-runs. Sizes are binomial, not exact.
"""

from __future__ import annotations

from typing import Literal

from ctsynth.aapm.schema import Array

SplitName = Literal["train", "val", "test"]
DEFAULT_FRACTIONS: tuple[float, float, float] = (0.8, 0.1, 0.1)


class TestSplitGuardError(PermissionError):
    """Raised when the held-out test split is requested without purpose='final_evaluation'."""

    __test__ = False  # not a pytest class


def case_split(
    case_index: int, seed: int, fractions: tuple[float, float, float] = DEFAULT_FRACTIONS
) -> SplitName:
    """Split name of one case by the frozen rule above. Names only; no guard."""
    raise NotImplementedError


def split_indices(
    n_cases: int,
    seed: int,
    split: SplitName,
    *,
    purpose: str = "development",
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> Array:
    """Sorted int64 case indices in [0, n_cases) assigned to `split`.

    split == 'test' raises TestSplitGuardError unless purpose == 'final_evaluation'.
    Fractions must be non-negative and sum to 1 (ValueError). Splits partition
    range(n_cases): disjoint, exhaustive.
    """
    raise NotImplementedError
