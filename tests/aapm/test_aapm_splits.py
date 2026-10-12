"""By-case deterministic splits and the held-out guard."""

import numpy as np
import pytest

from ctsynth.aapm import splits as P
from ctsynth.seeding import rng

NAMES: tuple[P.SplitName, ...] = ("train", "val", "test")


def idx(n: int, seed: int, name: P.SplitName) -> np.ndarray:
    return P.split_indices(n, seed, name, purpose="final_evaluation")


def test_disjoint_exhaustive_sorted() -> None:
    parts = [idx(500, 7, s) for s in NAMES]
    allc = np.concatenate(parts)
    assert sorted(allc.tolist()) == list(range(500))  # partition of case indices
    assert len(set(allc.tolist())) == 500
    assert all(np.array_equal(p, np.sort(p)) and p.dtype == np.int64 for p in parts)


def test_deterministic_and_seed_sensitive() -> None:
    assert np.array_equal(idx(300, 1, "train"), idx(300, 1, "train"))
    assert not np.array_equal(idx(300, 1, "train"), idx(300, 2, "train"))


def test_frozen_rule_and_case_split() -> None:
    """The documented rule: u = rng(seed, i).random() thresholded by cumulative fractions."""
    f = P.DEFAULT_FRACTIONS
    for i in range(50):
        u = rng(11, i).random()
        want = "train" if u < f[0] else ("val" if u < f[0] + f[1] else "test")
        assert P.case_split(i, 11) == want
        assert i in idx(50, 11, want)  # type: ignore[arg-type]


def test_prefix_stable_under_changing_n() -> None:
    """Case i keeps its split for any larger N (the chosen stability property)."""
    for s in NAMES:
        small, big = idx(100, 5, s), idx(250, 5, s)
        assert np.array_equal(big[big < 100], small)


def test_fractions_approximate() -> None:
    n = 2000
    for s, f in zip(NAMES, P.DEFAULT_FRACTIONS, strict=True):
        # binomial sd <= 0.009 at n=2000; 0.04 is > 4 sd
        assert abs(len(idx(n, 0, s)) / n - f) < 0.04


def test_custom_fractions_and_validation() -> None:
    assert len(idx(50, 0, "val")) >= 0
    only_train = P.split_indices(40, 0, "train", fractions=(1.0, 0.0, 0.0))
    assert only_train.tolist() == list(range(40))
    for bad in ((0.5, 0.5, 0.5), (-0.1, 0.6, 0.5)):
        with pytest.raises(ValueError):
            P.split_indices(10, 0, "train", fractions=bad)


def test_test_split_guard() -> None:
    with pytest.raises(P.TestSplitGuardError):
        P.split_indices(100, 0, "test")
    with pytest.raises(P.TestSplitGuardError):
        P.split_indices(100, 0, "test", purpose="tuning")
    assert len(P.split_indices(100, 0, "test", purpose="final_evaluation")) > 0
    # train / val are unguarded
    P.split_indices(100, 0, "train")
    P.split_indices(100, 0, "val", purpose="tuning")
    assert issubclass(P.TestSplitGuardError, PermissionError)
