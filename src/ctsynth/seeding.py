"""Deterministic seeding. All randomness flows from one root seed via SeedSequence."""

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy-legacy and (if installed) torch. Prefer `rng()` for new code."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  # legacy global state, for third-party code only
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def rng(seed: int, *stream: int) -> np.random.Generator:
    """Independent, reproducible generator for (seed, stream path); e.g. rng(0, patient_id)."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=stream))
