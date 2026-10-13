"""Deterministic seeding. All randomness flows from one root seed via SeedSequence."""

import numpy as np


def rng(seed: int, *stream: int) -> np.random.Generator:
    """Independent, reproducible generator for (seed, stream path); e.g. rng(0, patient_id)."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=stream))
