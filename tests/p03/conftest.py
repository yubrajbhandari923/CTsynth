"""Fixtures for the EXP-004 tests: the committed kit tables as ModelTable objects."""

from pathlib import Path

import numpy as np
import pytest

from ctsynth.aapm.schema import ModelTable

TABLE_DIR = Path(__file__).resolve().parents[2] / "data" / "aapm_kit_sample"


@pytest.fixture(scope="session")
def tables_np() -> dict[int, ModelTable]:
    out = {}
    for kvp in (50, 80):
        raw = np.load(TABLE_DIR / f"model_data_{kvp}kVp.npy").astype(np.float64)
        out[kvp] = ModelTable(kvp, raw[0], raw[1], raw[2:5])
    return out
