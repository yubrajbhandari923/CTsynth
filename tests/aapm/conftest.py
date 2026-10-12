"""Fixtures. Implementation objects are built inside tests so missing code shows up as
NotImplementedError failures, not fixture errors. Tests run without the real AAPM case arrays."""

from pathlib import Path

import numpy as np
import pytest

from ctsynth.aapm.schema import ModelTable
from ctsynth.aapm.synthetic import SyntheticConfig

TABLE_DIR = Path(__file__).resolve().parents[2] / "data" / "aapm_kit_sample"
# Real case arrays are never in the repo; optional tests skip unless this directory exists.
REAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "aapm"


@pytest.fixture(scope="session")
def tables() -> dict[int, ModelTable]:
    """Committed model tables, built directly with np.load (independent of load_model_table)."""
    out = {}
    for kvp in (50, 80):
        raw = np.load(TABLE_DIR / f"model_data_{kvp}kVp.npy").astype(np.float64)
        out[kvp] = ModelTable(kvp, raw[0], raw[1], raw[2:5])
    return out


@pytest.fixture
def cfg() -> SyntheticConfig:
    return SyntheticConfig(n_cases=4, n_pix=16, n_views=16, n_det=24, seed=3)
