"""Shared fixtures. Objects are built inside tests (via physics_reference.build_*) so that missing
implementation shows up as NotImplementedError failures, not fixture errors."""

import pytest
import torch


@pytest.fixture
def gen() -> torch.Generator:
    """Fresh CPU generator with fixed seed 20240607 (no global RNG use in tests)."""
    g = torch.Generator()
    g.manual_seed(20240607)
    return g
