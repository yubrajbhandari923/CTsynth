"""EXP-036 amendment 3: dropping zero-weight energies is exact and removes the 0*inf NaN."""

from pathlib import Path

import torch

from ctsynth.aapm import schema as S
from ctsynth.aapm.solve import drop_zero_weight
from ctsynth.aapm.torch_forward import AapmTorchForward

TAB = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"


def _tables():
    return {k: S.load_model_table(TAB / f"model_data_{k}kVp.npy", k) for k in S.KVPS}


def test_identical_on_physical_paths_and_finite_on_negative_paths():
    full = AapmTorchForward(_tables(), dtype=torch.float32)
    cut = AapmTorchForward(
        {k: drop_zero_weight(t) for k, t in _tables().items()}, dtype=torch.float32
    )
    pos = torch.rand(1, 3, 50, dtype=torch.float32) * 5
    assert torch.allclose(full.path_forward(pos), cut.path_forward(pos), rtol=0, atol=1e-7)
    neg = torch.full((1, 3, 4), -0.1, dtype=torch.float32)  # overflows the 1e3 cm^-1 rows in `full`
    assert not torch.isfinite(full.path_forward(neg)).all()
    assert torch.isfinite(cut.path_forward(neg)).all()
