"""T13 device and precision compatibility. CPU always; MPS only if available; CUDA not required."""

import numpy as np
import physics_reference as ref
import pytest
import torch

from ctsynth.physics import SpectralForwardModel

NX, NY = 6, 5


def _model(dtype: torch.dtype) -> SpectralForwardModel:
    p = ref.build_projector(NX, NY, 0.2, (0.0, 0.7, 1.9), 7, 0.2, 0.011, dtype)
    return SpectralForwardModel(ref.build_basis(dtype), ref.build_spectrum(dtype), p)


def _rho(dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    return (0.05 + 0.5 * torch.rand(2, NY, NX, generator=g, dtype=torch.float64)).to(dtype)


def test_t13_cpu_float32_matches_float64() -> None:
    """float32 on CPU vs float64 CPU reference. rtol 1e-4: float32 eps 1.2e-7, exponents <= ~10
    amplify rounding by <= ~10x, plus ~12-term sums -> ~1e-5 expected; 1e-4 is loose but would
    catch real precision bugs (e.g. float16 casts or int overflow)."""
    c64 = _model(torch.float64).expected_counts(_rho(torch.float64))
    c32 = _model(torch.float32).expected_counts(_rho(torch.float32))
    assert c32.dtype == torch.float32 and c32.device.type == "cpu"
    assert np.allclose(c32.numpy(), c64.numpy(), rtol=1e-4)


def test_t13_to_cpu_roundtrip() -> None:
    """model.to(dtype=float32) equals a natively built float32 model (rtol 1e-6)."""
    a = _model(torch.float64).to("cpu", torch.float32).expected_counts(_rho(torch.float32))
    b = _model(torch.float32).expected_counts(_rho(torch.float32))
    assert torch.allclose(a, b, rtol=1e-6, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_t13_mps_float32_matches_cpu_float64() -> None:
    """MPS float32 vs CPU float64. rtol 1e-3: MPS exp/log may use fast-math (~1e-6 each) and
    interpolation of kappa adds a few ulps; 1e-3 is a loose functional-equivalence bound."""
    ref_c = _model(torch.float64).expected_counts(_rho(torch.float64))
    m = _model(torch.float32).to("mps")
    got = m.expected_counts(_rho(torch.float32).to("mps"))
    assert got.device.type == "mps"
    assert np.allclose(got.cpu().numpy(), ref_c.numpy(), rtol=1e-3)
