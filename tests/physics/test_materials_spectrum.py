"""Validation/interpolation of MaterialBasis and Spectrum; model-constructor checks."""

import numpy as np
import physics_reference as ref
import pytest
import torch

from ctsynth.physics import MaterialBasis, SpectralForwardModel, Spectrum

F64 = torch.float64


def test_loglog_interpolation_exact_for_power_law() -> None:
    """kappa = c E^-p is linear in log-log, so log-log interpolation at off-grid E equals the
    closed form (reference: closed-form power law). rtol 1e-12: only exp/log rounding."""
    b = ref.build_basis()
    e = np.array([10.0, 13.7, 61.7, 99.9, 150.0])  # includes both grid ends (closed interval)
    got = b.kappa_at(torch.tensor(e, dtype=F64)).numpy()
    want = np.stack([ref.kappa_pe(e), ref.kappa_cs(e)])
    assert np.allclose(got, want, rtol=1e-12)


def test_no_extrapolation() -> None:
    """Energies outside the grid raise ValueError (spec: no extrapolation)."""
    b = ref.build_basis()
    for bad in (9.99, 150.01):
        with pytest.raises(ValueError):
            b.kappa_at(torch.tensor([bad], dtype=F64))


@pytest.mark.parametrize(
    "what", ["neg_kappa", "nan_kappa", "nonmono", "dup_names", "neg_density", "shape"]
)
def test_basis_validation(what: str) -> None:
    """Invalid tables raise ValueError (positivity, finiteness, monotone grid, shapes)."""
    a = ref.toy_arrays()
    k, e, rd, names = a["kappa_grid"].copy(), a["grid"].copy(), ref.REF_DENSITY.copy(), ("pe", "cs")
    if what == "neg_kappa":
        k[0, 3] = -1.0
    elif what == "nan_kappa":
        k[1, 3] = np.nan
    elif what == "nonmono":
        e[5], e[6] = e[6], e[5]
    elif what == "dup_names":
        names = ("pe", "pe")
    elif what == "neg_density":
        rd[0] = -2.0
    elif what == "shape":
        k = k[:, :-1]
    t = lambda x: torch.tensor(x, dtype=F64)  # noqa: E731
    with pytest.raises(ValueError):
        MaterialBasis(names, t(rd), t(e), t(k))


def test_volume_fraction_to_mass_density() -> None:
    """rho_m = f_m * rho_ref_m (reference: explicit multiplication with REF_DENSITY)."""
    b = ref.build_basis()
    f = torch.tensor([[[0.5, 1.0]], [[0.25, 0.0]]], dtype=F64)  # (2, 1, 2)
    want = f.numpy() * ref.REF_DENSITY[:, None, None]
    assert np.allclose(b.mass_density_from_volume_fraction(f).numpy(), want, rtol=1e-15)


@pytest.mark.parametrize("what", ["neg_fluence", "trans_gt1", "neg_resp", "shape", "nonmono"])
def test_spectrum_validation(what: str) -> None:
    """Invalid spectra raise ValueError (fluence>=0, 0<=T<=1, R>=0, shapes, monotone energies)."""
    a = ref.toy_arrays()
    e, phi, tr, r = a["e"].copy(), a["phi"].copy(), a["trans"].copy(), a["resp"].copy()
    if what == "neg_fluence":
        phi[2] = -1.0
    elif what == "trans_gt1":
        tr[2] = 1.5
    elif what == "neg_resp":
        r[0, 0] = -1.0
    elif what == "shape":
        r = r[:, :-1]
    elif what == "nonmono":
        e[[1, 2]] = e[[2, 1]]
    t = lambda x: torch.tensor(x, dtype=F64)  # noqa: E731
    with pytest.raises(ValueError):
        Spectrum(t(e), t(phi), t(tr), t(r))


def test_spectrum_weights_and_unattenuated() -> None:
    """W = R*Phi*T and unattenuated = row sums (reference: numpy einsum / explicit loop)."""
    s, a = ref.build_spectrum(), ref.toy_arrays()
    want = np.array(
        [sum(a["resp"][b, i] * a["phi"][i] * a["trans"][i] for i in range(12)) for b in range(3)]
    )
    assert np.allclose(s.weights().numpy(), a["w"], rtol=1e-14)
    assert np.allclose(s.unattenuated_counts().numpy(), want, rtol=1e-13)


def test_model_constructor_rejects_bad_inputs() -> None:
    """Spectrum outside basis grid, dtype mismatch, negative/ill-shaped scatter -> ValueError."""
    basis, spec, proj = ref.build_basis(), ref.build_spectrum(), ref.build_projector()
    narrow = MaterialBasis(
        ("pe", "cs"),
        basis.ref_density,
        basis.energies[8:20],  # 38..94 keV; spectrum spans 25..125
        basis.kappa[:, 8:20],
    )
    with pytest.raises(ValueError):
        SpectralForwardModel(narrow, spec, proj)
    with pytest.raises(ValueError):
        SpectralForwardModel(basis, ref.build_spectrum(torch.float32), proj)
    with pytest.raises(ValueError):
        SpectralForwardModel(basis, spec, proj, scatter=-0.1)
    with pytest.raises(ValueError):
        SpectralForwardModel(basis, spec, proj, scatter=torch.ones(3, 2, 2, dtype=F64))
    model = SpectralForwardModel(basis, spec, proj)
    with pytest.raises(ValueError):  # rho dtype must match model
        model.expected_counts(torch.zeros(2, 5, 6, dtype=torch.float32))
