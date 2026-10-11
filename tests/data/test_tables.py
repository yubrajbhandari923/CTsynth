"""Checks on the frozen toy-experiment tables in data/tables/.

numpy and the standard library only: xraylib and spekpy are NOT imported here.
Run with: uv run pytest tests/data
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLES = ROOT / "data" / "tables"
ATT = TABLES / "attenuation_v1.npz"
SPEC = TABLES / "spectra_v1.npz"
PROV = TABLES / "provenance_v1.json"

# Independent NIST mu/rho values (total with coherent), cm^2/g, hard-coded here on purpose.
# water: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
# iodine: https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z53.html
# calcium: https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z20.html
# All fetched or matched on 2026-10-11. NIST and xraylib differ in data lineage: rtol 2% is a
# loose consistency bound.
NIST = [
    ("water", 30.0, 0.3756),
    ("water", 60.0, 0.2059),
    ("water", 100.0, 0.1707),
    ("iodine", 60.0, 7.579),
    ("iodine", 100.0, 1.942),
    ("calcium", 60.0, 0.6578),
    ("calcium", 100.0, 0.2571),
]
NIST_RTOL = 0.02


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def att():
    d = np.load(ATT, allow_pickle=False)
    return {k: d[k] for k in d.files}


@pytest.fixture(scope="module")
def spec():
    d = np.load(SPEC, allow_pickle=False)
    return {k: d[k] for k in d.files}


@pytest.fixture(scope="module")
def prov():
    return json.loads(PROV.read_text(encoding="utf-8"))


def test_attenuation_shapes_and_dtypes(att):
    assert att["energies_keV"].shape == (141,)
    assert att["energies_keV"].dtype == np.float64
    assert att["kappa_cm2_per_g"].shape == (3, 141)
    assert att["kappa_cm2_per_g"].dtype == np.float64
    assert att["ref_density_g_cm3"].shape == (3,)
    assert list(att["names"]) == ["water", "iodine", "calcium"]
    np.testing.assert_allclose(att["ref_density_g_cm3"], [1.0, 4.93, 1.55])


def test_energy_grid_is_1keV_from_10_to_150(att, spec):
    e = att["energies_keV"]
    np.testing.assert_array_equal(e, np.arange(10.0, 151.0, 1.0))
    assert np.all(np.diff(e) > 0)
    np.testing.assert_array_equal(spec["energies_keV"], e)


def test_kappa_finite_and_positive(att):
    k = att["kappa_cm2_per_g"]
    assert np.all(np.isfinite(k))
    assert np.all(k > 0)


def test_kappa_nist_spot_checks(att):
    e = att["energies_keV"]
    names = list(att["names"])
    for name, energy, nist in NIST:
        value = att["kappa_cm2_per_g"][names.index(name), int(np.where(e == energy)[0][0])]
        assert value == pytest.approx(nist, rel=NIST_RTOL), (name, energy, value, nist)


def test_iodine_k_edge_is_under_resolved(att):
    # Documented limitation: the 1 keV grid straddles the I K-edge (33.17 keV). The jump between
    # the 33 and 34 keV grid points must be large, confirming the edge is not resolved.
    e = att["energies_keV"]
    k = att["kappa_cm2_per_g"][1]
    i33 = int(np.where(e == 33.0)[0][0])
    assert k[i33 + 1] / k[i33] > 3.0


def test_spectra_shapes_and_nonnegative(spec):
    for key in ("fluence_80", "fluence_120"):
        f = spec[key]
        assert f.shape == (141,)
        assert f.dtype == np.float64
        assert np.all(np.isfinite(f))
        assert np.all(f >= 0)
        assert f.sum() > 0


def test_no_photons_above_kvp(spec):
    # 1 keV bin n covers [n-0.5, n+0.5). A bin whose lower edge is >= kVp must be empty.
    e = spec["energies_keV"]
    for kvp in (80, 120):
        f = spec[f"fluence_{kvp}"]
        above = (e - 0.5) >= kvp
        assert np.any(above)
        assert np.all(f[above] == 0.0), kvp


def test_mean_energy_plausible_and_ordered(spec):
    # Photon-weighted mean energy of a filtered W spectrum is well below kVp. Sanity window fixed
    # in advance: 0.3-0.8 x kVp. Higher kVp must give a higher mean energy.
    e = spec["energies_keV"]
    means = {}
    for kvp in (80, 120):
        f = spec[f"fluence_{kvp}"]
        means[kvp] = float((e * f).sum() / f.sum())
        assert 0.3 * kvp < means[kvp] < 0.8 * kvp, (kvp, means[kvp])
    assert means[80] < means[120]


def test_spectra_fluence_conserved_against_provenance(spec, prov):
    # The rebinning must conserve the native in-grid fluence to 1e-9 relative.
    for kvp in (80, 120):
        total = float(spec[f"fluence_{kvp}"].sum())
        native = prov["spectra"]["per_kvp_stats"][str(kvp)]["native_in_grid_total_photons_cm2"]
        assert abs(total - native) / native <= 1e-9, kvp


def test_provenance_has_versions(prov):
    libs = prov["libraries"]
    for key in ("python", "xraylib", "spekpy", "numpy"):
        assert libs.get(key), key
    assert prov["table_version"] == "v1"
    assert prov["generated_utc"].endswith("Z")


def test_provenance_hashes_match_files(prov):
    assert prov["sha256"]["attenuation_v1.npz"] == _sha256(ATT)
    assert prov["sha256"]["spectra_v1.npz"] == _sha256(SPEC)
    generator = ROOT / prov["generator"]["script"]
    assert prov["generator"]["sha256"] == _sha256(generator)


def test_provenance_nist_checks_recorded_and_passing(prov):
    checks = prov["nist_spot_checks"]
    assert checks["rtol"] == NIST_RTOL
    assert checks["all_pass"] is True
    assert len(checks["results"]) == len(NIST)
    assert all(r["pass"] for r in checks["results"])
    assert checks["skipped"] == []


def test_provenance_records_limitations(prov):
    text = " ".join(prov["known_limitations"])
    assert "K-edge" in text
    assert "normalisation" in text.lower() or "normalization" in text.lower()
