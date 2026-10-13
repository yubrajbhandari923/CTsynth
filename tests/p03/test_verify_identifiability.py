"""Regression: Experiment A identifiability numbers (F-012, F-018, F-024) from committed tables.

Pinned values are those recorded in RUN-0007 (scipy.fsolve); the numpy Newton solver that replaced
it reproduces them to < 1e-9 relative. Test tolerances (1e-3..1e-5 relative) are far above float64
platform noise and far below any change of method or table. This is a regression check on the
arithmetic of an existing derivation, not independent evidence for the findings.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "projects/p03_unsupervised/verify_identifiability.py"
spec = importlib.util.spec_from_file_location("verify_identifiability", SCRIPT)
assert spec and spec.loader
vi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vi)


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("ident")
    assert vi.main(["--out", str(out)]) == 0
    return json.loads((out / "identifiability.json").read_text())


def test_conditioning_matches_recorded_values(result: dict) -> None:
    assert result["adipose_fibro_angle_deg"] == pytest.approx(1.04340, rel=1e-3)
    assert result["sum_to_one_cond_zero_thickness"] == pytest.approx(1272.96, rel=1e-5)
    assert result["sum_to_one_singular_values"][0] == pytest.approx(3.697, rel=1e-3)
    assert result["sigma_max_Binv_measurement_block"] == pytest.approx(343.4, rel=1e-3)
    assert result["deterministic"] is True


def test_decoys_match_recorded_separations_and_reproduce_measurements(result: dict) -> None:
    by = {d["label"]: d for d in result["decoys"]}
    assert by["f018_original_negative"]["separation_cm"] == pytest.approx(6.530, rel=1e-3)
    assert not by["f018_original_negative"]["nonnegative"]  # F-018's decoy is unphysical
    assert by["f024_nonnegative"]["separation_cm"] == pytest.approx(19.734, rel=1e-3)
    assert by["f024_nonnegative"]["nonnegative"]
    for d in result["decoys"]:  # a decoy is only a decoy if the data cannot tell it apart
        assert max(d["rel_err_T50"], d["rel_err_T80"]) < 1e-12


def test_newton_decoy_solver_solves_a_known_system() -> None:
    tables = {
        k: vi.S.load_model_table(vi.TABLE_DIR / f"model_data_{k}kVp.npy", k) for k in vi.S.KVPS
    }
    x_true = np.array([1.0, 14.0, 0.2])
    dec = vi.exact_decoy(x_true, 0.2, tables, guess=np.array([1.5, 13.0]))
    assert np.allclose(dec, x_true, atol=1e-8)  # calc held at the true value: unique solution
