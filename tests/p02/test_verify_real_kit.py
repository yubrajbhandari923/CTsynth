"""Driver test for verify_real_kit on a synthetic kit-layout directory (NOT real AAPM data).

Checks the driver's contract (outputs, determinism, refusal without data) and that the estimators it
ranks behave on data that obeys the linear effective-mu model exactly. It does not verify any claim
about the real kit; that needs the real kit and a registered run.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from ctsynth.aapm.schema import KVPS, load_model_table
from ctsynth.aapm.synthetic import SyntheticConfig, generate, write_synthetic_dir

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "verify_real_kit", ROOT / "projects/p02_aapm_benchmark/verify_real_kit.py"
)
assert spec and spec.loader
vk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vk)


@pytest.fixture(scope="module")
def kit(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tables = {k: load_model_table(vk.TABLE_DIR / f"model_data_{k}kVp.npy", k) for k in KVPS}
    d = tmp_path_factory.mktemp("synthetic_kit_not_aapm")
    write_synthetic_dir(generate(SyntheticConfig(n_cases=3, n_pix=16, seed=5), tables), d)
    return d


def _run(kit: Path, out: Path) -> dict:
    assert vk.main(["--kit", str(kit), "--out", str(out), "--n-cases", "3"]) == 0
    return json.loads((out / "real_kit_verification.json").read_text())


def test_deterministic_and_complete(kit: Path, tmp_path: Path) -> None:
    a, b = _run(kit, tmp_path / "a"), _run(kit, tmp_path / "b")
    assert a == b
    assert a["deterministic"] is True and a["n_cases"] == 3
    assert set(a["rmse"]) == {
        "zero_prediction",
        "unconstrained",
        "unconstrained_clipped",
        "sum_to_one",
        "sum_to_one_clipped",
        "simplex",
    }
    assert all(len(v) == 64 for v in a["kit_files_sha256"].values())


def test_exactly_linear_data_is_recovered_on_tissue_pixels(kit: Path, tmp_path: Path) -> None:
    res = _run(kit, tmp_path / "o")
    assert res["rmse"]["zero_prediction"] > 0.05  # the trivial baseline is clearly worse
    # images are built from the same effective mu, so the model residual is float noise ...
    assert res["linear_model_rel_residual_lo"] < 1e-5 and res["linear_model_rel_residual_hi"] < 1e-5
    # ... and the partition-constrained estimators are exact where the partition holds (tissue).
    # They are NOT exact on air (sum 0 violates sum-to-one): whole-image rmse stays large.
    assert res["tissue_rmse"]["sum_to_one"] < 1e-5 and res["tissue_rmse"]["simplex"] < 1e-5
    assert res["rmse"]["sum_to_one"] > 0.05


def test_refuses_without_kit_data(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="refusing"):
        vk.main(["--kit", str(tmp_path), "--out", str(tmp_path / "o")])
