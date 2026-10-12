"""Fast tests for the AAPM evaluation harness on tiny constructed or synthetic arrays."""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
P02 = ROOT / "projects/p02_aapm_benchmark"
sys.path.insert(0, str(P02))
import evaluate as ev  # noqa: E402

from ctsynth.aapm import metrics as M  # noqa: E402
from ctsynth.aapm import schema as S  # noqa: E402
from ctsynth.aapm.schema import ModelTable  # noqa: E402
from ctsynth.aapm.synthetic import SyntheticConfig, generate, write_synthetic_dir  # noqa: E402

REQUIRED_KEYS = {
    "record_version", "method_name", "split_name", "n_cases", "shapes", "s1", "s2",
    "s2_full_scan", "aggregate_rmse", "per_material", "partition", "regime_pixel_counts",
    "boundary_rmse", "failure_cases", "limitations",
}  # fmt: skip


def _rand(seed: int, c: int = 2, n: int = 6) -> tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(seed)
    t = g.random((c, 3, n, n)) * 0.5
    r = g.random((c, 3, n, n)) * 0.5
    return t, r


def test_record_has_required_keys_and_plain_json() -> None:
    t, r = _rand(0)
    rec = ev.evaluate_predictions(t, r, split_name="val", method_name="m", roi=3)
    assert REQUIRED_KEYS <= set(rec)
    assert rec["method_name"] == "m" and rec["split_name"] == "val"
    assert rec["shapes"] == {"truth": [2, 3, 6, 6], "pred": [2, 3, 6, 6]}
    json.dumps(rec, allow_nan=False)  # plain, strictly JSON
    assert isinstance(rec["s1"], float) and isinstance(rec["s2"], float)
    assert math.isclose(rec["s1"], M.s1_mean_rmse(t, r), rel_tol=1e-12)
    assert math.isclose(rec["s2"], M.s2_worst_roi_rmse(t, r, roi=3), rel_tol=1e-12)
    assert math.isclose(rec["aggregate_rmse"], M.aggregate_rmse(t, r), rel_tol=1e-12)


def test_per_material_names_match_schema() -> None:
    t, r = _rand(1)
    rec = ev.evaluate_predictions(t, r, split_name="val", method_name="m")
    assert list(rec["per_material"]) == list(S.MATERIALS)
    for name in S.MATERIALS:
        assert set(rec["per_material"][name]) == {"mae", "rmse"}
    assert math.isclose(
        rec["per_material"]["calcification"]["mae"],
        float(np.abs(t[:, 2] - r[:, 2]).mean()),
        rel_tol=1e-12,
    )


def test_regime_errors_use_the_right_pixels() -> None:
    # 2x2 pixels, one case. (0,0) air; (0,1) and (1,0) tissue; (1,1) boundary band (sum 0.3).
    t = np.zeros((1, 3, 2, 2))
    t[0, :, 0, 1] = (0.5, 0.5, 0.0)
    t[0, :, 1, 0] = (0.0, 1.0, 0.0)
    t[0, :, 1, 1] = (0.3, 0.0, 0.0)
    r = t.copy()
    r[0, 0, 0, 0] = 0.02  # error ONLY on air: Sp = 0.02 there
    r[0, 1, 0, 1] = 0.4  # tissue pixel (0,1): pred sum 0.9, error 0.1
    r[0, 0, 1, 1] = 0.9  # band pixel error: must not enter any regime partition error
    rec = ev.evaluate_predictions(t, r, split_name="val", method_name="m")
    assert rec["regime_pixel_counts"] == {"tissue": 2, "air": 1, "band": 1}
    assert math.isclose(rec["partition"]["air"]["mean_abs"], 0.02, rel_tol=1e-12)
    assert math.isclose(rec["partition"]["air"]["max_abs"], 0.02, rel_tol=1e-12)
    # tissue: |Sp-1| = 0.1 at (0,1) and 0 at (1,0); the band pixel is excluded.
    assert math.isclose(rec["partition"]["tissue"]["mean_abs"], 0.05, rel_tol=1e-12)
    assert math.isclose(rec["partition"]["tissue"]["max_abs"], 0.1, rel_tol=1e-12)


def test_boundary_rmse_and_undefined_values_are_null() -> None:
    t = np.zeros((1, 3, 2, 2))
    t[0, :, 1, 1] = (0.3, 0.0, 0.0)  # the only band pixel
    r = t.copy()
    r[0, 0, 1, 1] = 0.1
    rec = ev.evaluate_predictions(t, r, split_name="val", method_name="m")
    assert math.isclose(rec["boundary_rmse"], math.sqrt(0.04 / 3), rel_tol=1e-12)

    no_band = np.zeros((1, 3, 2, 2))  # all air: no band, no tissue
    rec2 = ev.evaluate_predictions(no_band, no_band, split_name="val", method_name="m")
    assert rec2["boundary_rmse"] is None
    assert rec2["partition"]["tissue"]["mean_abs"] is None
    json.dumps(rec2, allow_nan=False)


def test_worst_roi_recorder_picks_the_worst_window() -> None:
    """Brute-force reference over every stride-1 window (kit loop range) must agree in location."""
    roi = 3
    t = np.zeros((2, 3, 8, 8))
    t[:, 0] = 0.5  # every ROI has truth max > 0.01
    r = t.copy()
    r[1, 1, 5, 2] = 0.4  # case 1, one pixel: inside windows x0 in {3,4}, y0 in {0,1,2}
    r[0, 2, 1, 1] = 0.1
    rec = ev.evaluate_predictions(t, r, split_name="val", method_name="m", roi=roi)
    worst = rec["failure_cases"]["worst_roi_s2"]

    best = (-1.0, None)
    for c in range(2):
        for x in range(8 - roi):
            for y in range(8 - roi):
                e = (t[c, :, x : x + roi, y : y + roi] - r[c, :, x : x + roi, y : y + roi]) ** 2
                v = math.sqrt(e.sum() / (3 * roi * roi))
                if v > best[0]:
                    best = (v, (c, x, y))
    assert (worst["case"], worst["x0"], worst["y0"]) == best[1]
    assert math.isclose(worst["value"], best[0], rel_tol=1e-12)
    assert math.isclose(rec["s2"], best[0], rel_tol=1e-12)


def test_worst_case_recorder_orders_by_s1() -> None:
    t = np.zeros((3, 3, 4, 4))
    r = np.zeros((3, 3, 4, 4))
    r[0] = 0.01
    r[1] = 0.2
    r[2] = 0.05
    rec = ev.evaluate_predictions(t, r, split_name="val", method_name="m", k_worst=2)
    cases = [w["case"] for w in rec["failure_cases"]["worst_cases_by_s1"]]
    assert cases == [1, 2]
    assert math.isclose(
        rec["failure_cases"]["worst_cases_by_s1"][0]["s1"], M.s1_mean_rmse(t[1:2], r[1:2])
    )


def test_extra_extends_limitations_and_cannot_overwrite() -> None:
    t, r = _rand(4)
    rec = ev.evaluate_predictions(
        t, r, split_name="val", method_name="m", extra={"limitations": ["caveat"], "seed": 7}
    )
    assert rec["limitations"][-1] == "caveat" and rec["seed"] == 7
    with pytest.raises(ValueError):
        ev.evaluate_predictions(t, r, split_name="val", method_name="m", extra={"s1": 0.0})


def test_non_finite_and_bad_shapes_are_refused() -> None:
    t, r = _rand(5)
    r[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ev.evaluate_predictions(t, r, split_name="val", method_name="m")
    with pytest.raises(ValueError):
        ev.evaluate_predictions(t, t[:1], split_name="val", method_name="m")


# --- end-to-end on synthetic data (schema mirror, NOT AAPM) ---

TABLE_DIR = ROOT / "data" / "aapm_kit_sample"


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raw = {k: np.load(TABLE_DIR / f"model_data_{k}kVp.npy").astype(np.float64) for k in (50, 80)}
    tables = {k: ModelTable(k, v[0], v[1], v[2:5]) for k, v in raw.items()}
    scfg = SyntheticConfig(n_cases=8, n_pix=16, n_views=16, n_det=24, seed=3)
    d = tmp_path_factory.mktemp("synthetic_not_aapm")
    write_synthetic_dir(generate(scfg, tables), d)
    return d


def _config(tmp_path: Path) -> Path:
    cfg = {
        "experiment_id": "EXP-TEST",
        "name": "p02 harness test (synthetic)",
        "seed": 11,
        "params": {"n_pix": 16, "n_views": 16, "n_det": 24, "roi": 5},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run(synth_dir: Path, cfg: Path, out: Path) -> None:
    ev.main([str(cfg), "--data-dir", str(synth_dir), "--out", str(out), "--split", "train"])


def test_main_is_deterministic_and_writes_expected_files(synth_dir: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _run(synth_dir, cfg, tmp_path / "a")
    _run(synth_dir, cfg, tmp_path / "b")
    assert (tmp_path / "a/metrics.json").read_bytes() == (tmp_path / "b/metrics.json").read_bytes()
    m = json.loads((tmp_path / "a/metrics.json").read_text())
    rec = m["methods"]["zero_prediction"]
    assert REQUIRED_KEYS <= set(rec)
    assert rec["split_name"] == "train" and m["n_cases"] == len(m["case_indices"]) >= 1
    assert m["not_computed"] == {}
    for f in ("run_meta.json", "timing.json", "manifest.json", "worst_case_zero_prediction.npz"):
        assert (tmp_path / "a" / f).exists()
    assert "wall_s" in json.loads((tmp_path / "a/timing.json").read_text())["zero_prediction"]


def test_manifest_sha256_matches_files(synth_dir: Path, tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    _run(synth_dir, cfg, tmp_path / "a")
    manifest = json.loads((tmp_path / "a/manifest.json").read_text())
    names = {Path(e["path"]).name for e in manifest["inputs"]}
    assert names == {f"{s}.npy" for s in ("Phantom_Adipose", "Phantom_Fibroglandular",
                                          "Phantom_Calcification", "lowkVpTransmission",
                                          "highkVpTransmission", "lowkVpImages",
                                          "highkVpImages")} | {"cfg.yaml"}  # fmt: skip
    for e in manifest["inputs"]:
        p = Path(e["path"])
        assert e["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
        assert e["bytes"] == p.stat().st_size
