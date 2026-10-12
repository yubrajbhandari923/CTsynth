"""Fast tests for the EXP-004 runner on a tiny configuration: determinism, label-free isolation from
truth, the null-cos metric against a hand-built error vector, and the split guard."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "projects/p03_unsupervised"))
import run_exp004 as ex  # noqa: E402

from ctsynth.aapm import schema as S  # noqa: E402
from ctsynth.config import ExperimentConfig  # noqa: E402


def tiny_cfg() -> ExperimentConfig:
    params = {
        "n_pix": 16, "n_views": 16, "n_det": 24, "n_cases": 12,
        "f_train": 0.5, "f_val": 0.5, "f_test": 0.0,
        "table_dir": "data/aapm_kit_sample", "roi": 8,
        "depth": 1, "width": 4, "epochs": 2, "lr": 0.003, "batch_size": 4, "n_seeds": 1,
        "air_cutoff": 0.1, "prior_weight": 1.0, "lam_tv": 0.0001, "tv_iters": 5,
        "mixed_thresh": 0.9,
    }  # fmt: skip
    return ExperimentConfig(experiment_id="EXP-004", name="tiny", seed=0, params=params)


def rungs(*ids: str):  # type: ignore[no-untyped-def]
    return tuple(r for r in ex.RUNGS if r["id"] in ids)


def test_deterministic_same_config_and_seed():
    a, _, _ = ex.run_all(tiny_cfg(), rungs("B3", "B4b"), seeds=[0])
    b, _, _ = ex.run_all(tiny_cfg(), rungs("B3", "B4b"), seeds=[0])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c, _, _ = ex.run_all(tiny_cfg(), rungs("B3"), seeds=[1])
    assert c["rungs"]["B3"]["per_seed"]["seed1"] != a["rungs"]["B3"]["per_seed"]["seed0"]


def test_label_free_path_never_receives_truth(monkeypatch):
    seen = []
    real = ex.train_label_free

    def spy(factory, fwd, meas_tr, meas_va, weights, **kw):  # type: ignore[no-untyped-def]
        seen.append((meas_tr.shape, meas_va.shape, set(kw), dict(weights)))
        return real(factory, fwd, meas_tr, meas_va, weights, **kw)

    monkeypatch.setattr(ex, "train_label_free", spy)
    ex.run_all(tiny_cfg(), rungs("B3"), seeds=[0])
    assert seen, "label-free trainer was not called"
    shape_tr, shape_va, kw_names, weights = seen[0]
    assert (
        shape_tr[1] == 2 and shape_va[1] == 2
    )  # measurement channels only, never 3 truth fractions
    assert not kw_names & {"truth", "labels", "truth_train", "truth_val", "phantoms", "y"}
    assert set(weights) == {"consistency_image"}


def test_label_free_signature_has_no_truth():
    params = set(
        ex.train_label_free.__code__.co_varnames[: ex.train_label_free.__code__.co_argcount]
    )
    assert not params & {"truth", "labels", "truth_train", "truth_val", "phantoms", "y"}


def test_null_cos_hand_constructed():
    n = ex.NULL_F014
    truth = np.zeros((1, 3, 1, 4))
    truth[:, 0] = 0.5  # tissue: fractions sum to 1 (two materials at 0.5, one at 0)
    truth[:, 1] = 0.5
    pred = truth.copy()
    pred[0, :, 0, 0] += n  # error parallel to null  -> |cos| = 1
    pred[0, :, 0, 1] += [0.691, 0.723, 0.0]  # orthogonal to null -> cos = 0
    # pixel 2: zero error, excluded from the mean
    pred[0, :, 0, 3] += 0.1 * np.array(
        [-0.6, 0.0, 0.8]
    )  # cos = |(-0.6*n0 + 0.8*n2)|/1 on unit vector
    tissue = np.ones((1, 1, 4), bool)
    cos, npix = ex.null_cos_tissue(truth, pred, tissue)
    expected_third = abs(np.array([-0.6, 0.0, 0.8]) @ n) / np.linalg.norm([-0.6, 0.0, 0.8])
    assert npix == 3
    assert cos == pytest.approx((1.0 + 0.0 + expected_third) / 3, abs=1e-9)


def test_null_direction_matches_kit_tables(tables_np):
    assert ex._null_check(tables_np) > 0.999


def test_mixed_voxel_fraction_definition():
    maps = np.zeros((1, 3, 1, 3))
    maps[0, 0, 0, 0] = 0.95  # tissue, pure enough (max >= 0.9) -> not mixed
    maps[0, 0, 0, 1] = 0.5  # tissue, max < 0.9 -> mixed
    maps[0, 1, 0, 1] = 0.5
    maps[0, 0, 0, 2] = 0.0  # NOT tissue -> excluded from the fraction
    tissue = np.array([[[True, True, False]]])
    assert ex.mixed_voxel_frac(maps, tissue, 0.9) == pytest.approx(0.5)


def test_split_loads_only_train_and_val(monkeypatch):
    calls = []
    real = ex.split_indices

    def spy(n, seed, name, **kw):  # type: ignore[no-untyped-def]
        calls.append(name)
        return real(n, seed, name, **kw)

    monkeypatch.setattr(ex, "split_indices", spy)
    ex.run_all(tiny_cfg(), rungs("B1"), seeds=[0])
    assert set(calls) == {"train", "val"}


def test_aggregate_std_and_cv():
    recs = [
        {"s1": 1.0, "s2": 0.0, "per_material_rmse": {m: 0.0 for m in S.MATERIALS},
         "regime_rmse": {"air": None, "tissue": v, "band": None}, "consistency_train": 0.0,
         "consistency_val": 0.0, "null_cos_tissue": 0.5, "mixed_voxel_frac": 0.0}
        for v in (1.0, 3.0)
    ]  # fmt: skip
    agg = ex.aggregate(recs)
    assert agg["rmse_tissue"]["mean"] == pytest.approx(2.0)
    assert agg["rmse_tissue"]["std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert agg["rmse_tissue"]["cv"] == pytest.approx(np.std([1.0, 3.0], ddof=1) / 2.0)
    assert agg["rmse_air"] is None
