"""Official s1/s2 and diagnostics vs hand-computed values and explicit-loop references."""

import math

import aapm_reference as ref
import numpy as np

from ctsynth.aapm import metrics as M

# Tolerances: float64 inputs, hand arithmetic quoted to ~1e-12 relative.
RT = 1e-12


def z(c: int = 1, h: int = 2) -> np.ndarray:
    return np.zeros((c, 3, h, h))


def test_s1_hand_values() -> None:
    t, r = z(2), z(2)
    r[0, 0] = 0.1  # case 0: sumsq 4*0.01 -> sqrt(0.04/12)
    r[1, 0, 0, 0] = 0.3  # case 1: sqrt(0.09/12)
    want = (math.sqrt(0.04 / 12) + math.sqrt(0.09 / 12)) / 2
    assert math.isclose(M.s1_mean_rmse(t, r), want, rel_tol=RT)
    assert math.isclose(want, 0.0721688, abs_tol=1e-7)  # hand arithmetic
    assert math.isclose(M.s1_mean_rmse(t, r), ref.ref_s1(t, r), rel_tol=RT)


def test_s1_random_vs_loops_and_3d_input() -> None:
    g = np.random.default_rng(0)
    t, r = g.random((3, 3, 5, 5)), g.random((3, 3, 5, 5))
    assert math.isclose(M.s1_mean_rmse(t, r), ref.ref_s1(t, r), rel_tol=RT)
    assert math.isclose(M.s1_mean_rmse(t[0], r[0]), ref.ref_s1(t[:1], r[:1]), rel_tol=RT)
    assert M.s1_mean_rmse(t, t) == 0.0


def test_s1_not_normalised_not_per_material() -> None:
    """Scaling both maps by k scales s1 by k (no dynamic-range normalisation); materials pooled."""
    g = np.random.default_rng(1)
    t, r = g.random((2, 3, 4, 4)), g.random((2, 3, 4, 4))
    assert math.isclose(M.s1_mean_rmse(5 * t, 5 * r), 5 * M.s1_mean_rmse(t, r), rel_tol=RT)


def test_s2_hand_values_and_skipping() -> None:
    t, p = z(1, 4), z(1, 4)
    t[0, 0, 0, 0] = 0.5  # only ROI(0,0) of size 2 has truth max > 0.01
    p[0, 0, 1, 1] = 0.2  # error inside that ROI
    p[0, 0, 3, 3] = 1.0  # huge error in low-signal ROIs: must be ignored
    # ROI(0,0) holds 3 materials x 2x2 = 12 elements; errors: truth px 0.5 vs pred 0 -> 0.25,
    # pred px 0.2 vs truth 0 -> 0.04. The p=1.0 pixel is only in skipped ROIs.
    want = math.sqrt((0.25 + 0.04) / 12)
    assert math.isclose(M.s2_worst_roi_rmse(t, p, roi=2), want, rel_tol=RT)
    assert math.isclose(want, ref.ref_s2(t, p, 2), rel_tol=RT)


def test_s2_threshold_is_strict() -> None:
    t, p = z(1, 4), z(1, 4)
    p[0, 0, 0, 0] = 0.3
    t[0, 1, 0, 0] = 0.01  # max <= 0.01 -> skipped
    assert M.s2_worst_roi_rmse(t, p, roi=2) == 0.0
    t[0, 1, 0, 0] = 0.011  # just above -> counted (any material's max counts)
    # ROI(0,0), 12 elements: pred 0.3 vs truth 0 -> 0.09; truth 0.011 vs 0 -> 1.21e-4.
    want = math.sqrt((0.09 + 0.011**2) / 12)
    assert math.isclose(M.s2_worst_roi_rmse(t, p, roi=2), want, rel_tol=1e-9)


def test_s2_last_position_quirk() -> None:
    t, p = z(1, 4), z(1, 4)
    t[0, 0, 3, 3] = 0.5
    p[0, 0, 3, 3] = 1.1  # only ROI(2,2) covers pixel (3,3); kit loop range(H-roi) omits it
    assert M.s2_worst_roi_rmse(t, p, roi=2) == 0.0
    got = M.s2_worst_roi_rmse(t, p, roi=2, include_last_position=True)
    assert math.isclose(got, math.sqrt(0.36 / 12), rel_tol=RT)
    assert math.isclose(got, ref.ref_s2(t, p, 2, include_last=True), rel_tol=RT)


def test_s2_max_over_cases_and_random_vs_loops() -> None:
    g = np.random.default_rng(2)
    t, r = g.random((3, 3, 6, 6)) * 0.02, g.random((3, 3, 6, 6)) * 0.02
    t[1, 0, 2, 2] = 0.0  # keep some low-signal structure
    assert math.isclose(M.s2_worst_roi_rmse(t, r, roi=3), ref.ref_s2(t, r, 3), rel_tol=RT)
    assert M.s2_worst_roi_rmse(t, r, roi=3) >= M.s2_worst_roi_rmse(t[:1], r[:1], roi=3)


def test_default_roi_is_25() -> None:
    assert M.ROI_SIZE == 25 and M.ROI_TRUTH_MAX_MIN == 0.01
    t, p = np.ones((1, 3, 26, 26)), np.zeros((1, 3, 26, 26))
    assert math.isclose(M.s2_worst_roi_rmse(t, p), 1.0, rel_tol=RT)  # positions 0 only


def test_per_material_and_aggregate() -> None:
    t, r = z(1), z(1)
    r[0, 0] = 0.1
    r[0, 1, 0, 0] = 0.2
    pm = M.per_material_errors(t, r)
    assert list(pm) == ["adipose", "fibroglandular", "calcification"]
    assert math.isclose(pm["adipose"]["mae"], 0.1, rel_tol=RT)
    assert math.isclose(pm["adipose"]["rmse"], 0.1, rel_tol=RT)
    assert math.isclose(pm["fibroglandular"]["mae"], 0.05, rel_tol=RT)
    assert math.isclose(pm["fibroglandular"]["rmse"], 0.1, rel_tol=RT)  # sqrt(0.04/4)
    assert pm["calcification"] == {"mae": 0.0, "rmse": 0.0}
    assert math.isclose(M.aggregate_rmse(t, r), math.sqrt(0.08 / 12), rel_tol=RT)


def test_aggregate_differs_from_s1() -> None:
    t, r = z(2), z(2)
    r[0, 0] = 0.1  # whole 2x2 plane: 4 * 0.01 = 0.04 (case rmse sqrt(0.04/12))
    r[1, 0, 0, 0] = 0.3  # one pixel: 0.09 (case rmse sqrt(0.0075))
    # pooled: (0.04 + 0.09) / (2 cases * 3 * 2 * 2 = 24 elements)
    assert math.isclose(M.aggregate_rmse(t, r), math.sqrt(0.13 / 24), rel_tol=RT)
    assert not math.isclose(M.aggregate_rmse(t, r), M.s1_mean_rmse(t, r), rel_tol=1e-3)


def regime_case() -> tuple[np.ndarray, np.ndarray]:
    """2x2: A air, B tissue, C tissue, D band (flattened order A,B,C,D)."""
    t = np.zeros((1, 3, 2, 2))
    t[0, :, 0, 1] = (0.6, 0.4, 0.0)
    t[0, :, 1, 0] = (0.7, 0.2, 0.1)
    t[0, :, 1, 1] = (0.2, 0.1, 0.0)
    p = t.copy()
    p[0, :, 0, 0] = (0.05, 0.0, 0.01)  # air pixel, sum 0.06
    p[0, :, 0, 1] = (0.5, 0.4, 0.0)  # tissue sum 0.9
    p[0, :, 1, 0] = (0.72, 0.2, 0.1)  # tissue sum 1.02
    p[0, :, 1, 1] = (0.1, 0.1, 0.1)  # band: error (-0.1, 0, +0.1)
    return t, p


def test_regime_masks() -> None:
    t, _ = regime_case()
    air, tissue, band = M.regime_masks(t)
    assert air.shape == tissue.shape == band.shape == (1, 2, 2)
    assert air[0].tolist() == [[True, False], [False, False]]
    assert tissue[0].tolist() == [[False, True], [True, False]]
    assert band[0].tolist() == [[False, False], [False, True]]
    e = z(1, 3)
    e[0, 0, 0, 0], e[0, 0, 0, 1], e[0, 0, 0, 2] = 0.0005, 0.9995, 0.5  # edges of tol = 1e-3
    a2, t2, b2 = M.regime_masks(e)
    assert a2[0, 0, 0] and not t2[0, 0, 0]
    assert t2[0, 0, 1] and not a2[0, 0, 1]
    assert b2[0, 0, 2]
    assert np.all(a2 ^ t2 ^ b2) and not np.any(a2 & t2) and not np.any(a2 & b2 & t2)


def test_partition_errors_separate_regimes() -> None:
    t, p = regime_case()
    e = M.partition_errors(t, p)
    assert math.isclose(e["air_mean_abs"], 0.06, abs_tol=1e-12)  # |0.06|
    assert math.isclose(e["air_max_abs"], 0.06, abs_tol=1e-12)
    assert math.isclose(e["tissue_mean_abs"], 0.06, abs_tol=1e-12)  # (0.1 + 0.02) / 2
    assert math.isclose(e["tissue_max_abs"], 0.1, abs_tol=1e-12)
    assert (e["n_air"], e["n_tissue"], e["n_band"]) == (1.0, 2.0, 1.0)
    # perfect prediction conserves fractions in every regime
    z0 = M.partition_errors(t, t)
    assert z0["air_max_abs"] == 0.0 and z0["tissue_max_abs"] <= 1e-12


def test_partition_empty_regime_is_nan() -> None:
    t = np.zeros((1, 3, 2, 2))
    t[0, 0] = 1.0  # all tissue
    e = M.partition_errors(t, t)
    assert math.isnan(e["air_mean_abs"]) and math.isnan(e["air_max_abs"])
    assert e["n_air"] == 0.0 and e["n_band"] == 0.0
    assert math.isnan(M.boundary_rmse(t, t))


def test_boundary_rmse() -> None:
    t, p = regime_case()
    assert math.isclose(M.boundary_rmse(t, p), math.sqrt(0.02 / 3), rel_tol=RT)  # one band px
    p2 = p.copy()
    p2[0, :, 0, 0] = 9.0  # error outside the band must not matter
    assert math.isclose(M.boundary_rmse(t, p2), math.sqrt(0.02 / 3), rel_tol=RT)
