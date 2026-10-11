"""Unit tests of the independent numpy references. These pass without any implementation."""

import math

import numpy as np
import physics_reference as ref


def test_chord_rect_known_cases() -> None:
    assert math.isclose(ref.chord_rect(0.1, 0.0, -1, 1, -1, 1), 2.0, rel_tol=1e-14)
    assert math.isclose(
        ref.chord_rect(0.0, math.pi / 4, -1, 1, -1, 1), 2 * math.sqrt(2), rel_tol=1e-12
    )
    assert ref.chord_rect(1.5, 0.0, -1, 1, -1, 1) == 0.0


def test_dense_rows_sum_to_whole_rectangle_chord() -> None:
    """Pixel chords partition the chord of the full rectangle (any ray), to rounding."""
    nx, ny, dx = 5, 3, 0.4
    ang = np.array([0.3, 1.2, 2.5])
    mat = ref.dense_matrix(nx, ny, dx, ang, 6, 0.33, 0.05)
    s = ref.det_positions(6, 0.33, 0.05)
    want = [
        ref.chord_rect(sj, th, -nx * dx / 2, nx * dx / 2, -ny * dx / 2, ny * dx / 2)
        for th in ang
        for sj in s
    ]
    assert np.allclose(mat.sum(axis=1), want, rtol=1e-12, atol=1e-13)


def test_disk_chord_known() -> None:
    assert np.allclose(ref.disk_chord(np.array([0.0, 1.0, 2.0]), 0.4, 1.0, 0, 0), [2.0, 0.0, 0.0])
    # disk centred at (1, 0), view along x (theta=0, n=(0,1)): s_c = 0 -> chord at s=0 is 2R
    assert ref.disk_chord(np.array([0.0]), 0.0, 0.5, 1.0, 0.0)[0] == 1.0


def test_power_law_loglog_interp_is_exact() -> None:
    a = ref.toy_arrays()
    e = np.array([26.3, 61.7, 99.9])
    assert np.allclose(
        ref.loglog_interp(e, a["grid"], a["kappa_grid"][0]), ref.kappa_pe(e), rtol=1e-12
    )


def test_counts_zero_path_and_lm_recovers_known_A() -> None:
    a = ref.toy_arrays()
    w, k = a["w"], a["kappa_e"]
    assert np.allclose(ref.counts_ref(w, k, np.zeros(2)), w.sum(axis=1), rtol=1e-14)
    for truth in ([0.2, 2.0], [0.5, 8.0], [0.05, 0.5]):
        y = ref.counts_ref(w, k, np.array(truth))
        got = ref.solve_lm(y, w, k, np.array([0.1, 1.0]))
        assert np.allclose(got, truth, rtol=1e-7), (truth, got)


def test_jacobian_matches_finite_difference_and_has_rank_two() -> None:
    a = ref.toy_arrays()
    w, k = a["w"], a["kappa_e"]
    x = np.array([0.3, 3.0])
    jac = ref.jac_ref(w, k, x)
    h = 1e-6
    fd = np.stack(
        [
            (
                ref.counts_ref(w, k, x + h * np.eye(2)[m])
                - ref.counts_ref(w, k, x - h * np.eye(2)[m])
            )
            / (2 * h)
            for m in range(2)
        ],
        axis=1,
    )
    assert np.allclose(jac, fd, rtol=1e-6)
    sv = np.linalg.svd(jac / ref.counts_ref(w, k, x)[:, None], compute_uv=False)
    assert sv[1] / sv[0] > 1e-3


def test_one_bin_two_material_is_not_identifiable() -> None:
    a = ref.toy_arrays()
    w1, k = a["w"][:1], a["kappa_e"]
    target = ref.counts_ref(w1, k, np.array([0.3, 3.0]))[0]
    a2 = ref.level_set_a2(w1, k, 0.1, target)
    assert abs(a2 - 3.0) > 0.1
    assert math.isclose(ref.counts_ref(w1, k, np.array([0.1, a2]))[0], target, rel_tol=1e-12)
