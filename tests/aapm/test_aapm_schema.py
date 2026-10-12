"""Frozen schema conventions: accept valid, reject each violation naming its convention."""

import math
from pathlib import Path

import numpy as np
import pytest

from ctsynth.aapm import schema as S
from ctsynth.aapm.schema import GeometrySpec, SchemaError

G = GeometrySpec(n_pix=8, fov_cm=18.0, n_views=8, n_det=12)
N = 2
Arr = np.ndarray


def good_phantoms() -> Arr:
    p = np.zeros((N, 3, 8, 8), np.float32)
    p[:, 0, :, :4] = 0.7
    p[:, 1, :, :4] = 0.3
    p[:, 0, 1, 1], p[:, 1, 1, 1], p[:, 2, 1, 1] = 0.5, 0.3, 0.2  # one calcification px, sum 1
    return p


def good_t(v: float) -> Arr:
    return np.full((N, 4, 12), v, np.float32)


def good_img() -> Arr:
    return np.full((N, 8, 8), 0.2, np.float32)


def good_dataset() -> tuple[Arr, dict[int, Arr], dict[int, Arr]]:
    return good_phantoms(), {50: good_t(0.3), 80: good_t(0.5)}, {50: good_img(), 80: good_img()}


def test_constants_frozen() -> None:  # passes now: constants are part of the frozen spec
    assert S.MATERIALS == ("adipose", "fibroglandular", "calcification")
    assert (S.KVP_HIGH, S.KVP_LOW) == (80, 50)
    assert S.REAL_GEOMETRY == GeometrySpec(512, 18.0, 512, 1024, 50.0, 100.0)
    assert S.DTYPE == np.dtype("float32")


def test_geometry_methods() -> None:
    g = S.REAL_GEOMETRY
    assert math.isclose(g.pixel_cm(), 18 / 512, rel_tol=1e-15)  # exact ratio
    want = 2 * math.tan(math.asin(9 / 50)) * 100
    assert math.isclose(g.detector_length_cm(), want, rel_tol=1e-12)  # same closed form
    a = g.view_angles()
    assert a.shape == (512,) and a.dtype == np.float64
    assert a[0] == 0.0 and math.isclose(a[1], 2 * math.pi / 512, rel_tol=1e-14)
    assert math.isclose(a[-1], 2 * math.pi * 511 / 512, rel_tol=1e-14)


def test_kvp_parity() -> None:
    assert [S.kvp_of_view(k) for k in range(4)] == [80, 50, 80, 50]
    assert S.view_indices(80, 8).tolist() == [0, 2, 4, 6]
    assert S.view_indices(50, 8).tolist() == [1, 3, 5, 7]
    with pytest.raises(SchemaError) as e:
        S.view_indices(60, 8)
    assert e.value.convention == S.CONV_KVP_PARITY
    full = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    d = S.deinterlace(full)
    assert np.array_equal(d[80], full[:, 0::2]) and np.array_equal(d[50], full[:, 1::2])


def test_valid_dataset_accepted() -> None:
    p, t, i = good_dataset()
    S.validate_dataset(p, t, i, G)
    S.validate_phantoms(p, G, material_names=S.MATERIALS)
    S.validate_images(i[50], G)
    S.validate_transmission(t[50], G)


@pytest.mark.parametrize(
    ("mutate", "conv"),
    [
        (lambda p: p.transpose(1, 0, 2, 3).copy(), S.CONV_SHAPE),  # material axis first
        (lambda p: p[..., :7].copy(), S.CONV_SHAPE),  # non-square
        (lambda p: p.astype(np.float64), S.CONV_DTYPE),
        (lambda p: np.where(p == 0.7, np.nan, p).astype(np.float32), S.CONV_FINITE),
        (lambda p: np.where(p == 0.7, 1.5, p).astype(np.float32), S.CONV_RANGE),
        (lambda p: np.where(p == 0.3, -0.1, p).astype(np.float32), S.CONV_RANGE),
        (lambda p: np.where(p == 0.3, 0.7, p).astype(np.float32), S.CONV_PARTITION),  # sum 1.4
        (lambda p: p[:, [2, 1, 0]].copy(), S.CONV_MATERIAL_ORDER),  # calc <-> adipose swapped
    ],
)
def test_phantom_violations(mutate, conv) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SchemaError) as e:
        S.validate_phantoms(mutate(good_phantoms()), G)
    assert e.value.convention == conv


def test_material_names_permutation_detected() -> None:
    """adipose<->fibroglandular cannot be seen in pixels; names catch it."""
    for bad in (
        ("fibroglandular", "adipose", "calcification"),
        ("calcification", "fibroglandular", "adipose"),
        ("adipose", "fibroglandular"),
    ):
        with pytest.raises(SchemaError) as e:
            S.check_material_order(bad)
        assert e.value.convention == S.CONV_MATERIAL_ORDER
        with pytest.raises(SchemaError):
            S.validate_phantoms(good_phantoms(), G, material_names=bad)
    S.check_material_order(S.MATERIALS)


@pytest.mark.parametrize(
    ("arr", "conv"),
    [
        (np.zeros((N, 12, 4), np.float32) + 0.5, S.CONV_SHAPE),  # (view, bin) transposed
        (np.zeros((N, 8, 12), np.float32) + 0.5, S.CONV_SHAPE),  # full views, not per-kVp
        (good_t(0.5).astype(np.float64), S.CONV_DTYPE),
        (good_t(0.0), S.CONV_RANGE),  # T must be > 0
        (good_t(1.2), S.CONV_RANGE),  # T must be <= 1
        (good_t(np.nan), S.CONV_FINITE),
    ],
)
def test_transmission_violations(arr, conv) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SchemaError) as e:
        S.validate_transmission(arr, G)
    assert e.value.convention == conv


def test_image_violations() -> None:
    for arr, conv in (
        (good_img()[:, :, :7], S.CONV_SHAPE),
        (good_img().astype(np.float64), S.CONV_DTYPE),
        (np.where(good_img() > 0, np.inf, 0).astype(np.float32), S.CONV_FINITE),
    ):
        with pytest.raises(SchemaError) as e:
            S.validate_images(arr, G)
        assert e.value.convention == conv


def test_dataset_cross_array_violations() -> None:
    p, t, i = good_dataset()
    cases = {
        "swapped kVp labels": ({50: t[80], 80: t[50]}, i, S.CONV_KVP_PARITY),
        "bad key": ({40: t[50], 80: t[80]}, i, S.CONV_KVP_PARITY),
        "image keys": (t, {50: i[50]}, S.CONV_KVP_PARITY),
        "case count": ({50: t[50][:1], 80: t[80][:1]}, i, S.CONV_SHAPE),
    }
    for tt, ii, conv in cases.values():
        with pytest.raises(SchemaError) as e:
            S.validate_dataset(p, tt, ii, G)
        assert e.value.convention == conv


def test_load_model_table_committed(tables: dict[int, S.ModelTable]) -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "aapm_kit_sample"
    for kvp, n_e in ((50, 98), (80, 158)):
        t = S.load_model_table(root / f"model_data_{kvp}kVp.npy", kvp)
        assert t.kvp == kvp and t.energies_kev.shape == (n_e,) and t.mu_per_cm.shape == (3, n_e)
        assert np.array_equal(t.energies_kev, tables[kvp].energies_kev)
        assert np.array_equal(t.mu_per_cm, tables[kvp].mu_per_cm)  # rows 2-4 = a, f, c
        assert math.isclose(float((0.5 * t.spectrum).sum()), 1.0, abs_tol=1e-4)  # float32 file


def test_load_model_table_rejects_bad(tmp_path: Path) -> None:
    np.save(tmp_path / "bad_shape.npy", np.ones((4, 10)))
    raw = np.ones((5, 10))
    raw[1] = 5.0  # 0.5 * sum = 25, not 1
    np.save(tmp_path / "bad_norm.npy", raw)
    for name in ("bad_shape.npy", "bad_norm.npy"):
        with pytest.raises(SchemaError) as e:
            S.load_model_table(tmp_path / name, 50)
        assert e.value.convention == S.CONV_TABLE
