"""Cache formats: quantisation error bound, round trips, random 3D patch reads."""

import numpy as np
import pytest

from ctsynth.xcat.cache import FORMATS, HU_STEP, VolumeReader, hu_to_i16, i16_to_hu, write_hu


def _hu():
    r = np.random.default_rng(0)
    v = (r.normal(0, 30, (20, 33, 41)) + np.linspace(-1000, 900, 41)[None, None]).astype(np.float32)
    return v


def test_int16_quantisation_error_bound_and_clipping():
    v = _hu()
    assert float(np.abs(i16_to_hu(hu_to_i16(v)) - v).max()) <= HU_STEP / 2 + 1e-4
    assert i16_to_hu(hu_to_i16(np.array([1e5, -1e5], np.float32))).tolist() == [
        32767 * HU_STEP,
        -32767 * HU_STEP,
    ]


@pytest.mark.parametrize("fmt", [f for f in FORMATS if f.startswith("npy")])
def test_npy_round_trip_and_patches(tmp_path, fmt):
    v = _hu()
    w = write_hu(tmp_path / "ct", v, fmt)
    rd = VolumeReader(w.path, fmt)
    tol = {"npy_f32": 0, "npy_f16": 1.0, "npy_i16q": HU_STEP / 2 + 1e-4}[fmt]
    assert rd.shape == v.shape and float(np.abs(rd.read_all() - v).max()) <= tol
    p = rd.read_patch((3, 5, 7), (8, 9, 10))
    assert p.shape == (8, 9, 10) and np.allclose(p, rd.read_all()[3:11, 5:14, 7:17])
    assert w.bytes > 0


def test_zarr_round_trip_when_available(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    v = _hu()
    for fmt in ("zarr_raw_i16q", "zarr_lz4_i16q"):
        w = write_hu(tmp_path / fmt, v, fmt, chunks=(8, 16, 16))
        assert float(np.abs(VolumeReader(w.path, fmt).read_all() - v).max()) <= HU_STEP / 2 + 1e-4
