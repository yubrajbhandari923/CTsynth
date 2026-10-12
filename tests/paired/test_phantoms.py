import json
from pathlib import Path

import numpy as np
import pytest

from ctsynth.paired.phantoms import Disk, chords, fractions
from ctsynth.paired.xcist import write_voxel_phantom, xcist_affine


def test_disk_area_and_delta() -> None:
    d = [Disk(0, 0, 40.0, (1.0, 0.0)), Disk(10, 5, 10.0, (-0.2, 0.2))]
    f = fractions(d, 128, 2.0)
    assert f.shape == (2, 1, 128, 128)
    assert f[0].sum() * 4.0 == pytest.approx(np.pi * 40**2 - 0.2 * np.pi * 10**2, rel=0.01)
    assert f[1].sum() * 4.0 == pytest.approx(0.2 * np.pi * 10**2, rel=0.01)
    assert f.sum(0).max() <= 1 + 1e-9


def test_row0_is_plus_y_and_col_is_plus_x() -> None:
    f = fractions([Disk(30.0, 40.0, 6.0, (1.0,))], 64, 2.0)[0, 0]
    iy, ix = np.indices(f.shape)
    cx, cy = (ix * f).sum() / f.sum(), (iy * f).sum() / f.sum()  # centroid, not argmax (plateau)
    assert (cx - 31.5) * 2.0 == pytest.approx(30.0, abs=0.5)
    assert -(cy - 31.5) * 2.0 == pytest.approx(40.0, abs=0.5)  # +y is the upper rows


def test_chord_through_centre_is_diameter() -> None:
    src = np.array([[0.0, 500.0]])
    det = np.array([[[0.0, -400.0], [400.0, -400.0]]])
    c = chords([Disk(0, 0, 50.0, (1.0,))], src, det)
    assert c[0, 0, 0] == pytest.approx(100.0)
    assert c[0, 0, 1] < 100.0


def test_phantom_files_roundtrip(tmp_path: Path) -> None:
    f = fractions([Disk(20, -10, 15, (0.7, 0.3))], 32, 2.0)
    write_voxel_phantom(tmp_path, "t", f, ["water", "bone"], 2.0)
    vp = json.loads((tmp_path / "t.json").read_text())
    assert vp["mat_name"] == ["water", "bone"] and vp["x_offset"] == [16.5, 16.5]
    back = np.fromfile(tmp_path / "t_bone_32x32x1.raw", dtype="<f4").reshape(32, 32)  # x fastest
    assert np.array_equal(back, f[1, 0].astype(np.float32))


def test_rejects_overfull_fractions(tmp_path: Path) -> None:
    f = np.full((2, 1, 8, 8), 0.7)
    with pytest.raises(ValueError, match="sum to <= 1"):
        write_voxel_phantom(tmp_path, "t", f, ["a", "b"], 1.0)


def test_affine_matches_array_convention() -> None:
    a = np.array(xcist_affine(64, 2.0, 16.0))
    assert (a @ np.array([0, 0, 0, 1]))[:2] == pytest.approx(
        [-63.0, 63.0]
    )  # row 0 col 0 = (-x, +y)
    assert (a @ np.array([63, 63, 0, 1]))[:2] == pytest.approx([63.0, -63.0])
