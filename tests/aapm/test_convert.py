import numpy as np
import pytest

from ctsynth.aapm.convert import aapm_to_paired, paired_to_aapm

M, NX, NY = 3, 5, 7  # all different, so any axis mix-up changes the shape or the values


@pytest.fixture
def maps() -> np.ndarray:
    # value encodes (m, ix, iy) uniquely
    m, ix, iy = np.meshgrid(np.arange(M), np.arange(NX), np.arange(NY), indexing="ij")
    return (m * 10_000 + ix * 100 + iy).astype(np.float32)


def test_shape_and_element_mapping(maps: np.ndarray) -> None:
    t = aapm_to_paired(maps)
    assert t.shape == (M, 1, NY, NX)  # (material, iz, iy, ix)
    assert t.dtype == np.float32
    for m, ix, iy in [(0, 0, 0), (1, 4, 2), (2, 3, 6), (0, 4, 6)]:
        assert t[m, 0, iy, ix] == maps[m, ix, iy]


def test_round_trip_exact_and_contiguous(maps: np.ndarray) -> None:
    t = aapm_to_paired(maps)
    assert t.flags.c_contiguous
    back = paired_to_aapm(t)
    assert back.shape == maps.shape and back.flags.c_contiguous
    assert np.array_equal(back, maps)


def test_batch_axis_passes_through(maps: np.ndarray) -> None:
    batch = np.stack([maps, maps + 1])
    t = aapm_to_paired(batch)
    assert t.shape == (2, M, 1, NY, NX)
    assert np.array_equal(t[1], aapm_to_paired(maps + 1))
    assert np.array_equal(paired_to_aapm(t), batch)


def test_marker_pixel_lands_where_the_paired_convention_says() -> None:
    a = np.zeros((1, NX, NY), np.float32)
    a[0, 4, 1] = 1.0  # ix=4, iy=1
    t = aapm_to_paired(a)
    assert np.argwhere(t == 1).tolist() == [[0, 0, 1, 4]]  # (m, iz, iy, ix)


def test_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        aapm_to_paired(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="nz == 1"):
        paired_to_aapm(np.zeros((M, 2, NY, NX)))
