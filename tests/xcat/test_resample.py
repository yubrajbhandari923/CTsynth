"""Box-average resampling conserves attenuation; labels are nearest neighbour; axes are not
permuted (asymmetric shapes)."""

import numpy as np
import pytest
import torch

from ctsynth.xcat.materials import MaterialModel
from ctsynth.xcat.phantom import (
    affine_from_axcodes,
    box_weights,
    make_grid,
    mu_to_hu,
    nn_index,
    resample_phantom,
)


def _model():
    mu = np.zeros(256)
    mu[:4] = [1e-5, 0.02, 0.05, 0.04]
    rho = np.zeros(256)
    rho[:4] = [0.0012, 1.0, 1.5, 0.3]
    return MaterialModel(
        70.0, {}, rho, np.zeros(256), mu * 10, 0.0193, {}, np.arange(256, dtype=np.uint8)
    )


def _vol():
    v = np.zeros((40, 50, 60), np.uint8)  # distinct axis lengths
    v[5:35, 10:40, 8:50] = 1
    v[10:20, 15:30, 20:40] = 2
    v[22:30, 20:25, 30:35] = 3
    return v


@pytest.mark.parametrize("vox", [0.8, 0.6, 1.0, 0.75])
def test_total_attenuation_conserved_for_any_size_ratio(vox):
    v, m = _vol(), _model()
    g = make_grid(v.shape, vox)
    r = resample_phantom(v, g, m, "cpu", chunk=7)
    native = m.mu_mm[v].astype(np.float64).sum() * 0.25**3
    ext = np.array(v.shape) * 0.25
    pad_air = float(m.mu_mm[0]) * (np.prod(np.array(g.shape) * vox) - np.prod(ext))
    assert float(r.mu_mm.double().sum()) * vox**3 - pad_air == pytest.approx(native, rel=1e-6)
    assert np.array_equal(r.histogram[:4], np.bincount(v.ravel(), minlength=256)[:4])


def test_box_weights_rows_and_nn_index():
    w = box_weights(7, 0.8, 20, 0.25)
    assert torch.allclose(w.sum(1)[1:-1], torch.ones(5), atol=1e-6) and float(w.sum(1)[0]) <= 1.0
    i, ok = nn_index(7, 0.8, 20, 0.25)
    assert ok.sum() >= 5 and i.min() >= 0 and i.max() < 20


def test_label_object_stays_where_it_is_no_axis_swap():
    v, m = _vol(), _model()
    g = make_grid(v.shape, 0.5)
    lab = resample_phantom(v, g, m, "cpu", chunk=9).labels.numpy()

    def centre(a, vx, shape):
        return (np.array(np.nonzero(a)).mean(1) + 0.5) * vx - np.array(shape) * vx / 2

    assert np.allclose(centre(v == 2, 0.25, v.shape), centre(lab == 2, 0.5, g.shape), atol=0.5)
    assert lab.shape == g.shape and g.shape[0] != g.shape[1] != g.shape[2]


def test_hu_and_affine():
    assert float(mu_to_hu(torch.tensor(0.0193), 0.0193)) == pytest.approx(0.0, abs=1e-4)
    assert float(mu_to_hu(torch.tensor(0.0), 0.0193)) == pytest.approx(-1000.0)
    a = affine_from_axcodes((10, 20, 30), 0.8, "SLP")
    assert np.allclose(a @ np.array([4.5, 9.5, 14.5, 1]), [0, 0, 0, 1])  # centre voxel -> origin
    assert (
        a[2, 0] == 0.8 and a[0, 1] == -0.8 and a[1, 2] == -0.8
    )  # z -> +S, y -> L (-x), x -> P (-y)
