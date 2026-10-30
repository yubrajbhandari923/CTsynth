"""Physical concentration targets: composition basis, closure to the simulator's mu, partial-volume
mixing, nonnegativity, quantisation and axis alignment (asymmetric shapes)."""

import numpy as np
import openpyxl
import pytest
import torch

from ctsynth.xcat.materials import build_material_model, load_materials
from ctsynth.xcat.phantom import make_grid, resample_phantom
from ctsynth.xcat.physical import (
    CONC_STEP,
    build_basis,
    conc_to_u16,
    density_lut,
    dominant_labels,
    mu_mm_from_concentration,
    resample_concentrations,
    u16_to_conc,
)

HEADER = [
    "Name",
    1,
    6,
    7,
    8,
    11,
    12,
    15,
    16,
    17,
    19,
    20,
    26,
    53,
    "Density",
    "n",
    "Organ ID",
    "MC ID",
]
# name, {Z: w}, density scale, organ id, mc id. Organs 2 and 3 share a composition (like bone).
ROWS = [
    ("air", {7: 0.76, 8: 0.24}, 0.0012, 0, 1),
    ("soft", {1: 0.10, 6: 0.14, 7: 0.03, 8: 0.73}, 1.04, 1, 2),
    ("bone", {1: 0.03, 6: 0.15, 8: 0.45, 15: 0.10, 20: 0.27}, 1.5, 2, 3),
    ("bone2", {1: 0.03, 6: 0.15, 8: 0.45, 15: 0.10, 20: 0.27}, 1.2, 3, 3),
    ("lung", {1: 0.10, 6: 0.10, 7: 0.03, 8: 0.77}, 0.3, 4, 4),
]


def _xlsx(tmp_path, name="t.xlsx", rho_scale=1.0):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(HEADER)
    for nm, comp, rho, oid, mc in ROWS:
        ws.append([nm, *[comp.get(z, 0) for z in HEADER[1:14]], rho * rho_scale, 0, oid, mc])
    p = tmp_path / name
    wb.save(p)
    return p


def _vol():
    v = np.zeros((40, 50, 60), np.uint8)  # distinct axis lengths
    v[5:35, 10:40, 8:50] = 1
    v[10:20, 15:30, 20:40] = 2
    v[22:30, 20:25, 30:35] = 3
    v[12:18, 32:38, 12:18] = 4
    return v


def test_basis_merges_identical_compositions_and_rejects_patient_drift(tmp_path):
    t1 = load_materials(_xlsx(tmp_path))
    t2 = load_materials(_xlsx(tmp_path, "u.xlsx", rho_scale=1.03))  # density differs: allowed
    b = build_basis([t1, t2])
    assert b.k == 4 and b.organ_to_channel[2] == b.organ_to_channel[3]
    assert b.organ_to_channel[255] == -1 and np.allclose(b.fractions.sum(1), 1.0)
    b.check(t2)
    bad = dict(t2)
    bad[1] = type(bad[1])(1, "soft", 1.0, 2, {1: 0.2, 8: 0.8})
    with pytest.raises(ValueError):
        build_basis([t1, bad])
    with pytest.raises(ValueError):
        b.check(bad)


@pytest.mark.parametrize("vox", [0.8, 0.6, 0.75])
@pytest.mark.parametrize("e_kev", [70.0, 80.0])
def test_closure_to_simulator_mu_and_density(tmp_path, vox, e_kev):
    x = _xlsx(tmp_path)
    mats = load_materials(x)
    basis = build_basis([mats])
    v = _vol()
    g = make_grid(v.shape, vox)
    c = resample_concentrations(v, g, basis, density_lut(mats), "cpu", chunk=7)
    ref = resample_phantom(v, g, build_material_model(x, e_kev), "cpu", chunk=7)
    kap = torch.from_numpy(basis.kappa_cm2g(e_kev))
    assert c.shape == (basis.k, *g.shape) and float(c.min()) >= 0.0
    assert torch.allclose(c.sum(0), ref.density, atol=1e-5)
    mu = mu_mm_from_concentration(c, kap)
    assert float((mu - ref.mu_mm).abs().max()) < 1e-6 * float(ref.mu_mm.max())


def test_partial_volume_is_occupancy_times_density_not_label(tmp_path):
    mats = load_materials(_xlsx(tmp_path))
    basis = build_basis([mats])
    v = np.zeros((8, 8, 8), np.uint8)
    v[:, :, :4] = 2  # half bone (1.5 g/cm3), half air
    g = make_grid(v.shape, 0.5, pad_vox=0)  # 4^3 output voxels of 0.5 mm, native 0.25 mm
    c = resample_concentrations(v, g, basis, density_lut(mats), "cpu")
    cb, ca = basis.organ_to_channel[2], basis.organ_to_channel[0]
    # output voxel x=1 covers native x 2..3 (bone) and x=2 covers 4..5 (air); no mixed voxel here
    assert float(c[cb, 0, 0, 0]) == pytest.approx(1.5, abs=1e-6) and float(c[cb, 0, 0, 2]) == 0.0
    assert float(c[ca, 0, 0, 2]) == pytest.approx(0.0012, abs=1e-6)
    v2 = np.zeros(
        (6, 6, 6), np.uint8
    )  # 1.5 mm cube; 0.375 mm voxels: output x=2 straddles bone|air
    v2[:, :, :4] = 2
    g2 = make_grid(v2.shape, 0.375, pad_vox=0)
    c2 = resample_concentrations(v2, g2, basis, density_lut(mats), "cpu")
    # voxel 1 covers [0.375, 0.75) mm: pure bone (bone fills x < 1.0 mm)
    assert float(c2[cb, 0, 0, 1]) == pytest.approx(1.5, abs=1e-6)
    # voxel 2 covers [0.75, 1.125) mm: bone until 1.0 mm (2/3 of the voxel), air after
    assert float(c2[cb, 0, 0, 2]) == pytest.approx(1.5 * (0.25 / 0.375), rel=1e-5)
    assert float(c2[ca, 0, 0, 2]) == pytest.approx(0.0012 * (0.125 / 0.375), rel=1e-4)


def test_axes_not_permuted(tmp_path):
    mats = load_materials(_xlsx(tmp_path))
    basis = build_basis([mats])
    v = _vol()
    g = make_grid(v.shape, 0.5)
    c = resample_concentrations(v, g, basis, density_lut(mats), "cpu", chunk=9)
    lab = dominant_labels(v, g, basis, "cpu")
    cl = basis.organ_to_channel[4]

    def centre_mm(a, vox, shape):
        return (np.array(np.nonzero(a)).mean(1) + 0.5) * vox - np.array(shape) * vox / 2

    mask = (c[cl] > 0.15).numpy()
    assert np.allclose(centre_mm(mask, 0.5, g.shape), centre_mm(v == 4, 0.25, v.shape), atol=0.5)
    assert (lab == cl).sum() > 0 and lab.shape == g.shape and g.shape[0] != g.shape[1] != g.shape[2]


def test_uint16_roundtrip_error_bound():
    c = np.random.default_rng(0).uniform(0, 1.4, 10000).astype(np.float32)
    err = np.abs(u16_to_conc(conc_to_u16(c)) - c).max()
    assert err <= CONC_STEP / 2 + 1e-7
    assert conc_to_u16(np.array([-0.1, 0.0], np.float32)).tolist() == [0, 0]
