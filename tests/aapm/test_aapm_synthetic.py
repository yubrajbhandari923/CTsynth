"""Synthetic generator: exact schema, two-regime partition, forward model vs reference."""

from pathlib import Path

import aapm_reference as ref
import numpy as np
import pytest

from ctsynth.aapm import schema as S
from ctsynth.aapm import synthetic as Y
from ctsynth.aapm.schema import ModelTable
from ctsynth.aapm.synthetic import SyntheticConfig

TOL = 1e-3  # partition tolerance, same as schema.PARTITION_TOL (F-013 used 1e-3)


def test_config_geometry(cfg: SyntheticConfig) -> None:
    g = cfg.geometry()
    assert (g.n_pix, g.n_views, g.n_det, g.fov_cm) == (16, 16, 24, 18.0)
    assert (g.source_radius_cm, g.sdd_cm) == (50.0, 100.0)


@pytest.mark.parametrize("n_pix", [16, 32])
def test_schema_and_partition(n_pix: int, tables: dict[int, ModelTable]) -> None:
    c = SyntheticConfig(n_cases=5, n_pix=n_pix, n_views=16, n_det=24, seed=1)
    d = Y.generate(c, tables)
    S.validate_dataset(d.phantoms, d.transmission, d.images, d.geometry)
    assert d.phantoms.shape == (5, 3, n_pix, n_pix) and d.phantoms.dtype == np.float32
    assert sorted(d.transmission) == [50, 80] and sorted(d.images) == [50, 80]
    s = d.phantoms.sum(axis=1)
    for k in range(5):
        air, tissue = s[k] <= TOL, np.abs(s[k] - 1) <= TOL
        band = ~air & ~tissue
        assert air.sum() > 0 and tissue.sum() > 0 and band.sum() > 0, f"case {k}: all regimes exist"
        assert air.mean() < 0.6 and tissue.mean() > 0.2  # not degenerate
        assert np.all(d.phantoms[k][:, air] == 0)  # no fourth class: air is all-zero
        assert s[k].max() <= S.SUM_MAX
    assert d.phantoms.min() >= 0 and d.phantoms.max() <= 1


def test_calcifications_and_asymmetry(cfg: SyntheticConfig, tables: dict[int, ModelTable]) -> None:
    d = Y.generate(cfg, tables)
    s = d.phantoms.sum(axis=1)
    for k in range(cfg.n_cases):
        calc = d.phantoms[k, 2]
        idx = np.argwhere(calc > S.CALC_SUPPORT_THRESH)
        assert 1 <= len(idx) <= 3
        for ix, iy in idx:
            assert 0.3 - 1e-6 <= calc[ix, iy] <= 0.8 + 1e-6  # spec range, float32 slack
            assert abs(s[k, ix, iy] - 1) <= TOL  # inside full tissue, sum conserved
    a = d.phantoms[:, 0]
    assert not np.allclose(a, a.transpose(0, 2, 1)), "x/y must be distinguishable"
    assert not np.array_equal(d.phantoms[0], d.phantoms[1])


def test_determinism(cfg: SyntheticConfig, tables: dict[int, ModelTable]) -> None:
    a, b = Y.generate(cfg, tables), Y.generate(cfg, tables)
    assert np.array_equal(a.phantoms, b.phantoms)
    assert np.array_equal(a.transmission[50], b.transmission[50])
    other = Y.generate(SyntheticConfig(4, 16, 16, 24, seed=cfg.seed + 1), tables)
    assert not np.array_equal(a.phantoms, other.phantoms)


def test_line_integrals_match_reference(
    cfg: SyntheticConfig, tables: dict[int, ModelTable]
) -> None:
    d = Y.generate(cfg, tables)
    g = d.geometry
    want = ref.ref_line_integrals(
        d.phantoms, g.n_pix, g.fov_cm, g.n_views, g.n_det, g.source_radius_cm, g.sdd_cm
    )
    got = Y.line_integrals(d.phantoms, g)
    assert got.shape == want.shape == (4, 3, 16, 24)
    # both exact float64 chord geometry; only summation order differs
    assert np.allclose(got, want, rtol=1e-8, atol=1e-10)


def test_transmission_matches_independent_reference(
    cfg: SyntheticConfig, tables: dict[int, ModelTable]
) -> None:
    d = Y.generate(cfg, tables)
    g = d.geometry
    p = ref.ref_line_integrals(
        d.phantoms, g.n_pix, g.fov_cm, g.n_views, g.n_det, g.source_radius_cm, g.sdd_cm
    )
    for kvp in (50, 80):
        want = ref.ref_transmission(p, tables[kvp].spectrum, tables[kvp].mu_per_cm, kvp)
        got = d.transmission[kvp]
        assert got.dtype == np.float32 and got.shape == want.shape == (4, 8, 24)
        # float32 output of a float64 sum over <=158 energies: relative 1e-4 is generous
        assert np.allclose(got, want, rtol=1e-4, atol=1e-6)
        assert np.allclose(Y.simulate_transmission(d.phantoms, g, tables)[kvp], got, rtol=1e-6)
    assert d.transmission[50].mean() < d.transmission[80].mean()  # 50 kVp attenuates more
    assert 0 < d.transmission[50].min() and d.transmission[80].max() <= 1


def test_images_are_spectrum_weighted_mu(
    cfg: SyntheticConfig, tables: dict[int, ModelTable]
) -> None:
    d = Y.generate(cfg, tables)
    for kvp in (50, 80):
        mu_eff = (0.5 * tables[kvp].spectrum * tables[kvp].mu_per_cm).sum(axis=1)  # (3,)
        want = np.einsum("nmxy,m->nxy", d.phantoms.astype(np.float64), mu_eff)
        assert np.allclose(d.images[kvp], want, rtol=1e-5, atol=1e-7)  # float32 rounding


def test_write_synthetic_dir(
    cfg: SyntheticConfig, tables: dict[int, ModelTable], tmp_path: Path
) -> None:
    d = Y.generate(cfg, tables)
    Y.write_synthetic_dir(d, tmp_path)
    for j, name in enumerate(S.MATERIAL_FILE_NAMES):
        a = np.load(tmp_path / f"Phantom_{name}.npy")
        assert a.shape == (4, 16, 16) and a.dtype == np.float32
        assert np.array_equal(a, d.phantoms[:, j])  # canonical order on disk
    assert np.array_equal(np.load(tmp_path / "lowkVpTransmission.npy"), d.transmission[50])
    assert np.array_equal(np.load(tmp_path / "highkVpTransmission.npy"), d.transmission[80])
    assert np.array_equal(np.load(tmp_path / "lowkVpImages.npy"), d.images[50])
    assert np.array_equal(np.load(tmp_path / "highkVpImages.npy"), d.images[80])
