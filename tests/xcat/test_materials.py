"""Material model: mu = rho * sum w_i kappa_i(E); units; grouping; cross-check with CTsynth's
frozen water table."""

import numpy as np
import openpyxl
import pytest

from ctsynth.xcat.materials import (
    ElementTable,
    build_material_model,
    crosscheck_dukesim,
    organ_group,
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
    "No. of elements with non-zero mass fractions",
    "Organ ID",
    "MC ID",
]


def _xlsx(tmp_path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(HEADER)
    for r in rows:
        ws.append(r)
    p = tmp_path / "mat.xlsx"
    wb.save(p)
    return p


def _row(name, h, o, rho, oid, mc):
    comp = {1: h, 8: o}
    return [name] + [comp.get(z, 0.0) for z in HEADER[1:14]] + [rho, 2, oid, mc]


def test_water_matches_frozen_ctsynth_table_and_tabulated_value():
    et = ElementTable()
    frozen = np.load("data/tables/attenuation_v1.npz", allow_pickle=True)
    i, e = int(np.flatnonzero(frozen["names"] == "water")[0]), frozen["energies_keV"]
    k70 = float(np.interp(70.0, e, frozen["kappa_cm2_per_g"][i]))
    assert et.kappa_water(70.0) == pytest.approx(
        k70, rel=2e-3
    )  # xraylib H2O in both; the frozen grid is 1 keV, ours 0.5 keV
    assert et.kappa_water(70.0) == pytest.approx(
        0.1929, abs=5e-4
    )  # NIST XCOM water at 70 keV is 0.1929 cm^2/g


def test_mu_units_and_density_scaling(tmp_path):
    p = _xlsx(
        tmp_path,
        [
            _row("air", 0.0, 1.0, 0.001205, 0, 1),
            _row("body", 0.112, 0.888, 1.0, 1, 2),
            _row("dense", 0.112, 0.888, 2.0, 2, 3),
        ],
    )
    m = build_material_model(p, 70.0)
    assert m.mu_cm[1] == pytest.approx(m.kappa[1] * 1.0)
    assert m.mu_cm[2] == pytest.approx(2.0 * m.mu_cm[1])  # same composition, double density
    assert m.mu_mm[1] == pytest.approx(m.mu_cm[1] / 10.0, rel=1e-6)
    assert m.mu_water_mm == pytest.approx(0.019288, rel=1e-3)
    assert 1000 * (m.mu_mm[1] / m.mu_water_mm - 1) == pytest.approx(
        0.0, abs=1.5
    )  # H/O 0.112/0.888 is water
    assert m.mu_cm[0] < 1e-3  # air
    assert m.rho[200] == 0 and m.mu_cm[200] == 0  # unmapped label


def test_grouping_merges_only_by_name():
    assert organ_group("spine_trabecular") == organ_group("spine_cortical") == "spine"
    assert organ_group("lung") != organ_group("lung_parenchyma")


def test_groups_and_energy_range(tmp_path):
    p = _xlsx(
        tmp_path,
        [
            _row("air", 0, 1, 0.0012, 0, 1),
            _row("spine_cortical", 0.1, 0.9, 1.4, 9, 7),
            _row("spine_trabecular", 0.1, 0.9, 1.0, 39, 7),
            _row("air", 0, 1, 0.0012, 70, 1),
        ],
    )
    m = build_material_model(p, 60.0)
    assert (
        m.organ_to_group[9] == m.organ_to_group[39] != 0
        and m.organ_to_group[0] == m.organ_to_group[70] == 0
    )
    with pytest.raises(ValueError):
        build_material_model(p, 400.0)
    assert crosscheck_dukesim(m, None, None) == {}
