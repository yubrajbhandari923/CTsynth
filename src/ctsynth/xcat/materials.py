"""XCAT material tables -> linear attenuation at a chosen energy, plus organ-group labels.

mu(r, E) = rho(r) * sum_i w_i (mu/rho)_i(E). Units: rho g/cm^3, (mu/rho) cm^2/g, mu cm^-1 (LEAP
wants mm^-1: divide by 10).
Element (mu/rho) are xraylib CS_Total (incl. coherent scattering) stored in
data/tables/xcat_element_attenuation_v1.json.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

TABLE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "tables" / "xcat_element_attenuation_v1.json"
)
Array = NDArray[np.float64]


@dataclass(frozen=True)
class Material:
    """One row of a TrueCT material table."""

    organ_id: int
    name: str
    density: float  # g/cm^3
    mc_id: int
    mass_fractions: dict[int, float]  # atomic number -> mass fraction (rows may sum to 0.96-1.0)


def load_materials(xlsx: str | Path) -> dict[int, Material]:
    """Organ ID -> Material from a TrueCT `challenge_materials_*.xlsx` (needs openpyxl)."""
    import openpyxl

    ws = openpyxl.load_workbook(xlsx, data_only=True).active
    if ws is None:
        raise ValueError(f"{xlsx}: workbook has no active sheet")
    rows: list[tuple[Any, ...]] = list(ws.iter_rows(values_only=True))
    hdr = rows[0]
    col = {h: i for i, h in enumerate(hdr)}
    zs = [h for h in hdr if isinstance(h, int)]
    out: dict[int, Material] = {}
    for r in rows[1:]:
        if r[0] is None:
            continue
        oid = int(r[col["Organ ID"]])
        out[oid] = Material(
            oid,
            str(r[0]),
            float(r[col["Density"]]),
            int(r[col["MC ID"]]),
            {z: float(r[col[z]] or 0.0) for z in zs},
        )
    return out


class ElementTable:
    """Total mass attenuation (cm^2/g) per element on a 0.5 keV grid, log-log interpolated (avoid K
    edges)."""

    def __init__(self, path: str | Path = TABLE_PATH) -> None:
        d = json.loads(Path(path).read_text())
        self.energies = np.asarray(d["energies_keV"], dtype=np.float64)
        self._k = {int(z): np.asarray(v, dtype=np.float64) for z, v in d["elements"].items()}
        self._water = np.asarray(d["water_H2O_CS_Total_CP"], dtype=np.float64)

    def _interp(self, y: Array, e_kev: float) -> float:
        if not (self.energies[0] <= e_kev <= self.energies[-1]):
            raise ValueError(
                f"energy {e_kev} keV outside the table {self.energies[0]}-{self.energies[-1]} keV"
            )
        return float(np.exp(np.interp(math.log(e_kev), np.log(self.energies), np.log(y))))

    def kappa(self, z: int, e_kev: float) -> float:
        return self._interp(self._k[z], e_kev)

    def kappa_water(self, e_kev: float) -> float:
        return self._interp(self._water, e_kev)


def organ_group(name: str) -> str:
    """Group name = xlsx name without `_trabecular` / `_cortical`; no other anatomy is inferred."""
    n = name.lower()
    for suf in ("_trabecular", "_cortical"):
        n = n.removesuffix(suf)
    return n


def group_category(name: str) -> str:
    if name == "air":
        return "air"
    if name.startswith("insert"):
        return "phantom_insert"
    if "lesion" in name:
        return "lesion"
    if name == "emphysema":
        return "pathology"
    return "tissue_named_in_xlsx"


@dataclass
class MaterialModel:
    energy_kev: float
    materials: dict[int, Material]
    rho: Array  # (256,) g/cm^3, 0 for unmapped labels
    kappa: Array  # (256,) cm^2/g at energy_kev
    mu_cm: Array  # (256,) cm^-1
    mu_water_mm: float  # mm^-1 (water, rho = 1 g/cm^3)
    groups: dict[int, dict[str, object]] = field(default_factory=dict)
    organ_to_group: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(256, np.uint8))

    @property
    def mu_mm(self) -> NDArray[np.float32]:
        return (self.mu_cm / 10.0).astype(np.float32)


def build_material_model(
    xlsx: str | Path, energy_kev: float, elements: ElementTable | None = None
) -> MaterialModel:
    elements = elements or ElementTable()
    mats = load_materials(xlsx)
    rho, kap, mu = np.zeros(256), np.zeros(256), np.zeros(256)
    for oid, m in mats.items():
        sw = sum(
            m.mass_fractions.values()
        )  # normalised; DukeSim's own table does the same (checked in T-077)
        kap[oid] = sum(w * elements.kappa(z, energy_kev) for z, w in m.mass_fractions.items()) / sw
        rho[oid] = m.density
        mu[oid] = kap[oid] * rho[oid]
    names = sorted(
        {organ_group(m.name) for m in mats.values()},
        key=lambda n: min(o for o, m in mats.items() if organ_group(m.name) == n),
    )
    gid = {"air": 0}
    for n in names:
        if n != "air":
            gid[n] = len(gid)
    o2g = np.zeros(256, np.uint8)
    groups: dict[int, dict[str, object]] = {
        0: {"name": "air", "organ_ids": [], "member_names": ["air"], "category": "air"}
    }
    for oid, m in sorted(mats.items()):
        g = gid[organ_group(m.name)]
        o2g[oid] = g
        e = groups.setdefault(
            g,
            {
                "name": organ_group(m.name),
                "organ_ids": [],
                "member_names": [],
                "category": group_category(organ_group(m.name)),
            },
        )
        e["organ_ids"].append(oid)  # type: ignore[attr-defined]
        if m.name not in e["member_names"]:  # type: ignore[operator]
            e["member_names"].append(m.name)  # type: ignore[attr-defined]
    return MaterialModel(
        energy_kev, mats, rho, kap, mu, elements.kappa_water(energy_kev) / 10.0, groups, o2g
    )


_IN_RE = re.compile(r"^(\d+)\s+(\S+)\s+(\d+)\s+([0-9.eE+-]+)")


def crosscheck_dukesim(
    model: MaterialModel, lac_bin: str | Path | None, conversion_in: str | Path | None
) -> dict[str, object]:
    """Independent checks: our per-organ mu vs DukeSim's `lin_att_120kV_121materials.bin` (rows =
    organ ID, columns = 1..120 keV, cm^-1;
    integer energies only) and the xlsx vs the `.in` conversion table (name, density, MC id)."""
    out: dict[str, object] = {}
    e = model.energy_kev
    if (
        lac_bin is not None
        and Path(lac_bin).exists()
        and abs(e - round(e)) < 1e-9
        and 1 <= round(e) <= 120
    ):
        tab = np.fromfile(lac_bin, dtype=np.float32).reshape(121, 120)
        ids = sorted(model.materials)
        ref = tab[ids, round(e) - 1].astype(np.float64)
        rel = np.abs(model.mu_cm[ids] - ref) / np.maximum(ref, 1e-12)
        out["dukesim_lac_bin"] = {
            "max_rel_diff": float(rel.max()),
            "worst_organ_id": int(ids[int(rel.argmax())]),
            "n_organs": len(ids),
        }
    if conversion_in is not None and Path(conversion_in).exists():
        cross = {}
        for ln in Path(conversion_in).read_text().splitlines():
            m = _IN_RE.match(ln)
            if m:
                cross[int(m[1])] = (m[2], float(m[4]), int(m[3]))
        bad = [
            k
            for k, v in model.materials.items()
            if k not in cross
            or cross[k][0] != v.name
            or abs(cross[k][1] - v.density) > 1e-9
            or cross[k][2] != v.mc_id
        ]
        out["xlsx_vs_conversion_table_in"] = {"n_mismatch": len(bad), "mismatched_ids": bad}
    return out
