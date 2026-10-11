#!/usr/bin/env python3
"""Generate the frozen toy-experiment tables (attenuation v1, source spectra v1, provenance v1).

Run from the project root:
    uv run --no-project --with xraylib --with spekpy --with numpy python scripts/make_tables.py

Outputs (all under data/tables/):
    attenuation_v1.npz, spectra_v1.npz, provenance_v1.json

Conventions
- Energy grid: 10..150 keV, 1 keV step. Each value is a bin CENTRE; bin n covers [n-0.5, n+0.5).
- kappa: total mass attenuation coefficient including coherent scattering, cm^2/g, evaluated
  at the grid points only (no interpolation is stored).
- Spectra: SpekPy native normalisation (mAs=1, focus-to-detector 100 cm), photons per 1 keV bin
  (integrated over the bin, i.e. the SpekPy photons cm^-2 keV^-1 value times 1 keV summed over
  its two native 0.5 keV bins). Absolute normalisation is NOT verified; consumers must rescale.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata as _md
import io
import json
import pathlib
import platform
import sys
import zipfile

import numpy as np
import spekpy as sp
import xraylib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tables"

ENERGIES_KEV = np.arange(10.0, 150.0 + 0.5, 1.0, dtype=np.float64)  # 141 bin centres
NAMES = ["water", "iodine", "calcium"]
REF_DENSITY_G_CM3 = [1.0, 4.93, 1.55]  # elemental reference densities, NOT tissue densities
KVP_LIST = (80, 120)
ALUMINIUM_FILTER_MM = 2.5
SPEK_DK_KEV = 0.5

# NIST XCOM-based tables, column mu/rho (total WITH coherent scattering), cm^2/g.
# water: values supplied by the task brief, re-fetched and matched on 2026-10-11 from
#   https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
# iodine: fetched 2026-10-11 from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z53.html
# calcium: fetched 2026-10-11 from https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z20.html
NIST_CHECKS = [
    # (name, label for xraylib call, energy keV, NIST value cm^2/g, source url)
    ("water", "H2O", 30.0, 0.3756, "ComTab/water.html"),
    ("water", "H2O", 60.0, 0.2059, "ComTab/water.html"),
    ("water", "H2O", 100.0, 0.1707, "ComTab/water.html"),
    ("iodine", 53, 60.0, 7.579, "ElemTab/z53.html"),
    ("iodine", 53, 100.0, 1.942, "ElemTab/z53.html"),
    ("calcium", 20, 60.0, 0.6578, "ElemTab/z20.html"),
    ("calcium", 20, 100.0, 0.2571, "ElemTab/z20.html"),
]
NIST_RTOL = 0.02


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npz(path: pathlib.Path, **arrays: np.ndarray) -> None:
    """np.savez with fixed ZIP timestamps so identical inputs give byte-identical files."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, buf.getvalue())


def attenuation_table() -> tuple[np.ndarray, dict[str, float]]:
    kappa = np.array(
        [
            [xraylib.CS_Total_CP("H2O", float(e)) for e in ENERGIES_KEV],
            [xraylib.CS_Total(53, float(e)) for e in ENERGIES_KEV],
            [xraylib.CS_Total(20, float(e)) for e in ENERGIES_KEV],
        ],
        dtype=np.float64,
    )
    # Diagnostic for the documented K-edge limitation (I K-edge 33.17 keV).
    edge_info = {
        "iodine_kappa_33keV": float(xraylib.CS_Total(53, 33.0)),
        "iodine_kappa_34keV": float(xraylib.CS_Total(53, 34.0)),
        "iodine_K_edge_keV_xraylib": float(xraylib.EdgeEnergy(53, xraylib.K_SHELL)),
    }
    return kappa, edge_info


def spectrum_1kev(kvp: int) -> tuple[np.ndarray, dict[str, float]]:
    """SpekPy tungsten-anode spectrum, 2.5 mm Al total filtration, rebinned to 1 keV bins."""
    spk_obj = sp.Spek(
        kvp=kvp,
        targ="W",
        th=12.0,
        dk=SPEK_DK_KEV,
        physics="casim",
        mu_data_source="pene",
        z=100.0,
        mas=1.0,
    )
    spk_obj.filter("Al", ALUMINIUM_FILTER_MM)  # SpekPy filter thickness unit: mm (checked)
    centres, diff = spk_obj.get_spectrum(edges=False, flu=True, diff=True)
    centres = np.asarray(centres, dtype=np.float64)
    if not np.allclose(np.diff(centres), SPEK_DK_KEV):
        raise RuntimeError("unexpected SpekPy native bin spacing")
    per_bin = np.asarray(diff, dtype=np.float64) * SPEK_DK_KEV  # photons cm^-2 per native bin

    # 1 keV bin n covers [n-0.5, n+0.5). Native 0.5 keV bins have edges at multiples of 0.5,
    # so each lies wholly inside one 1 keV bin: n = floor(centre + 0.5).
    n = np.floor(centres + 0.5).astype(int)
    in_grid = (n >= ENERGIES_KEV[0].astype(int)) & (n <= ENERGIES_KEV[-1].astype(int))
    rebinned = np.bincount(
        n[in_grid] - int(ENERGIES_KEV[0]), weights=per_bin[in_grid], minlength=ENERGIES_KEV.size
    ).astype(np.float64)

    native_total = float(per_bin.sum())
    native_in_grid = float(per_bin[in_grid].sum())
    stats = {
        "native_bins": int(centres.size),
        "native_first_centre_keV": float(centres[0]),
        "native_last_centre_keV": float(centres[-1]),
        "native_total_photons_cm2": native_total,
        "native_in_grid_total_photons_cm2": native_in_grid,
        "rebinned_total_photons_cm2": float(rebinned.sum()),
        "dropped_below_9p5keV_fraction": float(per_bin[n < 10].sum() / native_total),
        "dropped_above_150p5keV_fraction": float(per_bin[n > 150].sum() / native_total),
    }
    rel = abs(stats["rebinned_total_photons_cm2"] - native_in_grid) / native_in_grid
    if rel > 1e-9:
        raise RuntimeError(f"rebinning not conserving fluence: rel diff {rel:.3e}")
    stats["conservation_rel_diff"] = rel
    return rebinned, stats


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "xraylib": _md.version("xraylib"),
        "spekpy": _md.version("spekpy"),
        "numpy": _md.version("numpy"),
    }


def first_lines(text: str | None, n: int = 6) -> list[str]:
    return [] if not text else [ln.strip() for ln in text.strip().splitlines()[:n] if ln.strip()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    kappa, edge_info = attenuation_table()
    ref_density = np.asarray(REF_DENSITY_G_CM3, dtype=np.float64)
    att_path = OUT / "attenuation_v1.npz"
    save_npz(
        att_path,
        energies_keV=ENERGIES_KEV,
        names=np.array(NAMES),
        kappa_cm2_per_g=kappa,
        ref_density_g_cm3=ref_density,
    )

    fluences: dict[str, np.ndarray] = {}
    spec_stats: dict[str, dict[str, float]] = {}
    for kvp in KVP_LIST:
        fluences[f"fluence_{kvp}"], spec_stats[str(kvp)] = spectrum_1kev(kvp)
    spec_path = OUT / "spectra_v1.npz"
    save_npz(spec_path, energies_keV=ENERGIES_KEV, **fluences)

    grid_index = {float(e): i for i, e in enumerate(ENERGIES_KEV)}
    nist_results = []
    for name, ident, energy, nist, source in NIST_CHECKS:
        if name == "water":
            value = float(xraylib.CS_Total_CP(ident, energy))
        else:
            value = float(xraylib.CS_Total(ident, energy))
        rel = abs(value - nist) / nist
        nist_results.append(
            {
                "material": name,
                "energy_keV": energy,
                "nist_mu_over_rho_cm2_per_g": nist,
                "nist_source": "https://physics.nist.gov/PhysRefData/XrayMassCoef/" + source,
                "xraylib_cm2_per_g": value,
                "xraylib_table_value_at_grid_index_matches": bool(
                    np.isclose(kappa[NAMES.index(name), grid_index[energy]], value, rtol=1e-12)
                ),
                "rel_diff": rel,
                "rtol": NIST_RTOL,
                "pass": bool(rel <= NIST_RTOL),
            }
        )

    spek_doc = sp.Spek.__init__.__doc__ or ""
    mu_src_lines = [
        ln.strip() for ln in spek_doc.splitlines() if "mu_data_source" in ln or "'pene'" in ln
    ]
    xl_doc = xraylib.CS_Total.__doc__ or ""

    provenance = {
        "table_version": "v1",
        "generated_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {
            "script": "scripts/make_tables.py",
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "command": "uv run --no-project --with xraylib --with spekpy --with numpy python scripts/make_tables.py",
        },
        "libraries": {
            **library_versions(),
            "spekpy_dependency_note": "spekpy requires scipy and matplotlib; resolved by uv (versions not pinned by this script)",
        },
        "database_descriptions": {
            "xraylib": {
                "installed_package_statement": first_lines(xraylib.__doc__),
                "installed_metadata_summary": _md.metadata("xraylib")["Summary"],
                "CS_Total_signature_doc": xl_doc.strip().splitlines()[0] if xl_doc else "",
                "data_lineage": "UNVERIFIED: the installed package does not state which tabulation underlies CS_Total/CS_Total_CP; its documentation is an external wiki that was not read offline.",
            },
            "spekpy": {
                "installed_metadata_summary": _md.metadata("spekpy")["Summary"],
                "mu_data_source_doc": mu_src_lines,
                "chosen": "physics='casim' (SpekPy v2 casim model), mu_data_source='pene' (SpekPy default for casim)",
                "data_lineage": "UNVERIFIED: the bundled files are named pene_mu.dat and nist_mu.dat; their original source was not stated in the installed docs.",
                "validation_reference": "Bujila R, Omar A, Poludniowski G. Phys Med 2020;75:44-54 (cited in the installed METADATA; not re-read)",
            },
        },
        "spectra": {
            "model": "SpekPy tungsten reflection target, 12 deg anode angle, kVp 80 and 120, mas=1, focus-to-detector 100 cm, native SpekPy defaults otherwise",
            "filtration": f"{ALUMINIUM_FILTER_MM} mm Al total (SpekPy filter thickness unit is mm; checked against xraylib Al mu/rho at 30 keV)",
            "inherent_filtration": "none added (SpekPy default filtration list is empty); real tube window/oil not modelled",
            "native_bin_width_keV": SPEK_DK_KEV,
            "rebinning": "native 0.5 keV bins are summed into 1 keV bins; bin n covers [n-0.5, n+0.5) and holds native bins with centre in that range; values are photons per 1 keV bin",
            "normalisation": "SpekPy native, as returned by get_spectrum(edges=False, flu=True, diff=True) times 0.5 keV. Absolute normalisation vs a physical air kerma or fluence: UNVERIFIED. Consumers must rescale to a target count.",
            "per_kvp_stats": spec_stats,
        },
        "attenuation": {
            "quantity": "total mass attenuation coefficient incl. coherent scattering, cm^2/g, at grid points",
            "water": "xraylib CS_Total_CP('H2O', E): compound composition H2O, density 1.0 g/cm^3 reference",
            "iodine": "xraylib CS_Total(53, E); ref density 4.93 g/cm^3 (solid I2, elemental)",
            "calcium": "xraylib CS_Total(20, E); ref density 1.55 g/cm^3 (Ca metal, elemental)",
            "k_edge_limitation_diagnostics": edge_info,
        },
        "nist_spot_checks": {
            "rtol": NIST_RTOL,
            "reason_for_rtol": "NIST and xraylib tabulations differ in data lineage; 2% is a loose consistency bound, not a precision claim",
            "results": nist_results,
            "skipped": [],
            "all_pass": all(r["pass"] for r in nist_results),
        },
        "known_limitations": [
            "The 1 keV grid under-resolves the iodine K-edge at 33.17 keV. kappa is stored only at grid points; any interpolation between 33 and 34 keV straddles the edge and does not reproduce the jump. Use the diagnostics above.",
            "Ref densities are elemental reference values (I2 solid 4.93, Ca metal 1.55), not tissue-mixture densities.",
            "Water kappa is a single compound H2O at unit density; no tissue composition is modelled.",
            "Spectra are SpekPy native normalisation and not verified against a physical reference. Rescale before use.",
            "Spectrum mass below 9.5 keV and above 150.5 keV is outside the grid and is dropped; fractions are in spectra.per_kvp_stats.",
            "Spectra assume SpekPy's characteristic W lines (default char=True) and the 12 deg reflection geometry; no heel effect or detector response.",
            "Only 2.5 mm Al filtration is modelled; no inherent tube filtration.",
            "Data lineage of xraylib and SpekPy attenuation tables is UNVERIFIED (see database_descriptions).",
        ],
        "sha256": {
            "attenuation_v1.npz": sha256_file(att_path),
            "spectra_v1.npz": sha256_file(spec_path),
        },
        "platform": platform.platform(),
        "python_executable_version": sys.version.split()[0],
    }
    (OUT / "provenance_v1.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "written": [str(att_path), str(spec_path), str(OUT / "provenance_v1.json")],
                "nist_all_pass": provenance["nist_spot_checks"]["all_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
