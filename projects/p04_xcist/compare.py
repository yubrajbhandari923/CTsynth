"""Forward-model mismatch ladder: CTsynth differentiable model vs XCIST on the same phantoms.

`research run exec EXP-... -- uv run python projects/p04_xcist/compare.py DATASET_DIR OUT_DIR`

Reference = XCIST noise-free, single-sample projections (`*_ref` records). Every stage changes ONE
ingredient of our model relative to the previous one, so each delta is attributable. All numbers are
-ln(transmission) differences (dimensionless); rms_obj / rms_all are RMS over rays through the
object (XCIST prep in (0.05, 8)) / all rays. This characterises mismatch; nothing is tuned
to reduce it.

  M1  analytic chords, XCIST mu + spectrum + detector weights -> geometry/sampling floor
  M1v voxel-marched chords instead of analytic                -> truth rasterisation share
  M2  our xraylib kappa tables instead of XCIST (Geant4) mu   -> attenuation-table effect
  M3  detector weights -> ideal energy-integrating (R = E)    -> detector absorption/prefilter
  M4  our spectra_v1 (SpekPy) instead of XCIST spectrum       -> spectrum calibration
M1 shares the transport equation with XCIST by construction, so a small M1 is a CHECK OF THE
ADAPTER AND GEOMETRY, not independent validation of either model's physics.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from ctsynth.paired.phantoms import Disk, chords
from ctsynth.paired.schema import PairedRecord, load_record
from ctsynth.paired.xcist import XcistAcq, fan_rays
from ctsynth.physics import MaterialBasis, ParallelBeamProjector2D, SpectralForwardModel, Spectrum

F64 = torch.float64
NAMES = ("water", "calcium", "iodine")
E_MIN = 10.0  # our attenuation tables start at 10 keV; lower XCIST bins are dropped (reported)


def _acq(rec: PairedRecord) -> XcistAcq:
    g = rec.acquisition["geometry"]
    return XcistAcq(
        n_view=g["n_view"],
        n_col=g["n_col"],
        col_size_mm=g["col_size_mm"],
        sid_mm=g["sid_mm"],
        sdd_mm=g["sdd_mm"],
        col_offset=g["col_offset"],
    )


def voxel_chords(truth: np.ndarray, dx: float, src: np.ndarray, det: np.ndarray) -> np.ndarray:
    """Pixel-constant ray march of volume fractions: (n_mat, n_view, n_col) in mm."""
    n_mat, _, n, _ = truth.shape
    f = truth[:, 0]
    c = (n - 1) / 2
    step = 0.5 * dx
    out = np.zeros((n_mat, det.shape[0], det.shape[1]))
    t = np.arange(0.0, 2 * np.linalg.norm(src, axis=1).max() + 1.0, step)
    for v in range(det.shape[0]):
        u = det[v] - src[v]
        u = u / np.linalg.norm(u, axis=-1, keepdims=True)
        p = src[v][None, None, :] + t[None, :, None] * u[:, None, :]  # (col, t, 2)
        ix = np.rint(p[..., 0] / dx + c).astype(int)
        iy = np.rint(-p[..., 1] / dx + c).astype(int)
        ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        ix, iy = np.where(ok, ix, 0), np.where(ok, iy, 0)
        out[:, v] = (f[:, iy, ix] * ok).sum(axis=-1) * step
    return out


def model_prep(
    a_gcm2: np.ndarray,
    energies: np.ndarray,
    fluence: np.ndarray,
    response: np.ndarray,
    grid: np.ndarray,
    kappa_grid: np.ndarray,
) -> np.ndarray:
    """CTsynth SpectralForwardModel: line integrals (3, v, c) g/cm^2 -> -ln(T) (v, c)."""

    def t(x: object) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=F64)

    basis = MaterialBasis(NAMES, torch.ones(3, dtype=F64), t(grid), t(kappa_grid))
    spec = Spectrum(
        t(energies), t(fluence), torch.ones(len(energies), dtype=F64), t(response)[None]
    )
    proj = ParallelBeamProjector2D(2, 2, 1.0, torch.zeros(1, dtype=F64), 1, 1.0)  # unused by chords
    model = SpectralForwardModel(basis, spec, proj)
    counts = model.expected_counts_from_line_integrals(t(a_gcm2))[0]
    return (-torch.log(counts / spec.unattenuated_counts()[0])).numpy()


def metrics(delta: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    obj = (ref > 0.05) & (ref < 8)
    return {
        "rms_all": float(np.sqrt(np.mean(delta**2))),
        "max_abs": float(np.abs(delta).max()),
        "rms_obj": float(np.sqrt(np.mean(delta[obj] ** 2))),
        "rel_rms_obj": float(np.sqrt(np.mean(delta[obj] ** 2)) / np.sqrt(np.mean(ref[obj] ** 2))),
        "mean_obj": float(delta[obj].mean()),
    }


def tables() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    d = np.load(Path("data/tables/attenuation_v1.npz"))
    order = [list(d["names"]).index(n) for n in NAMES]
    s = np.load(Path("data/tables/spectra_v1.npz"))
    return d["energies_keV"], d["kappa_cm2_per_g"][order], {k: s[k] for k in s.files}


def run_one(ds: Path, pid: str, kvp: int) -> dict[str, object]:
    ref = load_record(ds / f"{pid}__kvp{kvp}_ref")
    path = ds / f"{pid}__kvp{kvp}_ref"
    prep_x = np.load(path / "prep.npy")
    sp = ref.acquisition["spectrum"]
    ev = np.array(sp["energies_keV"])
    keep = ev >= E_MIN
    w_all = np.array(sp["net_fluence"]) * ev * np.array(sp["detector_weights"])
    dropped = float(w_all[~keep].sum() / w_all.sum())
    ev, flu, dw = (
        ev[keep],
        np.array(sp["net_fluence"])[keep],
        np.array(sp["detector_weights"])[keep],
    )
    rho = np.array([m.ref_density_g_cm3 for m in ref.materials])
    xk = np.array([np.array(sp["mu_per_cm"][n])[keep] for n in ("water", "Ca", "I")]) / rho[:, None]
    acq = _acq(ref)
    src, det = fan_rays(acq)
    disks = [Disk(**d) for d in ref.provenance["phantom_disks"]]
    a_an = chords(disks, src, det) / 10.0 * rho[:, None, None]  # g/cm^2
    truth = np.load(path / "truth.npy")
    dx = ref.grid.spacing_mm()[2]
    a_vx = voxel_chords(truth, dx, src, det) / 10.0 * rho[:, None, None]
    tg, tk, ts = tables()
    # our spectra_v1 (1 keV points, each covering [e-0.5, e+0.5)) mapped onto XCIST's energy bins
    # [E-h, E+h) with h = half the grid step (4 keV at 80 kVp, 6 keV at 120 kVp) by EXACT overlap
    # weights, so each photon is counted once. Review of F-035 found a fixed +-3 keV window
    # double-counted photons at 80 kVp. Photons per bin, as our Spectrum expects.
    e_s, f_s = ts["energies_keV"], ts[f"fluence_{kvp}"]
    step = np.diff(np.array(sp["energies_keV"]))
    assert np.allclose(step, step[0]), "XCIST energy grid must be uniform"
    h = step[0] / 2
    ov = lambda e: np.clip(np.minimum(e_s + 0.5, e + h) - np.maximum(e_s - 0.5, e - h), 0, 1)  # noqa: E731
    ours_flu = np.array([(f_s * ov(e)).sum() for e in ev])
    ours_flu_pt = np.array([f_s[(e_s >= e - h) & (e_s < e + h)].sum() for e in ev])  # sensitivity
    ideal = ev.copy()
    stages = {
        "M1_analytic_xcistphys": model_prep(a_an, ev, flu, ev * dw, ev, xk),
        "M1v_voxel_chords": model_prep(a_vx, ev, flu, ev * dw, ev, xk),
        "M2_our_kappa": model_prep(a_an, ev, flu, ev * dw, tg, tk),
        "M3_ideal_detector": model_prep(a_an, ev, flu, ideal, tg, tk),
        "M4_our_spectrum": model_prep(a_an, ev, ours_flu, ideal, tg, tk),
        "M4pt_point_binned_sensitivity": model_prep(a_an, ev, ours_flu_pt, ideal, tg, tk),
    }
    res: dict[str, object] = {"dropped_weight_fraction_below_10keV": dropped}
    for k, p in stages.items():
        res[k] = {"vs_xcist": metrics(p - prep_x, prep_x)}
    res["increments"] = {
        "voxel_rasterisation(M1v-M1)": metrics(
            stages["M1v_voxel_chords"] - stages["M1_analytic_xcistphys"], prep_x
        ),
        "attenuation_tables(M2-M1)": metrics(
            stages["M2_our_kappa"] - stages["M1_analytic_xcistphys"], prep_x
        ),
        "detector_response(M3-M2)": metrics(
            stages["M3_ideal_detector"] - stages["M2_our_kappa"], prep_x
        ),
        "spectrum(M4-M3)": metrics(stages["M4_our_spectrum"] - stages["M3_ideal_detector"], prep_x),
        "spectrum_pointbin(M4pt-M3)": metrics(
            stages["M4pt_point_binned_sensitivity"] - stages["M3_ideal_detector"], prep_x
        ),
    }
    sub = np.load(ds / f"{pid}__kvp{kvp}_sub" / "prep.npy")
    res["xcist_subsampling(sub-ref)"] = metrics(sub - prep_x, prep_x)
    noisy = np.load(ds / f"{pid}__kvp{kvp}_noisy" / "prep.npy")
    d = noisy - prep_x
    obj = (prep_x > 0.05) & (prep_x < 6)
    x = np.exp(prep_x[obj])
    bins = np.quantile(prep_x[obj], np.linspace(0, 1, 9))
    idx = np.digitize(prep_x[obj], bins[1:-1])
    bv = np.array([np.var(d[obj][idx == i]) for i in range(8)])
    bx = np.array([x[idx == i].mean() for i in range(8)])
    a, b = np.polyfit(bx, bv, 1)
    r2 = 1 - np.sum((bv - (a * bx + b)) ** 2) / np.sum((bv - bv.mean()) ** 2)
    res["noise_law"] = {
        "slope_per_exp_prep": float(a),
        "intercept": float(b),
        "r2_binned": float(r2),
        "n_eff_photons_at_unit_transmission": float(1 / a) if a > 0 else None,
        "sd_overall": float(d[obj].std()),
    }
    return res


def main(ds: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    allres = {f"{p}_kvp{k}": run_one(ds, p, k) for p in ("G1", "G2", "G3") for k in (80, 120)}
    (out / "mismatch.json").write_text(json.dumps(allres, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                k: {s: v[s]["vs_xcist"]["rel_rms_obj"] for s in v if s.startswith("M")}
                for k, v in allres.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
