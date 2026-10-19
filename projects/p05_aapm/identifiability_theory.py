"""Deterministic identifiability analysis of the AAPM acquisition (no data, no training).

Per ray, unknown line integrals P=(Pa,Pf,Pc) [cm of each material]; measurements are
polychromatic transmissions T_k = sum_E w_k(E) exp(-sum_m mu_m(E) P_m) (kit model, I0=1).
Tissue rays obey the partition constraint Pa+Pf+Pc = L (known chord), leaving 2 unknowns
(Pa,Pf), with Pc = L-Pa-Pf. We report, for acquisition options and composition/path length:
  * Jacobian of -log T w.r.t. (Pa,Pf) at fixed L: rank, singular values;
  * Fisher information with Poisson noise (var(-log T_k) = 1/(N_k T_k)) and the Cramer-Rao
    std of the null-direction coordinate t (P shift along n, in cm of path = fraction * L);
  * the same WITHOUT the partition constraint (3 unknowns), smallest eigenvalues.
Acquisitions: SECT-80, SECT-50, DECT (50+80, equal photons, kit), PCCT (80 kVp spectrum split
into B ideal energy bins; B in 2,4,8,all), PCCT+DECT. Total photon number held equal.
"""

import json
import os
from pathlib import Path

import numpy as np

from ctsynth.aapm import schema as S

NULL = np.array([-0.723, 0.691, -0.008])
NULL /= np.linalg.norm(NULL)
N_TOTAL = 2e5  # photons per ray, shared across all measurements of an acquisition


def load():
    tab = Path("data/aapm_kit_sample")
    return {k: S.load_model_table(tab / f"model_data_{k}kVp.npy", k) for k in S.KVPS}


def measurements(tables, kind):
    """list of (energies, weights w_k(E) summing to relative photon share, mu (3,E))."""

    def band(kvp, lo, hi):
        t = tables[kvp]
        m = (t.energies_kev >= lo) & (t.energies_kev < hi)
        w = np.where(m, 0.5 * t.spectrum, 0.0)
        return t.energies_kev, w, t.mu_per_cm

    t80, t50 = tables[80], tables[50]
    full = lambda t: (t.energies_kev, 0.5 * t.spectrum, t.mu_per_cm)  # noqa: E731
    if kind == "SECT-80":
        return [full(t80)]
    if kind == "SECT-50":
        return [full(t50)]
    if kind == "DECT":
        return [full(t50), full(t80)]
    if kind.startswith("PCCT"):
        B = int(kind.split("-")[1])  # equal-photon bins of the 80 kVp spectrum
        c = np.cumsum(0.5 * t80.spectrum)
        edges = np.interp(
            np.linspace(0, 1, B + 1),
            np.r_[0, c],
            np.r_[t80.energies_kev[0] - 0.25, t80.energies_kev],
        )
        return [band(80, edges[i], edges[i + 1] + (1e-9 if i == B - 1 else 0)) for i in range(B)]
    if kind.startswith("DECT+PCCT"):
        raise ValueError
    raise ValueError(kind)


def transmission(meas, P):
    return np.array([(w * np.exp(-(mu.T @ P))).sum() for (_, w, mu) in meas])


def jac_logT(meas, P):
    """d(-log T_k)/dP (K,3) and T (K,)."""
    J, T = [], []
    for _, w, mu in meas:
        e = w * np.exp(-(mu.T @ P))
        t = e.sum()
        J.append((mu @ e) / t)  # -d log t / dP = sum e mu / t
        T.append(t)
    return np.array(J), np.array(T)


def analyse(meas, P, L, photon_shared):
    J, T = jac_logT(meas, P)
    # Poisson: var(-log T_k) = 1/(N_k T_k). Separate-kVp acquisitions split the photon budget
    # equally; PCCT bins are disjoint parts of ONE beam of N_TOTAL photons (T_k already carries
    # the bin's spectrum share), so each bin sees the full budget.
    f = 1.0 if photon_shared else 1.0 / len(meas)
    var = 1.0 / (N_TOTAL * f * T)
    W = 1.0 / var
    # reduced (partition) parametrisation: P = (Pa, Pf, L-Pa-Pf)
    D = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])  # dP/d(Pa,Pf)
    Jr = J @ D
    Fr = Jr.T @ (W[:, None] * Jr)
    Ff = J.T @ (W[:, None] * J)
    sv = np.linalg.svd(Jr, compute_uv=False)
    ev_r = np.linalg.eigvalsh(Fr)
    ev_f = np.linalg.eigvalsh(Ff)
    # null-direction coordinate t: P += t*n  (n has ~zero sum: stays within the partition)
    nr = np.array([-1.0, 1.0]) / np.sqrt(2.0)  # sum-preserving adipose<->fibro swap (Pc fixed)
    # Cramer-Rao variance of the null coordinate WITH the other parameter unknown (nuisance)
    singular = ev_r[0] <= 1e-9 * ev_r[-1]
    sd_t = float("inf") if singular else float(np.sqrt(nr @ np.linalg.inv(Fr) @ nr))
    return {
        "n_meas": len(meas),
        "svals_reduced_J": sv.tolist(),
        "cond_reduced": float(sv[0] / sv[-1]) if len(sv) > 1 and sv[-1] > 0 else None,
        "fisher_eig_reduced": ev_r.tolist(),
        "fisher_eig_unconstrained3": ev_f.tolist(),
        "rank3_unconstrained(sv>1e-6*max)": int(
            (
                np.linalg.svd(J, compute_uv=False) > 1e-6 * np.linalg.svd(J, compute_uv=False)[0]
            ).sum()
        ),
        "crb_std_t_cm": sd_t,
        "crb_std_adipose_fraction": sd_t / np.sqrt(2.0) / L,
        # calcification known absent AND sum known: ONE unknown (the swap coordinate), no nuisance
        "crb_adipose_fraction_calc0_sum_known": float(
            1.0
            / np.sqrt(float((W * (J @ np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)) ** 2).sum()))
            / np.sqrt(2.0)
            / L
        ),
    }


def main() -> int:
    tables = load()
    out: dict = {"N_total_photons_per_ray": N_TOTAL, "cases": {}}
    # material-angle across energy: how close are mu_a(E) and mu_f(E) curves?
    for kvp, t in tables.items():
        ua, uf = t.mu_per_cm[0], t.mu_per_cm[1]
        w = 0.5 * t.spectrum
        out[f"mu_curve_angle_deg_adipose_fibro_{kvp}kVp(spectrum-weighted)"] = float(
            np.degrees(
                np.arccos((w * ua * uf).sum() / np.sqrt((w * ua**2).sum() * (w * uf**2).sum()))
            )
        )
        out[f"mu_ratio_f_over_a_{kvp}kVp(min,max over E where w>0)"] = [
            float((uf / ua)[w > 1e-6 * w.max()].min()),
            float((uf / ua)[w > 1e-6 * w.max()].max()),
        ]
    comps = {
        "adipose-rich(0.8/0.2/0)": (0.8, 0.2, 0.0),
        "mixed(0.5/0.5/0)": (0.5, 0.5, 0.0),
        "fibro-rich(0.2/0.8/0)": (0.2, 0.8, 0.0),
    }
    for L in (2.0, 6.0, 12.0):
        for cname, x in comps.items():
            P = L * np.array(x)
            for kind in ("SECT-80", "SECT-50", "DECT", "PCCT-2", "PCCT-4", "PCCT-8", "PCCT-32"):
                meas = measurements(tables, kind)
                out["cases"][f"L={L}cm|{cname}|{kind}"] = analyse(
                    meas, P, L, kind.startswith("PCCT")
                )
    (Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "theory.json").write_text(
        json.dumps(out, indent=1)
    )
    for k in (
        "L=6.0cm|mixed(0.5/0.5/0)|SECT-80",
        "L=6.0cm|mixed(0.5/0.5/0)|DECT",
        "L=6.0cm|mixed(0.5/0.5/0)|PCCT-4",
        "L=6.0cm|mixed(0.5/0.5/0)|PCCT-32",
    ):
        r = out["cases"][k]
        print(
            k,
            "sv",
            np.round(r["svals_reduced_J"], 5),
            f"crb_frac {r['crb_std_adipose_fraction']:.3g}",
            "rank3",
            r["rank3_unconstrained(sv>1e-6*max)"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
