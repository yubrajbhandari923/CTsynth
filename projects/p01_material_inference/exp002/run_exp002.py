# ruff: noqa: E501
"""EXP-002 runner: spectral identifiability toy (bins x basis size x spectrum mismatch).

Usage: uv run python projects/p01_material_inference/exp002/run_exp002.py [configs/exp002.yaml]
Writes <out_dir>/metrics.json, run_meta.json, recon_*.npz. See README.md for scope/limitations.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phantom as ph  # noqa: E402
from estimator import DT, build_design, fisher_analysis, lm_poisson  # noqa: E402

from ctsynth import runmeta  # noqa: E402
from ctsynth.config import ExperimentConfig, load_config  # noqa: E402
from ctsynth.physics import MaterialBasis, SpectralForwardModel, Spectrum  # noqa: E402
from ctsynth.seeding import rng  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MAT = ("water", "iodine", "calcium")
A_MIN = 1e-4  # g/cm^2: a ray "sees" material m if A_m exceeds this
LIMITATION = (
    "D-002: the iodine K-edge (33.17 keV) is under-resolved on the 1 keV table. Data generation "
    "and estimation use the SAME table, a shared-model limitation. No absolute iodine-"
    "concentration accuracy claim is made; only identifiability, conditioning and noise claims."
)
# (phantom, design, k) cells. P2=water+iodine, P3 adds calcium. k=2 on P3 is misspecified.
MATCHED = [(p, d, k) for p, k in (("P2", 2), ("P3", 3), ("P3", 2)) for d in ("M1", "M2", "M4")]
MISMATCH_CELLS = [("P2", "M2", 2), ("P2", "M4", 2), ("P3", "M4", 3)]
FULL_CELLS = [("P2", "M2", 2), ("P3", "M4", 3), ("P3", "M2", 2), ("P2", "M4", 2)]


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    return o


def _gen(seed: int, *stream: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(rng(seed, *stream).integers(2**31 - 1)))
    return g


def _fl(x: torch.Tensor) -> float:
    return float(x)


def select_rays(
    rho: torch.Tensor, ins: list[dict[str, float]], proj: Any, n_per: int, n_rand: int, seed: int
) -> np.ndarray:
    """Flat ray indices (view*n_det+det): n_per rays through each insert centre + n_rand random."""
    sel: set[int] = set()
    for it in ins:
        for i in range(n_per):
            v = int(round(i * proj.n_view / n_per)) % proj.n_view
            th = float(proj.angles[v])
            s = -it["x"] * math.sin(th) + it["y"] * math.cos(th)
            j = int(round((s - proj.det_offset) / proj.det_spacing + (proj.n_det - 1) / 2))
            sel.add(v * proj.n_det + j)
    n_all = proj.n_view * proj.n_det
    sel.update(int(i) for i in rng(seed, 1).choice(n_all, size=n_rand, replace=False))
    return np.array(sorted(sel))


class Ctx:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg, self.p = cfg, cfg.params
        att = np.load(ROOT / "data/tables/attenuation_v1.npz")
        spec = np.load(ROOT / "data/tables/spectra_v1.npz")
        self.E = att["energies_keV"]
        self.kappa = torch.from_numpy(att["kappa_cm2_per_g"]).to(DT)
        self.spectra = {k: spec[k] for k in ("fluence_80", "fluence_120")}
        self.proj = ph.make_projector()
        self.basis = MaterialBasis(
            MAT,
            torch.from_numpy(att["ref_density_g_cm3"]).to(DT),
            torch.from_numpy(self.E).to(DT),
            self.kappa,
        )
        self.ph: dict[str, Any] = {}
        for name in ("P2", "P3"):
            rho, ins, masks = ph.build(name)
            A = self.proj.forward(rho)  # (3, V, D) g/cm^2
            rays = select_rays(
                rho,
                ins,
                self.proj,
                int(self.p["rays_per_insert"]),
                int(self.p["n_random_rays"]),
                cfg.seed,
            )
            self.ph[name] = {"rho": rho, "ins": ins, "masks": masks, "A": A, "rays": rays}

    def design(self, name: str, n_tot: float, tilt: float = 0.0, escale: float = 0.0) -> Any:
        return build_design(name, self.spectra, self.E, n_tot, tilt, escale)

    def data_lambda(self, W: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Expected counts via ctsynth.physics.SpectralForwardModel (fluence=1, response=W)."""
        spec = Spectrum(
            torch.from_numpy(self.E).to(DT),
            torch.ones(len(self.E), dtype=DT),
            torch.ones(len(self.E), dtype=DT),
            W,
        )
        model = SpectralForwardModel(self.basis, spec, self.proj)
        return model.expected_counts_from_line_integrals(A)  # (nB, ...)


CONDS: list[tuple[str, float, float]] = []  # filled in run()


def run_subset_cell(
    c: Ctx, phn: str, dname: str, k: int, n_tot: float, cond: tuple[str, float, float], stream: int
) -> dict[str, Any]:
    tag, tilt, esc = cond
    d = c.ph[phn]
    A3 = d["A"].reshape(3, -1)[:, d["rays"]].T.contiguous()  # (n,3)
    nom = c.design(dname, n_tot)
    dat = c.design(dname, n_tot, tilt, esc)
    for dd in (nom, dat):  # pre-registered equal-photon check
        assert abs(float(dd.W.sum()) - n_tot) / n_tot < 1e-9, "N_tot scaling violated"
    kap = c.kappa[:k]
    fa = fisher_analysis(A3[:, :k], nom.W, kap, float(c.p["rank_rtol"]))
    rec: dict[str, Any] = {
        "cell": f"{phn}|{dname}|k{k}|N{n_tot:g}|{tag}",
        "phantom": phn,
        "design": dname,
        "k": k,
        "n_tot": n_tot,
        "condition": tag,
        "n_rays": len(d["rays"]),
        "n_meas": nom.n_meas,
        "unattenuated_total_nominal": float(nom.W.sum()),
        "unattenuated_total_data": float(dat.W.sum()),
        "rank_per_ray_min_max": [int(fa["rank"].min()), int(fa["rank"].max())],
        "frac_rays_rank_k": _fl(fa["ident"].double().mean()),
        "sv_median": [_fl(v) for v in fa["sv"].median(dim=0).values],
        "identifiable": bool(fa["ident"].all()),
    }
    if not rec["identifiable"]:
        rec["status"] = "non-identifiable: no recovery claim"
        return rec
    rec["status"] = "identifiable"
    cond_v = fa["cond"]
    rec["jac_cond"] = {"median": _fl(cond_v.median()), "max": _fl(cond_v.max())}
    R, n = int(c.p["n_real"]), A3.shape[0]
    lam = c.data_lambda(dat.W, A3.T)  # (nB, n)
    y = torch.poisson(
        lam.T[None].expand(R, n, -1).contiguous(), generator=_gen(c.cfg.seed, 2, stream)
    )
    Ahat, conv = lm_poisson(y.reshape(R * n, -1), nom.W, kap, int(c.p["lm_max_iter"]))
    Ahat, conv = Ahat.reshape(R, n, k), conv.reshape(R, n)
    rec["lm_fail_frac"] = 1.0 - _fl(conv.double().mean())
    Ah = torch.where(conv[..., None], Ahat, torch.full_like(Ahat, float("nan")))
    err = Ah - A3[None, :, :k]
    cnt = (~torch.isnan(err[..., 0])).sum(0).double()  # converged reps per ray
    mean_err = torch.nanmean(err, 0)
    var = (
        torch.nanmean((err - mean_err[None]) ** 2, 0)
        * cnt[:, None]
        / (cnt[:, None] - 1).clamp_min(1)
    )
    emp = var.sqrt()
    per_mat: dict[str, Any] = {}
    for m in range(k):
        mk = (A3[:, m] > A_MIN) & (cnt >= 2)
        if not bool(mk.any()):
            continue
        cr = fa["crlb"][mk, m]
        per_mat[MAT[m]] = {
            "n_rays": int(mk.sum()),
            "crlb_std_gcm2_median": _fl(cr.median()),
            "crlb_rel_std_median": _fl((cr / A3[mk, m]).median()),
            "emp_std_gcm2_median": _fl(emp[mk, m].median()),
            "emp_over_crlb_median": _fl((emp[mk, m] / cr).median()),
            "emp_over_crlb_min_max": [_fl((emp[mk, m] / cr).min()), _fl((emp[mk, m] / cr).max())],
            "bias_gcm2_mean_signed": _fl(mean_err[mk, m].mean()),
            "bias_gcm2_rms_per_ray": _fl(mean_err[mk, m].pow(2).mean().sqrt()),
            "bias_noise_floor_rms": _fl((var[mk, m] / cnt[mk]).mean().sqrt()),
        }
    rec["per_material"] = per_mat
    return rec


def run_full_cell(
    c: Ctx,
    phn: str,
    dname: str,
    k: int,
    n_tot: float,
    cond: tuple[str, float, float],
    stream: int,
    out: Path,
) -> dict[str, Any]:
    tag, tilt, esc = cond
    d = c.ph[phn]
    nom, dat = c.design(dname, n_tot), c.design(dname, n_tot, tilt, esc)
    assert abs(float(dat.W.sum()) - n_tot) / n_tot < 1e-9
    V, D = c.proj.n_view, c.proj.n_det
    lam = c.data_lambda(dat.W, d["A"])  # (nB, V, D)
    y = torch.poisson(lam, generator=_gen(c.cfg.seed, 3, stream))
    Ah, conv = lm_poisson(
        y.reshape(lam.shape[0], -1).T.contiguous(), nom.W, c.kappa[:k], int(c.p["lm_max_iter"])
    )
    Ah = Ah.T.reshape(k, V, D)
    # Unconstrained LS: rho_hat = argmin ||P rho - A_hat|| per material (dense lstsq).
    eye = torch.eye(ph.N_PIX**2, dtype=DT).reshape(-1, ph.N_PIX, ph.N_PIX)
    P = c.proj.forward(eye).reshape(ph.N_PIX**2, -1).T  # (V*D, npix)
    sol = torch.linalg.lstsq(P, Ah.reshape(k, -1).T).solution  # (npix, k)
    rho_hat = sol.T.reshape(k, ph.N_PIX, ph.N_PIX)
    tag_id = f"{phn}_{dname}_k{k}_N{n_tot:g}_{tag}"
    np.savez_compressed(
        out / f"recon_{tag_id}.npz",
        rho_hat=rho_hat.numpy(),
        rho_true=d["rho"].numpy(),
        converged=conv.reshape(V, D).numpy(),
    )
    per: list[dict[str, Any]] = []
    for j, it in enumerate(d["ins"]):
        m = int(it["material"])
        if m >= k:  # material not in the estimator basis: not recoverable by construction
            continue
        msk = d["masks"][j]
        e = rho_hat[m][msk] - d["rho"][m][msk]
        per.append(
            {
                "insert": j,
                "material": MAT[m],
                "true_gcm3": it["density"],
                "mean_hat_gcm3": _fl(rho_hat[m][msk].mean()),
                "bias_gcm3": _fl(e.mean()),
                "rmse_gcm3": _fl(e.pow(2).mean().sqrt()),
            }
        )
    # insert-edge pixels are pixel-constant/partial-volume free here (masks are binary), so pooled
    # RMSE includes the discretisation of the Siddon projector + LS inversion.
    rec = {
        "cell": tag_id,
        "phantom": phn,
        "design": dname,
        "k": k,
        "n_tot": n_tot,
        "condition": tag,
        "lm_fail_frac": 1.0 - _fl(conv.double().mean()),
        "inserts": per,
        "n_rays": V * D,
        "rho_hat_file": f"recon_{tag_id}.npz",
    }
    for mname in ("iodine", "calcium"):
        rows = [r for r in per if r["material"] == mname]
        if rows:
            rec[f"insert_rmse_gcm3_{mname}"] = math.sqrt(
                sum(r["rmse_gcm3"] ** 2 for r in rows) / len(rows)
            )
            rec[f"mismatch_bias_gcm3_{mname}"] = sum(r["bias_gcm3"] for r in rows) / len(rows)
    return rec


def evaluate(
    cells: list[dict[str, Any]],
    full: list[dict[str, Any]],
    n_hi: float,
    n_lo: float,
    tilt: float,
    escale: float,
) -> dict[str, Any]:
    by = {x["cell"]: x for x in cells}

    def get(p: str, dn: str, k: int, n: float, tag: str = "matched") -> dict[str, Any]:
        return by[f"{p}|{dn}|k{k}|N{n:g}|{tag}"]

    out: dict[str, Any] = {}
    allr = [get(p, "M1", k, n) for p, k in (("P2", 2), ("P3", 3), ("P3", 2)) for n in (n_lo, n_hi)]
    out["a"] = {
        "criterion": "M1, k>=2: every sampled ray has rank 1 (<k)",
        "rank_ranges": {x["cell"]: x["rank_per_ray_min_max"] for x in allr},
        "pass": all(x["rank_per_ray_min_max"] == [1, 1] for x in allr),
    }
    b = [get("P2", "M2", 2, n) for n in (n_lo, n_hi)]
    out["b"] = {
        "criterion": "M2 k=2: rank 2 on every ray, finite cond",
        "cond_median": {x["cell"]: x.get("jac_cond", {}).get("median") for x in b},
        "pass": all(x["identifiable"] for x in b),
    }
    cc = [get("P3", "M2", 3, n) for n in (n_lo, n_hi)]
    out["c"] = {
        "criterion": "M2 k=3: rank<=2<3 on every ray => non-identifiable",
        "rank_ranges": {x["cell"]: x["rank_per_ray_min_max"] for x in cc},
        "pass": all((not x["identifiable"]) and x["rank_per_ray_min_max"][1] <= 2 for x in cc),
    }
    m4, m2 = get("P3", "M4", 3, n_hi), get("P3", "M2", 2, n_hi)
    if m4["identifiable"] and m2["identifiable"]:
        r = (
            m4["per_material"]["iodine"]["crlb_std_gcm2_median"]
            / m2["per_material"]["iodine"]["crlb_std_gcm2_median"]
        )
        out["d"] = {
            "criterion": "M4 k=3 identifiable (all rays) and iodine CRLB std (median over iodine rays) "
            "larger than M2 k=2 at equal N_tot (P3, N=1e6)",
            "iodine_crlb_ratio_M4k3_over_M2k2": r,
            "cond_median": {"M4k3": m4["jac_cond"]["median"], "M2k2": m2["jac_cond"]["median"]},
            "pass": r > 1.0,
        }
    else:
        out["d"] = {"pass": False, "note": "M4 k=3 or M2 k=2 not identifiable on P3"}

    def bias_ok(lo: dict[str, Any], hi: dict[str, Any]) -> dict[str, Any]:
        if not (lo["identifiable"] and hi["identifiable"]):
            return {"pass": False, "note": "non-identifiable"}
        a, h = lo["per_material"]["iodine"], hi["per_material"]["iodine"]
        ratio = h["bias_gcm2_rms_per_ray"] / a["bias_gcm2_rms_per_ray"]
        sig = min(
            a["bias_gcm2_rms_per_ray"] / a["bias_noise_floor_rms"],
            h["bias_gcm2_rms_per_ray"] / h["bias_noise_floor_rms"],
        )
        return {
            "bias_rms_lo": a["bias_gcm2_rms_per_ray"],
            "bias_rms_hi": h["bias_gcm2_rms_per_ray"],
            "ratio_hi_over_lo": ratio,
            "min_bias_over_noise_floor": sig,
            "pass": bool(ratio > 0.5 and sig > 3.0),
        }

    out["e"] = {
        "criterion": "P3 data, k=2 estimator: iodine-A per-ray rms bias > 3x its noise floor at "
        "both N_tot and bias(1e6)/bias(1e5) > 0.5 (a 1/sqrt(N) bias would give 0.32)",
        "cells": {
            dn: bias_ok(get("P3", dn, 2, n_lo), get("P3", dn, 2, n_hi)) for dn in ("M2", "M4")
        },
    }
    out["e"]["pass"] = all(v["pass"] for v in out["e"]["cells"].values())
    fcells = {}
    for p, dn, k in MISMATCH_CELLS:
        for tag in (
            f"tilt{tilt:+g}",
            f"tilt{-tilt:+g}",
            f"escale{escale:+g}",
            f"escale{-escale:+g}",
        ):
            fcells[f"{p}|{dn}|k{k}|{tag}"] = bias_ok(
                get(p, dn, k, n_lo, tag), get(p, dn, k, n_hi, tag)
            )
    out["f"] = {
        "criterion": "same bias test as (e) for each matched-k cell x mismatch condition",
        "cells": fcells,
        "n_pass": sum(v["pass"] for v in fcells.values()),
        "n_total": len(fcells),
    }
    out["f"]["pass"] = out["f"]["n_pass"] == out["f"]["n_total"]
    g_cells = {}
    for p, dn, k in (("P2", "M2", 2), ("P2", "M4", 2), ("P3", "M4", 3)):
        x = get(p, dn, k, n_hi)
        g_cells[x["cell"]] = {mn: v["emp_over_crlb_median"] for mn, v in x["per_material"].items()}
    vals = [v for cc_ in g_cells.values() for v in cc_.values()]
    out["g"] = {
        "criterion": "median emp/CRLB per material within 25% (success) at N=1e6, matched cells; >50% = STOP",
        "emp_over_crlb_median": g_cells,
        "pass": all(0.75 <= v <= 1.25 for v in vals),
        "stop_triggered": any(abs(v - 1) > 0.5 for v in vals),
    }
    out["lm_failures"] = {
        x["cell"]: x["lm_fail_frac"] for x in cells + full if x.get("lm_fail_frac", 0) > 0
    }
    return out


def run(cfg: ExperimentConfig, out_dir: Path) -> dict[str, Any]:
    c = Ctx(cfg)
    p = cfg.params
    nl, nh = float(p["n_tot_low"]), float(p["n_tot_high"])
    tl, es = float(p["tilt"]), float(p["escale"])
    matched = ("matched", 0.0, 0.0)
    mism = [
        (f"tilt{tl:+g}", tl, 0.0),
        (f"tilt{-tl:+g}", -tl, 0.0),
        (f"escale{es:+g}", 0.0, es),
        (f"escale{-es:+g}", 0.0, -es),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    s = 0
    for n in (nl, nh):
        for phn, dn, k in MATCHED:
            cells.append(run_subset_cell(c, phn, dn, k, n, matched, s))
            s += 1
        for phn, dn, k in MISMATCH_CELLS:
            for cd in mism:
                cells.append(run_subset_cell(c, phn, dn, k, n, cd, s))
                s += 1
    full: list[dict[str, Any]] = []
    if p["full_image"]:
        for n in (nl, nh):
            for phn, dn, k in FULL_CELLS:
                for cd in (matched, mism[0]):
                    full.append(run_full_cell(c, phn, dn, k, n, cd, s, out_dir))
                    s += 1
    preds = evaluate(cells, full, nh, nl, tl, es)
    ident = [x for x in cells if x["identifiable"]]
    metrics = {
        "jac_cond": {x["cell"]: x["jac_cond"] for x in ident},
        "crlb_rel_std": {
            x["cell"]: {m: v["crlb_rel_std_median"] for m, v in x["per_material"].items()}
            for x in ident
        },
        "emp_over_crlb": {
            x["cell"]: {m: v["emp_over_crlb_median"] for m, v in x["per_material"].items()}
            for x in ident
        },
        "insert_rmse_gcm3": {
            x["cell"]: {
                m: x[f"insert_rmse_gcm3_{m}"]
                for m in ("iodine", "calcium")
                if f"insert_rmse_gcm3_{m}" in x
            }
            for x in full
        },
        "mismatch_bias_gcm3": {
            x["cell"]: {
                m: x[f"mismatch_bias_gcm3_{m}"]
                for m in ("iodine", "calcium")
                if f"mismatch_bias_gcm3_{m}" in x
            }
            for x in full
        },
        "non_identifiable_cells": [x["cell"] for x in cells if not x["identifiable"]],
    }
    res = {
        "experiment_id": cfg.experiment_id,
        "seed": cfg.seed,
        "limitations": [
            LIMITATION,
            "k=2 Jacobian/CRLB for P3 are evaluated at the truncated A[:, :2]; "
            "CRLB is not valid for misspecified cells and emp/CRLB is not claimed there.",
        ],
        "scaling": "For every design and data condition the fluence (SpekPy native normalisation, "
        "absolute scale UNVERIFIED) is rescaled so that sum over rows b and energies e of "
        "R[b,e]*Phi[e] (unattenuated expected counts, all rows summed) equals N_tot; "
        "exposures of M2 share N_tot/2 each. Asserted to 1e-9 relative for every cell. "
        "M4 counts only photons inside 20-120 keV. Tilt/energy-scale mismatch is applied to "
        "the data fluence BEFORE rescaling, so mismatch cells also have exactly N_tot photons.",
        "weighting": {
            "M1_M2": "energy-integrating: R[b,e] = E_e / Ebar (Ebar = photon-mean energy of the "
            "nominal exposure), i.e. proportional to E, normalised so unattenuated lambda = photon "
            "count; Poisson noise is applied to this normalised count (an idealisation)",
            "M4": "photon counting: R[b,e] = indicator(window), weight 1 per photon; windows "
            "[20,50),[50,70),[70,90),[90,120] keV",
        },
        "mismatch_definition": f"tilt: phi*(1 + t(E-Ebar)/Ebar), t=+-{tl}; energy scale: phi(E/(1+s)), s=+-{es}; "
        "windows/responses stay nominal; estimator always uses nominal spectra",
        "scope": {
            "per_ray_subset": "Fisher/CRLB/rank/empirical std/bias on a fixed subset of rays "
            f"({len(c.ph['P2']['rays'])} rays P2, {len(c.ph['P3']['rays'])} rays P3: "
            f"{p['rays_per_insert']} rays through each insert centre + {p['n_random_rays']} "
            f"random rays), {p['n_real']} Poisson realisations each. NOT full image.",
            "full_image": "single noise realisation, all 90x72 rays, for cells in full_image records only "
            "(N_tot in both levels; matched and tilt+10% data); rho via unconstrained dense LS of A_hat",
            "rank_rtol": p["rank_rtol"],
            "stats_exclude_nonconverged": "non-converged fits are excluded from bias/std but their fraction is "
            "reported (lm_fail_frac) per cell",
        },
        "metrics": metrics,
        "cells": cells,
        "full_image": full,
        "predictions": preds,
    }
    return _clean(res)  # type: ignore[no-any-return]


def main(argv: list[str]) -> None:
    cfg_path = Path(argv[1]) if len(argv) > 1 else ROOT / "configs/exp002.yaml"
    cfg = load_config(cfg_path)
    out_dir = ROOT / str(cfg.params["out_dir"])
    t0 = time.time()
    res = run(cfg, out_dir)
    elapsed = time.time() - t0
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    runmeta.write(cfg, out_dir)
    (out_dir / "timing.json").write_text(json.dumps({"runtime_s": elapsed}))  # kept out of metrics
    print(f"done in {elapsed:.1f}s -> {out_dir}")
    for k, v in res["predictions"].items():
        print(k, v.get("pass") if isinstance(v, dict) else v)


if __name__ == "__main__":
    main(sys.argv)
