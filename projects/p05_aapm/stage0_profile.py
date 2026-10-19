"""Stage 0 (PLAN-006): prior profile test along the exact null direction and the A<->F swap.

Deterministic, noise-free, TRAIN-split cases only; no training, no network. For truth + t*d the
projection MSE (mean over rays and kVp of (T_pred - T)^2, same scale as the training loss) and each
prior R are evaluated on a t grid for d in {exact null n, swap, calc-scale, smooth field, rough
field}; plus the wrong-label point. Weight windows come from `priors.lambda_window`. P4 uses a
marginal built from OTHER train cases' labels (weak supervision, flagged).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ctsynth import seeding
from ctsynth.aapm import priors as P
from ctsynth.aapm import schema as S
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.physics_loss import stack_kvp

TGRID = np.round(np.linspace(-0.4, 0.4, 41), 4)
RES_CLASSICAL = 2e-8  # MSE of the best label-free physics refinement so far (EXP-028: RMSE 1.4e-4)
RES_NETWORK = 1e-6  # MSE the physics-only networks reach (RMSE ~1e-3)
LAMS = np.logspace(-12, 0, 25)


def smooth_field(shape, sigma, g):
    """Low-pass random field in [-1, 1] (FFT gaussian), deterministic from generator g."""
    w = g.standard_normal(shape)
    fx = np.fft.fftfreq(shape[0])[:, None]
    fy = np.fft.fftfreq(shape[1])[None, :]
    k = np.exp(-2 * (np.pi * sigma) ** 2 * (fx**2 + fy**2))
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * k))
    return f / np.abs(f).max()


def exact_null(tables) -> np.ndarray:
    M = np.array(
        [
            [(0.5 * tables[k].spectrum * tables[k].mu_per_cm[m]).sum() for m in range(3)]
            for k in S.KVPS
        ]
    )
    n = np.linalg.svd(M)[2][-1]
    return n * np.sign(-n[0])


def record(acc, fwd, prior_fns, name, f, T, tis) -> None:
    acc[name]["mse"].append(float(((stack_kvp(fwd(f)) - T) ** 2).mean()))
    for k, fn in prior_fns.items():
        acc[name][k].append(float(fn(f.double(), tis)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--n-cases", type=int, default=20)
    ap.add_argument("--n-ref", type=int, default=100)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    tables = {k: S.load_model_table(tab / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    store = CaseStore(a.data / "files")
    sp = json.loads(a.split.read_text())
    assert "train" in sp
    cases, ref_cases = sp["train"][: a.n_cases], sp["train"][a.n_cases : a.n_cases + a.n_ref]
    fwd = AapmFanbeamForward(tables, exp_clip=None).to(dev)
    nx = exact_null(tables)
    sw = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)

    ref_a = []  # P4 reference marginal from other train cases (labels: weak supervision)
    for i in ref_cases:
        ph = store[i].phantom.astype(np.float64)
        ti = np.abs(ph.sum(0) - 1) <= 1e-3
        ref_a.append((ph[0] / np.maximum(ph[0] + ph[1], 1e-8)).clip(0, 1)[ti][::7])
    ref_q = P.ref_quantiles(np.concatenate(ref_a), 256)

    prior_fns = {
        "calc_l1": lambda f, m: P.calc_l1(f, m),
        "calc_l0": lambda f, m: P.calc_l0(f, m, eps=1e-3),
        "bimodal": P.bimodal,
        "tv_a": P.tv_a,
        "w1_marginal": lambda f, m: P.w1_marginal(f, m, ref_q),
    }
    dirs = ["null", "swap", "calc_scale", "smooth", "rough", "wrong_label"]
    acc = {d: {"mse": [], **{k: [] for k in prior_fns}} for d in dirs}
    g = seeding.rng(0, 35)
    with torch.no_grad():
        for ci, i in enumerate(cases):
            c = store[i]
            tru = torch.tensor(c.phantom[None], dtype=torch.float32, device=dev)
            T = torch.tensor(
                np.stack([c.transmission[50], c.transmission[80]])[None],
                dtype=torch.float32,
                device=dev,
            )
            tis = (tru.sum(1) - 1).abs() <= 1e-3
            m = tis.unsqueeze(1).float()
            sm = torch.tensor(smooth_field(tru.shape[2:], 20.0, g), dtype=torch.float32, device=dev)
            rg = torch.tensor(
                np.sign(g.standard_normal(tru.shape[2:])), dtype=torch.float32, device=dev
            )

            def vec(v):
                return torch.tensor(v, dtype=torch.float32, device=dev).view(1, 3, 1, 1)

            for t in TGRID:
                t = float(t)
                perturbed = {
                    "null": tru + t * vec(nx) * m,
                    "swap": tru + t * vec(sw) * m,
                    "calc_scale": torch.cat([tru[:, :2], tru[:, 2:3] * (1 + t)], 1),
                    "smooth": tru + t * vec(sw) * sm * m,
                    "rough": tru + t * vec(sw) * rg * m,
                }
                for d, f in perturbed.items():
                    record(acc, fwd, prior_fns, d, f, T, tis)
            record(acc, fwd, prior_fns, "wrong_label", P.wrong_label(tru, tis), T, tis)
            print(f"case {ci + 1}/{len(cases)} done", flush=True)

    n_t = len(TGRID)
    out: dict = {
        "t_grid": TGRID.tolist(),
        "cases": cases,
        "ref_cases": ref_cases,
        "n_ref_pixels": int(sum(len(r) for r in ref_a)),
    }

    def prof(d, key):
        arr = np.array(acc[d][key], dtype=np.float64)
        return arr.reshape(len(cases), n_t).mean(0)

    mse = {d: prof(d, "mse") for d in dirs[:5]}
    out["data_mse_profiles"] = {d: mse[d].tolist() for d in mse}
    out["detection_floor"] = {}
    for d in ("null", "swap"):
        for name, res in (("classical_2e-8", RES_CLASSICAL), ("network_1e-6", RES_NETWORK)):
            ok = [abs(t) for t, v in zip(TGRID, mse[d], strict=True) if v >= res and t != 0]
            out["detection_floor"][f"{d}|{name}"] = min(ok) if ok else None
    wl = {k: float(np.mean(acc["wrong_label"][k])) for k in ["mse", *prior_fns]}
    tr = {k: float(prof("null", k)[n_t // 2]) for k in ["mse", *prior_fns]}  # t=0 is truth
    out["truth_values"], out["wrong_label_values"] = tr, wl
    out["priors"] = {}
    for k in prior_fns:
        id_dir = "calc_scale" if k.startswith("calc") else "swap"
        r_n, r_id = prof("null", k), prof(id_dir, k)
        entry: dict = {"identifiable_direction": id_dir, "R_truth": tr[k], "R_wrong_label": wl[k]}
        for rname, res in (("res_classical", RES_CLASSICAL), ("res_network", RES_NETWORK)):
            lo, hi = P.lambda_window(TGRID, r_n, TGRID, mse[id_dir], r_id, LAMS, res)
            entry[f"window_{rname}"] = [lo, hi]
        entry["R_null_profile"] = r_n.tolist()
        entry["R_id_profile"] = r_id.tolist()
        for d in ("swap", "smooth", "rough"):
            entry[f"R_{d}_profile"] = prof(d, k).tolist()
        # argmin at the lower window edge (classical residual) along each direction
        lo = entry["window_res_classical"][0]
        entry["argmin_at_lam_min"] = (
            {
                d: P.argmin_t(TGRID, mse[d], prof(d, k), lo)
                for d in ("null", "swap", "smooth", "rough")
            }
            if np.isfinite(lo)
            else None
        )
        entry["wrong_label_total_margin_at_lam_min"] = (
            (wl["mse"] - tr["mse"]) + lo * (wl[k] - tr[k]) if np.isfinite(lo) else None
        )
        entry["wrong_label_dR"] = wl[k] - tr[k]
        out["priors"][k] = entry
    dest = Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "stage0.json"
    dest.write_text(json.dumps(out, indent=1))
    for k, e in out["priors"].items():
        print(
            k,
            "window(classical)",
            e["window_res_classical"],
            "window(net)",
            e["window_res_network"],
            "dR_wrong",
            e["wrong_label_dR"],
            flush=True,
        )
    print("detection floor", out["detection_floor"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
