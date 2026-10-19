"""EXP-024 anomaly diagnostics (read-only on finished runs; no training, no labels in any model).

For each listed run: reload its checkpoint, re-evaluate on the frozen 100 val cases and report
(a) an independent s1 recomputation (eval-bug check), (b) tissue-pixel adipose histograms, error
decomposition along the F-012 null direction, per-case s1, (c) occupancy / softmax saturation,
(d) the data-term profile L(t) along the null direction (how flat is the polychromatic projection
loss, and is the truth lower than the solutions found?). Truth enters only as evaluation reference.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from milestone_a import NULL, load

from ctsynth.aapm import schema as S
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.metrics import s1_mean_rmse
from ctsynth.aapm.physics_loss import stack_kvp
from ctsynth.aapm.unet import UNet

HB = np.linspace(0, 1, 11)


def proj_rmse(fwd, f, T):
    return float(((stack_kvp(fwd(f)) - T) ** 2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument(
        "--run-root", type=Path, required=True, help="dir holding RUN-xxxx/checkpoint.pt"
    )
    ap.add_argument("--runs", nargs="+", required=True, help="RUN-id:output[:oracle][:expclip]")
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-profile", type=int, default=12)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    tables = {k: S.load_model_table(tab / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    store = CaseStore(a.data / "files")
    va = json.loads(a.split.read_text())["val"][: a.n_val]
    img, T, tru = load(store, va, dev)
    t = tru.cpu().numpy().astype(np.float64)
    tissue = np.abs(t.sum(1) - 1) <= 1e-3
    # EXACT null vector of the zero-thickness spectrum-weighted 2x3 matrix (float64 SVD). The
    # 3-decimal NULL constant is 0.024 deg off, leaving M@n ~ -7e-4/cm: that residual, not loss
    # curvature, dominated the first profile (RUN-0086; reviewer finding).
    M = np.array(
        [
            [(0.5 * tables[k].spectrum * tables[k].mu_per_cm[m]).sum() for m in range(3)]
            for k in S.KVPS
        ]
    )
    nx = np.linalg.svd(M)[2][-1]
    nx = nx * np.sign(-nx[0])
    sw = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)  # sum-preserving adipose<->fibro swap
    dirs = {"exact_null": nx, "swap": sw}
    nvec = torch.tensor(nx, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    out: dict = {}
    fwd0 = AapmFanbeamForward(tables, exp_clip=None).to(dev)
    ts = [-0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4]
    with torch.no_grad():
        # truth reference: data-term floor and profile along the null direction (tissue pixels)
        m = torch.tensor(tissue[:, None], device=dev, dtype=torch.float32)
        for dn, dv in dirs.items():
            dvec = torch.tensor(dv, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
            out[f"truth_profile_rmse_along_{dn}"] = {
                str(tt): float(
                    np.mean(
                        [
                            proj_rmse(fwd0, tru[c : c + 1] + tt * dvec * m[c : c + 1], T[c : c + 1])
                            for c in range(a.n_profile)
                        ]
                    )
                )
                for tt in ts
            }
        out["exact_null_vector"] = nx.tolist()
        out["M_dot_exact_null"] = (M @ nx).tolist()
        out["truth_proj_rmse_all"] = float(
            np.mean([proj_rmse(fwd0, tru[c : c + 1], T[c : c + 1]) for c in range(a.n_profile)])
        )
    for spec in a.runs:
        rid, output, *rest = spec.split(":")
        oracle = "oracle" in rest
        ec = [float(x) for x in rest if x not in ("oracle",)]
        fwd = AapmFanbeamForward(tables, exp_clip=ec[0] if ec else None).to(dev)
        model = UNet(depth=4, width=32, output=output).to(dev)
        model.load_state_dict(torch.load(a.run_root / rid / "checkpoint.pt", map_location=dev))
        model.eval()
        ps, sat = [], []
        with torch.no_grad():
            for s in range(0, len(va), 2):
                f = model(img[s : s + 2])
                if oracle:
                    Sv = tru[s : s + 2].sum(1, keepdim=True).clamp(0, 1)
                    sat.append(f.max(1).values[(Sv[:, 0] > 0.5)].cpu())
                    f = f * Sv
                elif output == "softmax":
                    sat.append(f.max(1).values.cpu().flatten())
                ps.append(f.cpu())
            p = torch.cat(ps).numpy().astype(np.float64)
            e = p - t
            sq = (e**2).sum(axis=(1, 2, 3))
            per_case = np.sqrt(sq / (3 * t.shape[2] * t.shape[3]))
            ta, pa = t[:, 0][tissue], p[:, 0][tissue]
            en = np.einsum("cmhw,m->chw", e, NULL)
            tmask = tissue
            r: dict = {
                "s1_recomputed": s1_mean_rmse(t, p),
                "per_case_s1": {
                    "min": float(per_case.min()),
                    "median": float(np.median(per_case)),
                    "max": float(per_case.max()),
                    "n_below_0.15": int((per_case < 0.15).sum()),
                },
                "tissue_adipose_hist_truth": np.histogram(ta, HB)[0].tolist(),
                "tissue_adipose_hist_pred": np.histogram(pa, HB)[0].tolist(),
                "tissue_adipose_pearson": float(np.corrcoef(ta, pa)[0, 1])
                if pa.std() > 0
                else None,
                "tissue_adipose_pred_mean_std": [float(pa.mean()), float(pa.std())],
                "tissue_calc_pred_mean": float(p[:, 2][tissue].mean()),
                "frac_tissue_within_0.1": float((np.abs(pa - ta) < 0.1).mean()),
                "null_energy_frac_L2_tissue": float(
                    (en[tmask] ** 2).sum() / (e.transpose(0, 2, 3, 1)[tmask] ** 2).sum()
                ),
                "null_coord_t_tissue_mean_std": [float(en[tmask].mean()), float(en[tmask].std())],
                "sum_pred_tissue_mean": float(p.sum(1)[tissue].mean()),
                "sum_abs_err_air": float(
                    np.abs(p.sum(1)[~tissue & (np.abs(t.sum(1)) < 1e-3)]).mean()
                ),
            }
            if sat:
                mx = torch.cat([x.flatten() for x in sat]).numpy()
                r["softmax_maxprob_tissue"] = {
                    "mean": float(mx.mean()),
                    "frac_gt_0.999": float((mx > 0.999).mean()),
                    "frac_gt_0.99999": float((mx > 0.99999).mean()),
                }
                am = np.argmax(p, 1)[tissue] if not oracle else np.argmax(p, 1)[tissue]
                r["argmax_class_frac_tissue"] = [float((am == k).mean()) for k in range(3)]
            m = torch.tensor(tissue[:, None], device=dev, dtype=torch.float32)
            pt = torch.tensor(p, dtype=torch.float32, device=dev)
            r["pred_profile_rmse_along_null"] = {
                str(tt): float(
                    np.mean(
                        [
                            proj_rmse(fwd, pt[c : c + 1] + tt * nvec * m[c : c + 1], T[c : c + 1])
                            for c in range(a.n_profile)
                        ]
                    )
                )
                for tt in ts
            }
        out[spec] = r
        print(spec, json.dumps({k: r[k] for k in ("s1_recomputed", "per_case_s1")}), flush=True)
    (Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "diag.json").write_text(
        json.dumps(out, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
