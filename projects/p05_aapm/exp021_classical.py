"""EXP-021: classical image-domain baselines on the frozen split (512x512, official s1/s2).

Inputs are the two reconstructed images only (no transmissions, no labels). Methods: unconstrained
(min-norm pinv), nonneg_ls (per-pixel NNLS), sum_to_one (3x3 solve), simplex (projection of
sum_to_one), tv{lam} (FISTA, simplex on tissue). Air rule/cutoff are the documented defaults
(classical.AIR_CUTOFF_PER_CM, NOT tuned). The TV weight is selected on VALIDATION s1 from the grid
fixed in the experiment record. `--stage test` evaluates the test split ONCE with the chosen lam.
Usage: exp021_classical.py --data D --split S --stage val [--n-cases 100] [--workers 48]
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ctsynth.aapm import schema as S
from ctsynth.aapm.classical import AIR_CUTOFF_PER_CM, decompose_lstsq, decompose_tv, effective_mu
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.metrics import (
    boundary_rmse,
    partition_errors,
    per_material_errors,
    regime_masks,
    s1_mean_rmse,
    s2_worst_roi_rmse,
)

LAMS = [1e-4, 1e-3, 1e-2, 1e-1]  # pre-registered grid
_G: dict = {}


def nonneg_ls(lo, hi, mu, n_iter=300):
    """Per-pixel min 0.5||mu x - b||^2 s.t. x >= 0 (FISTA); air -> 0 like the other modes."""
    air = (lo < AIR_CUTOFF_PER_CM) & (hi < AIR_CUTOFF_PER_CM)
    b = np.stack([lo.ravel(), hi.ravel()], 1)
    x = np.maximum(b @ np.linalg.pinv(mu).T, 0.0)
    y, t, lip = x.copy(), 1.0, np.linalg.norm(mu, 2) ** 2
    for _ in range(n_iter):
        xn = np.maximum(y - (y @ mu.T - b) @ mu / lip, 0.0)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y, x, t = xn + (t - 1) / tn * (xn - x), xn, tn
    x[air.ravel()] = 0.0
    return np.moveaxis(x.reshape(*lo.shape, 3), -1, -3)


def run_method(name, lo, hi, mu):
    if name == "zero":  # reference floor: predicting nothing
        return np.zeros((3, *lo.shape))
    if name == "nonneg_ls":
        return nonneg_ls(lo, hi, mu)
    if name.startswith("tv"):
        return decompose_tv(lo, hi, mu, lam=float(name[2:]), n_iter=300)
    return decompose_lstsq(lo, hi, mu, mode=name)


def work(args):
    idx, names = args
    c = _G["store"][idx]
    lo, hi = c.image[50].astype(np.float64), c.image[80].astype(np.float64)
    t = c.phantom.astype(np.float64)
    out = {}
    for n in names:
        t0 = time.time()
        p = run_method(n, lo, hi, _G["mu"])
        rt = time.time() - t0
        e2 = (p - t) ** 2
        air, tis, band = regime_masks(t[None])
        out[n] = {
            "s1": s1_mean_rmse(t, p),
            "s2": s2_worst_roi_rmse(t, p),
            "sq_sum_mat": e2.sum((1, 2)).tolist(),
            "per_material": per_material_errors(t, p),
            "partition": partition_errors(t, p),
            "boundary_rmse": boundary_rmse(t, p),
            "sq_air": float(e2[:, air[0]].sum()),
            "n_air": int(air.sum()) * 3,
            "sq_tissue": float(e2[:, tis[0]].sum()),
            "n_tissue": int(tis.sum()) * 3,
            "sq_band": float(e2[:, band[0]].sum()),
            "n_band": int(band.sum()) * 3,
            "runtime_s": rt,
            "frac_negative": float((p < -1e-6).mean()),
        }
    return idx, out


def init(data):
    tab = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    tables = {k: S.load_model_table(tab / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    _G["store"], _G["mu"] = CaseStore(Path(data) / "files"), effective_mu(tables)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--stage", choices=["val", "test"], required=True)
    ap.add_argument("--n-cases", type=int, default=100)
    ap.add_argument("--lam", type=float, help="chosen TV weight (stage test only)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    run = Path(os.environ.get("RESEARCH_RUN_DIR", "."))
    sp = json.loads(a.split.read_text())
    ids = sp[a.stage][: a.n_cases]
    base = ["zero", "unconstrained", "nonneg_ls", "sum_to_one", "simplex"]
    if a.stage == "test":
        assert a.lam in LAMS, "test stage needs the validation-selected --lam from the grid"
        names = [*base, f"tv{a.lam:g}"]
    else:
        names = [*base, *[f"tv{lam:g}" for lam in LAMS]]
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.data,)) as pool:
        res = dict(pool.map(work, [(i, names) for i in ids], chunksize=1))
    summ = {}
    for n in names:
        rs = [res[i][n] for i in ids]
        sq_mat = np.sum([r["sq_sum_mat"] for r in rs], 0)
        npix = 512 * 512 * len(ids)

        def reg(k, rs=rs):
            return float(
                np.sqrt(sum(r[f"sq_{k}"] for r in rs) / max(sum(r[f"n_{k}"] for r in rs), 1))
            )

        summ[n] = {
            "s1": float(np.mean([r["s1"] for r in rs])),
            "s2": float(np.max([r["s2"] for r in rs])),
            "rmse_per_material": dict(
                zip(S.MATERIALS, np.sqrt(sq_mat / npix).tolist(), strict=True)
            ),
            "rmse_interior": reg("tissue"),
            "rmse_boundary": reg("band"),
            "rmse_air": reg("air"),
            "mean_boundary_rmse_case": float(np.nanmean([r["boundary_rmse"] for r in rs])),
            "tissue_mean_abs_sum_err": float(
                np.mean([r["partition"]["tissue_mean_abs"] for r in rs])
            ),
            "frac_negative": float(np.mean([r["frac_negative"] for r in rs])),
            "runtime_s_per_case": float(np.mean([r["runtime_s"] for r in rs])),
        }
    out = {
        "stage": a.stage,
        "n_cases": len(ids),
        "air_cutoff_per_cm": AIR_CUTOFF_PER_CM,
        "split_manifest_sha256": sp["manifest_sha256"],
        "methods": summ,
        "wall_s": time.time() - t0,
        "workers": a.workers,
    }
    tv = {n: v["s1"] for n, v in summ.items() if n.startswith("tv")}
    if tv and a.stage == "val":
        out["selected_tv"] = min(tv, key=tv.get)
        out["selection_rule"] = "lowest validation s1 among the pre-registered grid"
    (run / f"exp021_{a.stage}.json").write_text(json.dumps(out, indent=1))
    (run / f"exp021_{a.stage}_per_case.json").write_text(json.dumps(res))
    with open(run / "artifacts.jsonl", "a") as f:
        for nme in (f"exp021_{a.stage}.json", f"exp021_{a.stage}_per_case.json"):
            f.write(json.dumps({"path": str((run / nme).resolve()), "name": nme}) + "\n")
    with open(run / "metrics.jsonl", "a") as f:
        for n, v in summ.items():
            f.write(json.dumps({"name": f"s1_{n}", "value": v["s1"]}) + "\n")
            f.write(json.dumps({"name": f"s2_{n}", "value": v["s2"]}) + "\n")
    print(json.dumps({n: round(v["s1"], 4) for n, v in summ.items()}), out.get("selected_tv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
