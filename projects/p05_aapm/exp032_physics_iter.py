"""EXP-032: classical projection-domain polychromatic iterative reconstruction (nothing learned).

Per case, minimise mean over rays and kVp of (T_pred(F) - T_meas)^2 + lam * TV(F) over F >= 0 by
Adam with projection onto F >= 0, random view subsets per step, using the verified differentiable
fan-beam forward. Inputs: measured transmissions (objective) and the two reconstructed images
(ONLY for the initial guess = clipped pinv and the air mask, as in EXP-021). No labels.
Usage: exp032_physics_iter.py --data D --split S --stage val --lam 1e-5 [--part 0 --parts 1]
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.aapm import schema as S
from ctsynth.aapm.classical import AIR_CUTOFF_PER_CM, decompose_lstsq, effective_mu
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.losses import tv
from ctsynth.aapm.metrics import boundary_rmse, partition_errors, s1_mean_rmse, s2_worst_roi_rmse
from ctsynth.aapm.physics_loss import projection_loss, stack_kvp


def solve(fwd, c, mu, dev, lam, iters, views, lr, gen):
    lo, hi = c.image[50].astype(np.float64), c.image[80].astype(np.float64)
    x0 = np.maximum(decompose_lstsq(lo, hi, mu, mode="unconstrained"), 0.0)
    air = torch.tensor((lo < AIR_CUTOFF_PER_CM) & (hi < AIR_CUTOFF_PER_CM), device=dev)
    f = torch.tensor(x0[None], dtype=torch.float32, device=dev, requires_grad=True)
    ref = stack_kvp({k: torch.tensor(c.transmission[k][None], device=dev) for k in S.KVPS})
    opt = torch.optim.Adam([f], lr=lr)
    for _ in range(iters):
        sub = torch.randperm(256, generator=gen)[:views].to(dev)
        loss = projection_loss(fwd, f, ref, sub)
        if lam > 0:
            loss = loss + lam * tv(f)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            f.clamp_(min=0.0)
            f[:, :, air] = 0.0
    with torch.no_grad():
        full = float(projection_loss(fwd, f, ref).sqrt())
    return f.detach()[0].double().cpu().numpy(), full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--stage", choices=["val", "test"], default="val")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--views", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--n-cases", type=int, default=100)
    a = ap.parse_args()
    run = Path(os.environ.get("RESEARCH_RUN_DIR", "."))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    tables = {k: S.load_model_table(tab / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    mu, store = effective_mu(tables), CaseStore(Path(a.data) / "files")
    fwd = AapmFanbeamForward(tables).to(dev)
    sp = json.loads(a.split.read_text())
    ids = sp[a.stage][: a.n_cases][a.part :: a.parts]
    gen = torch.Generator().manual_seed(0)
    rows, t0 = [], time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in ids:
        c = store[i]
        p, pr = solve(fwd, c, mu, dev, a.lam, a.iters, a.views, a.lr, gen)
        t = c.phantom.astype(np.float64)
        rows.append(
            {
                "case": i,
                "s1": s1_mean_rmse(t, p),
                "s2": s2_worst_roi_rmse(t, p),
                "boundary_rmse": boundary_rmse(t, p),
                "proj_rmse_T": pr,
                "rmse_mat": np.sqrt(((p - t) ** 2).mean((1, 2))).tolist(),
                "tissue_abs_sum_err": partition_errors(t, p)["tissue_mean_abs"],
            }
        )
        print(i, round(rows[-1]["s1"], 4), round(pr, 5), flush=True)
    out = {
        "stage": a.stage,
        "lam": a.lam,
        "iters": a.iters,
        "views": a.views,
        "lr": a.lr,
        "part": a.part,
        "parts": a.parts,
        "n": len(rows),
        "wall_s": time.time() - t0,
        "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
        "s1_mean": float(np.mean([r["s1"] for r in rows])),
        "s2_max": float(np.max([r["s2"] for r in rows])),
        "proj_rmse_T_mean": float(np.mean([r["proj_rmse_T"] for r in rows])),
        "split_manifest_sha256": sp["manifest_sha256"],
        "rows": rows,
    }
    (run / "exp032.json").write_text(json.dumps(out, indent=1))
    with open(run / "artifacts.jsonl", "a") as f:
        f.write(
            json.dumps({"path": str((run / "exp032.json").resolve()), "name": "exp032.json"}) + "\n"
        )
    with open(run / "metrics.jsonl", "a") as f:
        for k in ("s1_mean", "s2_max", "proj_rmse_T_mean"):
            f.write(json.dumps({"name": k, "value": out[k]}) + "\n")
    print(
        json.dumps(
            {k: out[k] for k in ("lam", "n", "s1_mean", "s2_max", "proj_rmse_T_mean", "wall_s")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
