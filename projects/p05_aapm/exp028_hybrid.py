"""EXP-028: physics-gradient refinement of a FROZEN supervised U-Net (hybrid baseline).

Start F0 = U-Net(images) (EXP-022 checkpoint; weights never updated). Then K Adam steps on
  L(F) = mean (T(F) - T_meas)^2 + mu * mean (F - F0)^2     (F >= 0 by projection; mu=0: non-learned)
with the verified differentiable fan-beam forward and random 32-view subsets. Metrics are recorded
at several K from ONE optimisation per case. Nothing is trained: an INSPIRED baseline, not a
reproduction of any published method. Validation only unless --stage test is given explicitly.
Usage: exp028_hybrid.py --data D --split S --ckpt PATH --mu 0 --ks 0,10,25,50,100,200
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.aapm import schema as S
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.metrics import boundary_rmse, s1_mean_rmse, s2_worst_roi_rmse
from ctsynth.aapm.physics_loss import projection_loss, stack_kvp
from ctsynth.aapm.unet import UNet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--stage", choices=["val", "test"], default="val")
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--ks", default="0,10,25,50,100,200")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--views", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n-cases", type=int, default=100)
    a = ap.parse_args()
    run = Path(os.environ.get("RESEARCH_RUN_DIR", "."))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ks = sorted({int(k) for k in a.ks.split(",")})
    tab = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    tables = {k: S.load_model_table(tab / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    store, fwd = CaseStore(Path(a.data) / "files"), AapmFanbeamForward(tables).to(dev)
    net = UNet(depth=a.depth, width=a.width, output="softplus").to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    sp = json.loads(a.split.read_text())
    ids = sp[a.stage][: a.n_cases]
    gen = torch.Generator().manual_seed(0)
    rows, t0 = [], time.time()
    for i in ids:
        c = store[i]
        t = c.phantom.astype(np.float64)
        img = torch.tensor(np.stack([c.image[50], c.image[80]])[None], device=dev)
        ref = stack_kvp({k: torch.tensor(c.transmission[k][None], device=dev) for k in S.KVPS})
        with torch.no_grad():
            f0 = net(img)
        f = f0.clone().requires_grad_(True)
        opt = torch.optim.Adam([f], lr=a.lr)
        rec = {}
        for k in range(max(ks) + 1):
            if k in ks:
                with torch.no_grad():
                    p = f.detach()[0].double().cpu().numpy()
                    rec[k] = {
                        "s1": s1_mean_rmse(t, p),
                        "s2": s2_worst_roi_rmse(t, p),
                        "boundary_rmse": boundary_rmse(t, p),
                        "proj_rmse_T": float(projection_loss(fwd, f, ref).sqrt()),
                        "rmse_mat": np.sqrt(((p - t) ** 2).mean((1, 2))).tolist(),
                    }
            sub = torch.randperm(256, generator=gen)[: a.views].to(dev)
            loss = projection_loss(fwd, f, ref, sub)
            if a.mu:
                loss = loss + a.mu * ((f - f0) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                f.clamp_(min=0.0)
        rows.append({"case": i, "by_k": rec})
        print(i, {k: round(v["s1"], 4) for k, v in rec.items()}, flush=True)
    summ = {
        k: {
            "s1": float(np.mean([r["by_k"][k]["s1"] for r in rows])),
            "s2": float(np.max([r["by_k"][k]["s2"] for r in rows])),
            "boundary_rmse": float(np.mean([r["by_k"][k]["boundary_rmse"] for r in rows])),
            "proj_rmse_T": float(np.mean([r["by_k"][k]["proj_rmse_T"] for r in rows])),
        }
        for k in ks
    }
    out = {
        "stage": a.stage,
        "mu": a.mu,
        "lr": a.lr,
        "ks": ks,
        "ckpt": str(a.ckpt),
        "n": len(rows),
        "wall_s": time.time() - t0,
        "summary": summ,
        "rows": rows,
        "split_manifest_sha256": sp["manifest_sha256"],
    }
    (run / "exp028.json").write_text(json.dumps(out, indent=1))
    with open(run / "artifacts.jsonl", "a") as f_:
        f_.write(
            json.dumps({"path": str((run / "exp028.json").resolve()), "name": "exp028.json"}) + "\n"
        )
    with open(run / "metrics.jsonl", "a") as f_:
        for k, v in summ.items():
            f_.write(json.dumps({"name": f"s1_K{k}", "value": v["s1"]}) + "\n")
            f_.write(json.dumps({"name": f"proj_rmse_K{k}", "value": v["proj_rmse_T"]}) + "\n")
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
