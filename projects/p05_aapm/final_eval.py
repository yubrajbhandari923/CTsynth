"""Final benchmark: ONE evaluation of frozen U-Net checkpoints on the held-out TEST split.

Every config (architecture, output parameterisation, checkpoint) must be frozen before this runs;
it never trains or selects. Per checkpoint: official s1/s2 at 512x512, per-material, interior and
boundary RMSE, full-view projection RMSE, null-direction error fraction, per-case values.
`--refine-k/--refine-mu` optionally applies the EXP-028 physics refinement with frozen settings.
Usage: final_eval.py --data D --split S --ckpt PATH --name NAME [--output softplus --depth 4]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ctsynth.aapm import schema as S
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.metrics import (
    boundary_rmse,
    partition_errors,
    regime_masks,
    s1_mean_rmse,
    s2_worst_roi_rmse,
)
from ctsynth.aapm.physics_loss import projection_loss, stack_kvp
from ctsynth.aapm.unet import UNet

NULL = np.array([-0.723, 0.691, -0.008])
NULL = NULL / np.linalg.norm(NULL)


def refine(fwd, f0, ref, k, mu, lr, views, gen, dev):
    f = f0.clone().requires_grad_(True)
    opt = torch.optim.Adam([f], lr=lr)
    for _ in range(k):
        sub = torch.randperm(256, generator=gen)[:views].to(dev)
        loss = projection_loss(fwd, f, ref, sub)
        if mu:
            loss = loss + mu * ((f - f0) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            f.clamp_(min=0.0)
    return f.detach()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--output", default="softplus")
    ap.add_argument("--oracle-occ", action="store_true")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--refine-k", type=int, default=0)
    ap.add_argument("--refine-mu", type=float, default=0.0)
    ap.add_argument("--n-cases", type=int, default=100)
    a = ap.parse_args()
    run = Path(os.environ.get("RESEARCH_RUN_DIR", "."))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    tables = {k: S.load_model_table(tab / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    store, fwd = (
        CaseStore(Path(a.data) / "files"),
        AapmFanbeamForward(tables, exp_clip=20.0).to(dev),
    )
    net = UNet(depth=a.depth, width=a.width, output=a.output).to(dev)
    net.load_state_dict(torch.load(a.ckpt, map_location=dev))
    net.eval()
    sp = json.loads(a.split.read_text())
    ids = sp["test"][: a.n_cases]
    gen = torch.Generator().manual_seed(0)
    rows, sq = [], {"air": 0.0, "tissue": 0.0, "band": 0.0}
    cnt = {"air": 0, "tissue": 0, "band": 0}
    sqmat = np.zeros(3)
    for i in ids:
        c = store[i]
        t = c.phantom.astype(np.float64)
        img = torch.tensor(np.stack([c.image[50], c.image[80]])[None], device=dev)
        ref = stack_kvp({k: torch.tensor(c.transmission[k][None], device=dev) for k in S.KVPS})
        with torch.no_grad():
            f = net(img)
            if a.oracle_occ:
                f = f * torch.tensor(t.sum(0, keepdims=True)[None], device=dev).float().clamp(0, 1)
        if a.refine_k:
            f = refine(fwd, f, ref, a.refine_k, a.refine_mu, 0.005, 32, gen, dev)
        with torch.no_grad():
            pr = float(projection_loss(fwd, f, ref).sqrt())
        p = f[0].double().cpu().numpy()
        e = p - t
        air, tis, band = regime_masks(t[None])
        for k, m in (("air", air[0]), ("tissue", tis[0]), ("band", band[0])):
            sq[k] += float((e[:, m] ** 2).sum())
            cnt[k] += int(m.sum()) * 3
        sqmat += (e**2).sum((1, 2))
        rows.append(
            {
                "case": i,
                "s1": s1_mean_rmse(t, p),
                "s2": s2_worst_roi_rmse(t, p),
                "boundary_rmse": boundary_rmse(t, p),
                "proj_rmse_T": pr,
                "null_frac": float(
                    np.abs(np.einsum("mhw,m->hw", e, NULL)).sum() / np.sqrt((e**2).sum(0)).sum()
                ),
                "tissue_abs_sum_err": partition_errors(t, p)["tissue_mean_abs"],
            }
        )
        print(i, round(rows[-1]["s1"], 5), flush=True)
    n = len(rows)
    out = {
        "name": a.name,
        "ckpt": str(a.ckpt),
        "n_test_cases": n,
        "output": a.output,
        "oracle_occupancy": a.oracle_occ,
        "refine_k": a.refine_k,
        "refine_mu": a.refine_mu,
        "s1": float(np.mean([r["s1"] for r in rows])),
        "s2": float(np.max([r["s2"] for r in rows])),
        "boundary_rmse": float(np.mean([r["boundary_rmse"] for r in rows])),
        "proj_rmse_T": float(np.mean([r["proj_rmse_T"] for r in rows])),
        "null_frac": float(np.mean([r["null_frac"] for r in rows])),
        "rmse_per_material": dict(
            zip(S.MATERIALS, np.sqrt(sqmat / (n * 512 * 512)).tolist(), strict=True)
        ),
        "rmse_interior": float(np.sqrt(sq["tissue"] / cnt["tissue"])),
        "rmse_boundary": float(np.sqrt(sq["band"] / cnt["band"])),
        "rmse_air": float(np.sqrt(sq["air"] / max(cnt["air"], 1))),
        "split_manifest_sha256": sp["manifest_sha256"],
        "rows": rows,
    }
    fn = f"final_{a.name}.json"
    (run / fn).write_text(json.dumps(out, indent=1))
    with open(run / "artifacts.jsonl", "a") as f_:
        f_.write(json.dumps({"path": str((run / fn).resolve()), "name": fn}) + "\n")
    with open(run / "metrics.jsonl", "a") as f_:
        for k in ("s1", "s2", "boundary_rmse", "proj_rmse_T", "null_frac"):
            f_.write(json.dumps({"name": f"{a.name}_{k}", "value": out[k]}) + "\n")
    print(json.dumps({k: out[k] for k in ("name", "s1", "s2", "proj_rmse_T", "null_frac")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
