"""EXP-018: gradients of the projection-domain loss w.r.t. material fractions vs finite differences.

Criteria are pre-registered in the experiment record; the U-Net is not involved.
Usage: exp018_gradients.py [--data <dataset dir>]   (--data adds the real 512^2 check)
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
from ctsynth.aapm.physics_loss import projection_loss, stack_kvp

EPS = 1e-6


def valid_fractions(g, seed, b=1):
    r = np.random.default_rng(seed)
    x = r.random((b, 3, g.n_pix, g.n_pix))
    return torch.tensor(x / (x.sum(1, keepdims=True) + r.random((b, 1, g.n_pix, g.n_pix))))  # sum<1


def fd_check(fwd, x, ref, sub, log, n_pick, seed, floor=1e-8):
    x = x.clone().requires_grad_(True)
    loss = projection_loss(fwd, x, ref, sub, log=log, floor=floor)
    (g,) = torch.autograd.grad(loss, x)
    r = np.random.default_rng(seed)
    idx = [
        (int(r.integers(3)), int(r.integers(x.shape[2])), int(r.integers(x.shape[3])))
        for _ in range(n_pick)
    ]
    gmax = float(g.abs().max())
    rel = []
    for m, i, j in idx:
        xp, xm = x.detach().clone(), x.detach().clone()
        xp[0, m, i, j] += EPS
        xm[0, m, i, j] -= EPS
        with torch.no_grad():
            fd = (
                projection_loss(fwd, xp, ref, sub, log=log, floor=floor)
                - projection_loss(fwd, xm, ref, sub, log=log, floor=floor)
            ).item() / (2 * EPS)
        a = float(g[0, m, i, j])
        rel.append(abs(a - fd) / max(abs(a), abs(fd), 1e-12 * gmax))
    return g, {
        "median_rel": float(np.median(rel)),
        "max_rel": float(np.max(rel)),
        "n": len(rel),
        "grad_abs_max": gmax,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    a = ap.parse_args()
    run = Path(os.environ.get("RESEARCH_RUN_DIR", "."))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    tables = {k: S.load_model_table(tab / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    res: dict = {"device": str(dev), "checks": {}}
    ok = True

    g = S.GeometrySpec(n_pix=24, fov_cm=18.0, n_views=32, n_det=40)
    fwd = AapmFanbeamForward(tables, g, torch.float64).to(dev)
    x0 = valid_fractions(g, 1).to(dev)
    xt = valid_fractions(g, 2).to(dev)
    with torch.no_grad():
        ref = stack_kvp(fwd(xt))
    sub = torch.arange(0, 16, 2, device=dev)
    grads = {}
    for log in (False, True):
        for floor in (1e-8, 1e-4) if log else (1e-8,):
            name = f"tiny_{'logT' if log else 'T'}_floor{floor:g}"
            gr, st = fd_check(fwd, x0, ref, sub, log, 60, 3, floor)
            grads[name] = gr
            per_mat = [float(gr[0, m].abs().sum()) for m in range(3)]
            st.update(per_material_abs_sum=per_mat, finite=bool(torch.isfinite(gr).all()))
            res["checks"][name] = st
            ok &= (
                st["median_rel"] < 1e-6
                and st["max_rel"] < 1e-4
                and st["finite"]
                and min(per_mat) > 0
            )

    # precision and batch invariance (T loss)
    f32 = AapmFanbeamForward(tables, g, torch.float32).to(dev)
    xx = x0.float().clone().requires_grad_(True)
    (g32,) = torch.autograd.grad(projection_loss(f32, xx, ref.float(), sub), xx)
    g64 = grads["tiny_T_floor1e-08"]
    res["float32_vs_float64_grad_rel_l2"] = float((g32.double() - g64).norm() / g64.norm())
    xb = torch.cat([x0, xt]).clone().requires_grad_(True)
    refb = torch.cat([ref, ref])
    (gb,) = torch.autograd.grad(projection_loss(fwd, xb, refb, sub), xb)
    xs = x0.clone().requires_grad_(True)
    (g1,) = torch.autograd.grad(projection_loss(fwd, xs, ref, sub), xs)
    xs2 = xt.clone().requires_grad_(True)
    (g2,) = torch.autograd.grad(projection_loss(fwd, xs2, ref, sub), xs2)
    res["batch_invariance_rel_l2"] = float((gb - 0.5 * torch.cat([g1, g2])).norm() / gb.norm())
    ok &= res["batch_invariance_rel_l2"] < 1e-10

    # direct optimisation from a valid wrong init
    p = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=5e-3)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = projection_loss(fwd, p, ref, sub)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    ma = np.convolve(losses, np.ones(20) / 20, mode="valid")
    res["optim"] = {"initial": losses[0], "final": losses[-1], "reduction": 1 - losses[-1] / losses[0],
                    "moving_avg_nonincreasing": bool(np.all(np.diff(ma) <= 1e-12)),
                    "finite": bool(np.isfinite(losses).all())}  # fmt: skip
    ok &= (
        res["optim"]["reduction"] >= 0.5
        and res["optim"]["moving_avg_nonincreasing"]
        and res["optim"]["finite"]
    )
    res["optim"]["curve"] = losses[::10]

    if a.data:  # real geometry, subset of rays
        st_ = CaseStore(a.data / "files")
        c = st_[0]
        fw = AapmFanbeamForward(tables, dtype=torch.float64).to(dev)
        x = torch.tensor(c.phantom[None], dtype=torch.float64, device=dev) * 0.9
        r = stack_kvp(
            {
                k: torch.tensor(c.transmission[k][None], dtype=torch.float64, device=dev)
                for k in S.KVPS
            }
        )
        sb = torch.arange(0, 256, 16, device=dev)
        for log in (False, True):
            _, st = fd_check(fw, x, r, sb, log, 12, 5)
            res["checks"][f"real512_{'logT' if log else 'T'}"] = st
            ok &= st["median_rel"] < 1e-6 and st["max_rel"] < 1e-4
    res["pass"] = bool(ok)
    (run / "exp018_results.json").write_text(json.dumps(res, indent=1, default=float))
    with open(run / "artifacts.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "path": str((run / "exp018_results.json").resolve()),
                    "name": "exp018_results.json",
                }
            )
            + "\n"
        )
    with open(run / "metrics.jsonl", "a") as f:
        for k, v in res["checks"].items():
            f.write(json.dumps({"name": f"{k}_median_rel", "value": v["median_rel"]}) + "\n")
            f.write(json.dumps({"name": f"{k}_max_rel", "value": v["max_rel"]}) + "\n")
        for k in ("float32_vs_float64_grad_rel_l2", "batch_invariance_rel_l2"):
            f.write(json.dumps({"name": k, "value": res[k]}) + "\n")
        f.write(json.dumps({"name": "optim_reduction", "value": res["optim"]["reduction"]}) + "\n")
        f.write(json.dumps({"name": "pass", "value": float(res["pass"])}) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "optim"}, indent=1, default=float))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
