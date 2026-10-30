#!/usr/bin/env python
"""Train the physical-phantom U-Net on cached simulated CT + physical targets (no LEAP in this process). Resumable.

usage: train_phys.py --arm {A,B,C,D} --cache DIR --train-patients ... --val-patients ... --out RUN_DIR [--steps N] [--max-minutes M]
Arms (same patients/split/optimiser budget as EXP-039): A clean | B noisy n0=1e5 | C varied (clean, n0_1e5, var1) | D = C + FiLM.
Held-out acquisition settings (e80, pitch1p4) are never trained on. Model selection: lowest validation sum_k MAE_k (g/cm3) on val patients.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.seg.physical import PhysPatchDataset, PhysRecord, PhysUNet, phys_loss, predict_conc
from ctsynth.xcat.cache import VolumeReader
from ctsynth.xcat.materials import ElementTable
from ctsynth.xcat.physical import load_basis, u16_to_conc

ARM_VARIANTS = {
    "A": ["clean"],
    "B": ["n0_1e5"],
    "C": ["clean", "n0_1e5", "var1"],
    "D": ["clean", "n0_1e5", "var1"],
}
HELD_OUT = ["e80", "pitch1p4"]


def build_records(cache, patients, variants):
    recs = []
    for pid in patients:
        pdir = Path(cache) / f"p{pid}"
        cs = np.load(pdir / "class_samples.npz")
        samples = {int(k): cs[k] for k in cs.files}
        for v in variants:
            if (pdir / f"ct_{v}.json").exists():
                meta = json.loads((pdir / f"ct_{v}.json").read_text())
                recs.append(
                    PhysRecord(
                        pid,
                        str(pdir / f"ct_{v}.npy"),
                        meta["fmt"],
                        str(pdir / "conc_v1.npy"),
                        tuple(meta["cond"]),
                        samples,
                    )
                )
    return recs


@torch.no_grad()
def validate(net, recs, cond_on, dev, stride, kappa70, mu_w_cm):
    tot = {"sum_mae_gcm3": 0.0, "density_mae_gcm3": 0.0, "mu70_mae_hu": 0.0}
    for r in recs:
        hu = VolumeReader(r.ct_path, r.ct_fmt).read_all()
        cond = torch.tensor([r.cond], dtype=torch.float32) if cond_on else None
        pred = predict_conc(net, hu, 96, stride, 4, cond, dev)
        c = torch.from_numpy(u16_to_conc(np.load(r.conc_path, mmap_mode="r")[:])).to(dev)
        tot["sum_mae_gcm3"] += float((pred - c).abs().mean((1, 2, 3)).sum())
        tot["density_mae_gcm3"] += float((pred.sum(0) - c.sum(0)).abs().mean())
        tot["mu70_mae_hu"] += float(
            ((pred - c) * kappa70.reshape(-1, 1, 1, 1)).sum(0).abs().mean() / mu_w_cm * 1000.0
        )
        del pred, c
    return {k: v / len(recs) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_VARIANTS))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--train-patients", nargs="+", required=True)
    ap.add_argument("--val-patients", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--val-stride", type=int, default=96)
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    variants = ARM_VARIANTS[a.arm]
    assert not set(variants) & set(HELD_OUT), "held-out acquisition settings must not be trained on"
    cond_on = a.arm == "D"
    basis, et = load_basis(), ElementTable()
    kappa70 = torch.from_numpy(basis.kappa_cm2g(70.0, et)).float().to(dev)
    mu_w_cm = et.kappa_water(70.0)
    train = build_records(a.cache, a.train_patients, variants)
    val_vars = [v for v in variants if v in ("clean", "n0_1e5")][:2] or variants[:1]
    val = build_records(a.cache, a.val_patients, val_vars)
    cfg = {
        **vars(a),
        "variants": variants,
        "val_variants": val_vars,
        "held_out": HELD_OUT,
        "n_train_volumes": len(train),
        "n_val_volumes": len(val),
        "cond_dim": 7 if cond_on else 0,
        "k": basis.k,
        "channel_names": list(basis.names),
        "huber_delta": 0.02,
        "torch": torch.__version__,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.manual_seed(a.seed)
    net = PhysUNet(basis.k, a.base, 3, 7 if cond_on else 0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    step, best = 0, float("inf")
    ck = out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        step, best = s["step"], s["best"]
        print(f"resumed from step {step}", flush=True)
    ds = PhysPatchDataset(train, a.patch, a.steps * a.batch, seed=a.seed)
    idx = list(range(step * a.batch, a.steps * a.batch))
    dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx),
        batch_size=a.batch,
        num_workers=a.workers,
        pin_memory=True,
        persistent_workers=a.workers > 0,
        prefetch_factor=4 if a.workers else None,
    )
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    for x, y, c in dl:
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, c = (
            x.to(dev, non_blocking=True),
            y.to(dev, non_blocking=True),
            c.to(dev, non_blocking=True),
        )
        net.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x, c if cond_on else None)
        L = phys_loss(pred.float(), y, kappa70, mu_w_cm)
        L["total"].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 25 == 0:
            log.write(
                json.dumps(
                    {
                        "step": step,
                        **{k: float(v) for k, v in L.items()},
                        "lr": lr,
                        "elapsed_s": time.time() - t0,
                        "finite": bool(torch.isfinite(L["total"])),
                        "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2**20,
                    }
                )
                + "\n"
            )
            log.flush()
        stop = (time.time() - t_start) / 60 > a.max_minutes
        if step % a.ckpt_every == 0 or step == a.steps or stop:
            torch.save(
                {"net": net.state_dict(), "opt": opt.state_dict(), "step": step, "best": best}, ck
            )
        if step % a.val_every == 0 or step == a.steps or stop:
            v = validate(net, val, cond_on, dev, a.val_stride, kappa70, mu_w_cm)
            v["step"] = step
            log.write(json.dumps({"val": v}) + "\n")
            log.flush()
            print(
                f"step {step} loss {float(L['total']):.4f} val sum-MAE {v['sum_mae_gcm3']:.4f} g/cm3, density {v['density_mae_gcm3']:.4f}, mu70 {v['mu70_mae_hu']:.1f} HU",
                flush=True,
            )
            if v["sum_mae_gcm3"] < best:
                best = v["sum_mae_gcm3"]
                torch.save(
                    {"net": net.state_dict(), "step": step, "val": v, "cfg": cfg}, out / "best.pt"
                )
                torch.save(
                    {"net": net.state_dict(), "opt": opt.state_dict(), "step": step, "best": best},
                    ck,
                )
        if stop:
            print("time limit reached: checkpoint saved, resubmit to resume", flush=True)
            break
    (out / "status.json").write_text(
        json.dumps(
            {
                "step": step,
                "best_val_sum_mae_gcm3": best,
                "finished": step >= a.steps,
                "wall_minutes": (time.time() - t_start) / 60,
            }
        )
    )


if __name__ == "__main__":
    main()
