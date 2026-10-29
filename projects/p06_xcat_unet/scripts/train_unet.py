#!/usr/bin/env python
"""Train the 3D U-Net on cached simulated CT (no LEAP in this process). Resumable; model selection on validation patients only.

usage: train_unet.py --arm {A,B,C,D} --cache DIR --train-patients ... --val-patients ... --out RUN_DIR [--steps N] [--max-minutes M]
Arms (same patients, same split): A clean fixed acquisition | B noisy fixed (n0=1e5) | C varied acquisition/noise/kernel | D = C + FiLM conditioning.
Held-out acquisition settings (e80, pitch1p4) are NEVER used for training in any arm.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.seg.dataset import PatchDataset, VolumeRecord
from ctsynth.seg.infer import predict_volume
from ctsynth.seg.losses import confusion_matrix, dice_ce_loss, dice_iou_from_confusion
from ctsynth.seg.unet3d import UNet3D
from ctsynth.xcat.cache import VolumeReader

ARM_VARIANTS = {
    "A": ["clean"],
    "B": ["n0_1e5"],
    "C": [
        "clean",
        "n0_1e6",
        "n0_3e5",
        "n0_1e5",
        "n0_3e4",
        "lowpass_w2",
        "lowpass_w4",
        "apod_0p75",
        "e60",
        "dec2",
    ],
}
ARM_VARIANTS["D"] = ARM_VARIANTS["C"]
HELD_OUT = ["e80", "pitch1p4"]
N_CLASSES = 15


def build_records(cache, patients, variants):
    recs = []
    for pid in patients:
        pdir = Path(cache) / f"p{pid}"
        cs = np.load(pdir / "class_samples.npz")
        samples = {int(k): cs[k] for k in cs.files}
        for v in variants:
            meta_p = pdir / f"ct_{v}.json"
            if not meta_p.exists():
                continue
            meta = json.loads(meta_p.read_text())
            recs.append(
                VolumeRecord(
                    pid,
                    str(pdir / f"ct_{v}.npy"),
                    meta["fmt"],
                    str(pdir / "labels.npy"),
                    tuple(meta["cond"]),
                    samples,
                )
            )
    return recs


@torch.no_grad()
def validate(net, recs, cond_on, dev, stride=64):
    cm = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=dev)
    for r in recs:
        hu = VolumeReader(r.ct_path, r.ct_fmt).read_all()
        cond = torch.tensor([r.cond], dtype=torch.float32) if cond_on else None
        pred = predict_volume(net, hu, 96, stride, 4, cond, dev)
        lab = np.load(r.label_path)
        cm += confusion_matrix(
            torch.from_numpy(pred).to(dev), torch.from_numpy(lab).to(dev), N_CLASSES
        )
    dice, iou = dice_iou_from_confusion(cm.cpu())
    fg = dice[1:]
    return {
        "macro_dice_fg": float(torch.nanmean(fg)),
        "dice": dice.tolist(),
        "iou": iou.tolist(),
    }, cm.cpu()


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
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--val-variants", nargs="*", default=None)
    a = ap.parse_args()
    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    variants = ARM_VARIANTS[a.arm]
    assert not set(variants) & set(HELD_OUT), "held-out acquisition settings must not be trained on"
    cond_on = a.arm == "D"
    train = build_records(a.cache, a.train_patients, variants)
    val_vars = (
        a.val_variants
        or [v for v in variants if v in ("clean", "n0_1e5", "n0_3e5")][:2]
        or variants[:1]
    )
    val = build_records(a.cache, a.val_patients, val_vars)
    cfg = {
        **vars(a),
        "variants": variants,
        "val_variants": val_vars,
        "held_out": HELD_OUT,
        "n_train_volumes": len(train),
        "n_val_volumes": len(val),
        "cond_dim": 7 if cond_on else 0,
        "torch": torch.__version__,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.manual_seed(a.seed)
    net = UNet3D(1, N_CLASSES, base=a.base, depth=3, cond_dim=7 if cond_on else 0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    step, best = 0, -1.0
    ck = out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        net.load_state_dict(s["net"])
        opt.load_state_dict(s["opt"])
        step, best = s["step"], s["best"]
        print(f"resumed from step {step}", flush=True)
    ds = PatchDataset(train, a.patch, a.steps * a.batch, seed=a.seed)
    ds.epoch = 0
    # resume by skipping the already-consumed items: the dataset is deterministic in (seed, epoch, index)
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
            loss = dice_ce_loss(net(x, c if cond_on else None), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 25 == 0:
            rec = {
                "step": step,
                "loss": float(loss),
                "lr": lr,
                "elapsed_s": time.time() - t0,
                "finite": bool(torch.isfinite(loss)),
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
        stop = (time.time() - t_start) / 60 > a.max_minutes
        if step % a.ckpt_every == 0 or step == a.steps or stop:
            torch.save(
                {"net": net.state_dict(), "opt": opt.state_dict(), "step": step, "best": best}, ck
            )
        if step % a.val_every == 0 or step == a.steps or stop:
            v, _ = validate(net, val, cond_on, dev)
            v["step"] = step
            log.write(json.dumps({"val": v}) + "\n")
            log.flush()
            print(
                f"step {step} loss {float(loss):.4f} val macro Dice(fg) {v['macro_dice_fg']:.4f}",
                flush=True,
            )
            if v["macro_dice_fg"] > best:
                best = v["macro_dice_fg"]
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
                "best_val_macro_dice_fg": best,
                "finished": step >= a.steps,
                "wall_minutes": (time.time() - t_start) / 60,
            }
        )
    )


if __name__ == "__main__":
    main()
