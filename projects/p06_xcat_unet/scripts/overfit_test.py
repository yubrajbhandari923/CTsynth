#!/usr/bin/env python
"""Stage-1 sanity check: can the U-Net overfit ONE fixed batch of cached patches? (not a generalisation result)
usage: overfit_test.py --cache DIR --patient 22728 --variant clean --out report.json"""

import argparse
import json
import time

import numpy as np
import torch
from train_unet import build_records

from ctsynth.seg.dataset import PatchDataset
from ctsynth.seg.losses import confusion_matrix, dice_ce_loss, dice_iou_from_confusion
from ctsynth.seg.unet3d import UNet3D

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--patient", default="22728")
ap.add_argument("--variant", default="clean")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=400)
a = ap.parse_args()
dev = torch.device("cuda:0")
rec = build_records(a.cache, [a.patient], [a.variant])
ds = PatchDataset(rec, 96, 4, seed=0, p_foreground=1.0)
x = torch.stack([ds[i][0] for i in range(4)]).to(dev)
y = torch.stack([ds[i][1] for i in range(4)]).to(dev)
torch.manual_seed(0)
net = UNet3D(1, 15, base=16, depth=3).to(dev)
opt = torch.optim.AdamW(net.parameters(), 3e-3)
hist, t0 = [], time.time()
for s in range(a.steps):
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = dice_ce_loss(net(x), y)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
    opt.step()
    if s % 20 == 0 or s == a.steps - 1:
        hist.append({"step": s, "loss": float(loss), "grad_norm": gn})
net.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    pred = net(x).argmax(1)
cm = confusion_matrix(pred, y, 15).cpu()
dice, iou = dice_iou_from_confusion(cm)
res = {
    "patient": a.patient,
    "variant": a.variant,
    "steps": a.steps,
    "seconds": time.time() - t0,
    "loss_first": hist[0]["loss"],
    "loss_last": hist[-1]["loss"],
    "pixel_accuracy": float((pred == y).float().mean()),
    "macro_dice_present_classes": float(
        torch.nanmean(dice[torch.isin(torch.arange(15), y.cpu().unique())])
    ),
    "per_class_dice": dice.tolist(),
    "classes_in_batch": sorted(int(c) for c in y.cpu().unique()),
    "history": hist,
    "all_finite": all(np.isfinite(h["loss"]) for h in hist),
}
open(a.out, "w").write(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k not in ("history", "per_class_dice")})
