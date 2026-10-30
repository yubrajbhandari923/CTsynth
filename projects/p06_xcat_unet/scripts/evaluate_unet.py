#!/usr/bin/env python
"""Evaluate a trained U-Net (best.pt) and/or the intensity-only baseline on held-out patients for every cached variant.

usage: evaluate_unet.py --cache DIR --patients ... --variants ... --out OUT_DIR [--ckpt RUN/best.pt] [--baseline-fit-patients P ...]
Writes per_variant.json (per-class Dice/IoU, macro Dice over foreground classes, confusion matrix), summary.csv, and (with --export N)
NIfTI CT / truth / prediction for N cases plus an overlay PNG. Dice is over ALL voxels of each volume (class 0 = air/background included).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ctsynth.seg.infer import predict_volume
from ctsynth.seg.losses import confusion_matrix, dice_iou_from_confusion
from ctsynth.seg.unet3d import UNet3D
from ctsynth.xcat.cache import VolumeReader

N_CLASSES = 15


def fit_intensity_baseline(cache, patients, variant="clean", per_class=20000, seed=0):
    r = np.random.default_rng(seed)
    vals = {k: [] for k in range(N_CLASSES)}
    for p in patients:
        ct = VolumeReader(Path(cache) / f"p{p}" / f"ct_{variant}.npy", "npy_i16q").read_all()
        lab = np.load(Path(cache) / f"p{p}" / "labels.npy")
        for k in range(N_CLASSES):
            idx = np.flatnonzero(lab == k)
            if idx.size:
                vals[k].append(
                    ct.ravel()[r.choice(idx, size=min(per_class, idx.size), replace=False)]
                )
    mu = np.array([np.concatenate(v).mean() if v else np.nan for v in vals.values()])
    sd = np.array([np.concatenate(v).std() + 1.0 if v else np.nan for v in vals.values()])
    return mu, sd


def predict_baseline(ct, mu, sd):
    """Per-voxel maximum-likelihood class from a 1D Gaussian per class (HU only, no spatial context)."""
    ll = -0.5 * ((ct[..., None] - mu) / sd) ** 2 - np.log(sd)
    ll[..., np.isnan(mu)] = -np.inf
    return ll.argmax(-1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--patients", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline-fit-patients", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument(
        "--export-cases", nargs="*", default=[], help="patient:variant pairs to export as NIfTI"
    )
    ap.add_argument(
        "--export-ct",
        action="store_true",
        help="also export CT and ground truth (once per case, not per model)",
    )
    ap.add_argument("--name", default="model")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0")
    models = {}
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        cond_on = ck["cfg"]["arm"] == "D"
        net = UNet3D(
            1, N_CLASSES, base=ck["cfg"]["base"], depth=3, cond_dim=7 if cond_on else 0
        ).to(dev)
        net.load_state_dict(ck["net"])
        models["unet"] = (net, cond_on, ck["step"])
    base_stats = (
        fit_intensity_baseline(a.cache, a.baseline_fit_patients)
        if a.baseline_fit_patients
        else None
    )
    res = {"unet": {}, "baseline": {}}
    rows = []
    per_patient = {"unet": {}, "baseline": {}}
    for v in a.variants:
        cms = {
            "unet": torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long),
            "baseline": torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long),
        }
        for p in a.patients:
            pdir = Path(a.cache) / f"p{p}"
            if not (pdir / f"ct_{v}.json").exists():
                continue
            meta = json.loads((pdir / f"ct_{v}.json").read_text())
            hu = VolumeReader(pdir / f"ct_{v}.npy", meta["fmt"]).read_all()
            lab = np.load(pdir / "labels.npy")
            lt = torch.from_numpy(lab)
            preds = {}
            if "unet" in models:
                net, cond_on, _ = models["unet"]
                cond = torch.tensor([meta["cond"]], dtype=torch.float32) if cond_on else None
                preds["unet"] = predict_volume(net, hu, 96, a.stride, 4, cond, dev)
            if base_stats is not None:
                preds["baseline"] = predict_baseline(hu, *base_stats)
            for k, pr in preds.items():
                cm_p = confusion_matrix(torch.from_numpy(pr), lt, N_CLASSES)
                cms[k] += cm_p
                per_patient[k].setdefault(v, {})[p] = cm_p.tolist()
            if f"{p}:{v}" in a.export_cases and "unet" in preds:
                import nibabel as nib

                from ctsynth.xcat.phantom import affine_from_axcodes

                aff = affine_from_axcodes(hu.shape, meta["vox_mm"])
                arrays = [("pred_" + a.name, preds["unet"])] + (
                    [("ct", hu), ("truth", lab)] if a.export_ct else []
                )
                for name, arr in arrays:
                    nib.save(nib.Nifti1Image(arr, aff), out / f"p{p}_{v}_{name}.nii.gz")
        for k, cm in cms.items():
            if int(cm.sum()) == 0:
                continue
            dice, iou = dice_iou_from_confusion(cm)
            res[k][v] = {
                "macro_dice_fg": float(torch.nanmean(dice[1:])),
                "dice": dice.tolist(),
                "iou": iou.tolist(),
                "confusion": cm.tolist(),
                "n_voxels": int(cm.sum()),
                "per_patient_confusion": per_patient[k].get(v, {}),
            }
            rows.append(
                {
                    "model": k,
                    "variant": v,
                    "macro_dice_fg": res[k][v]["macro_dice_fg"],
                    **{f"dice_c{i}": float(d) for i, d in enumerate(dice)},
                }
            )
            print(k, v, f"macro Dice(fg) {res[k][v]['macro_dice_fg']:.4f}", flush=True)
    (out / "per_variant.json").write_text(json.dumps(res))
    if rows:
        with open(out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=sorted(
                    {c for r in rows for c in r},
                    key=lambda c: (c not in ("model", "variant", "macro_dice_fg"), c),
                ),
            )
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
