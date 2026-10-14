"""Milestone A trainer (EXP-019 supervised smoke, EXP-020 arms).

U-Net -> fractions -> fan-beam polychromatic forward. Arms: supervised (label MSE); physics
(projection loss ONLY: no truth in loss, selection or targets); physics_feas (projection loss +
nonneg + sum<=1 hinge: feasibility only, does not select the null direction). Truth is loaded for
evaluation but enters the training step only in the supervised arm.
Usage: milestone_a.py --data <dataset dir> --split <split.json> --arm physics --steps 300 ...
"""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.aapm import schema as S
from ctsynth.aapm.fbp import AapmFBP
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.losses import entropy, tv
from ctsynth.aapm.metrics import (
    boundary_rmse,
    partition_errors,
    per_material_errors,
    s1_mean_rmse,
    s2_worst_roi_rmse,
)
from ctsynth.aapm.physics_loss import projection_loss, stack_kvp
from ctsynth.aapm.unet import UNet

NULL = np.array([-0.723, 0.691, -0.008])  # F-012 null direction of the 2x3 attenuation system
NULL = NULL / np.linalg.norm(NULL)


def load(store: CaseStore, idx: list[int], dev: torch.device):
    cs = [store[i] for i in idx]
    img = torch.tensor(np.stack([[c.image[50], c.image[80]] for c in cs]))
    tr = torch.tensor(np.stack([[c.transmission[50], c.transmission[80]] for c in cs]))
    tru = torch.tensor(np.stack([c.phantom for c in cs]))
    return img.to(dev), tr.to(dev), tru.to(dev)


def pred(model, x, tru, oracle):
    """Model fractions; `oracle`: times the TRUE per-pixel sum S (upper-bound control only)."""
    f = model(x)
    return f * tru.sum(1, keepdim=True).clamp(0, 1) if oracle else f


def partition_penalty(frac, img):
    """Soft partition: (S-1)^2 on tissue, S^2 on air; regimes from the MEASURED images."""
    air = (img < 0.1).all(1)
    s = frac.sum(1)
    return ((s - 1) ** 2)[~air].mean() + (s**2)[air].mean()


PRIORS = {
    "tv": lambda f: tv(f),
    "entropy": lambda f: entropy(f),
    "range_hinge": lambda f: torch.relu(f - 1).mean(),
    "calc_l1": lambda f: f[:, 2].mean(),
}


def feasibility(frac: torch.Tensor) -> torch.Tensor:
    return torch.relu(-frac).mean() + torch.relu(frac.sum(1) - 1).mean()


def pred_tissue_std(t: np.ndarray, p: np.ndarray) -> float:
    """Mean over cases of the spatial std of predicted adipose in truth-tissue pixels (evaluation
    only): ~0 means a uniform, anatomy-free output (collapse); truth itself has std ~0.5."""
    tissue = np.abs(t.sum(1) - 1) <= 1e-3
    return float(np.mean([p[c, 0][tissue[c]].std() for c in range(p.shape[0])]))


@torch.no_grad()
def evaluate(model, fwd, img, tr, tru, bs=2, oracle=False):
    model.eval()
    preds, pe = [], []
    for s in range(0, img.shape[0], bs):
        f = pred(model, img[s : s + bs], tru[s : s + bs], oracle)
        preds.append(f.cpu())
        pe.append(((stack_kvp(fwd(f)) - tr[s : s + bs]) ** 2).mean(dim=(1, 2, 3)).cpu())
    p, t = torch.cat(preds).numpy().astype(np.float64), tru.cpu().numpy().astype(np.float64)
    e = p - t
    out = {
        "s1": s1_mean_rmse(t, p),
        "s2": s2_worst_roi_rmse(t, p),
        "pred_adipose_std_in_tissue": pred_tissue_std(t, p),
        "boundary_rmse": boundary_rmse(t, p),
        "per_material": per_material_errors(t, p),
        "partition": partition_errors(t, p),
        "proj_rmse_T": float(torch.cat(pe).mean().sqrt()),
        "null_dir_error_frac": float(
            np.abs(np.einsum("cmhw,m->chw", e, NULL)).sum() / np.sqrt((e**2).sum(1)).sum()
        ),
    }
    return out, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", type=Path, help="split manifest json (train/val index lists)")
    ap.add_argument("--cases", type=int, nargs="*", help="explicit case indices (smoke: train=val)")
    ap.add_argument(
        "--arm",
        choices=["supervised", "physics", "physics_feas", "image_phys", "recon_phys"],
        required=True,
    )
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--views", type=int, default=32, help="view positions per step (stochastic)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-feas", type=float, default=1.0)
    ap.add_argument(
        "--output",
        default="softplus",
        choices=["linear", "softplus", "softmax", "occ_softmax"],
    )
    ap.add_argument("--oracle-occ", action="store_true", help="UPPER-BOUND CONTROL: x truth sum S")
    ap.add_argument("--w-partition", type=float, default=0.0, help="soft partition penalty weight")
    ap.add_argument(
        "--prior", action="append", default=[], help="NAME=WEIGHT, NAME in " + ",".join(PRIORS)
    )
    ap.add_argument("--exp-clip", type=float, default=None, help="guard for negative fractions")
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cosine", action="store_true", help="cosine lr decay over --steps")
    ap.add_argument("--warmup", type=int, default=0, help="linear lr warmup steps")
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip (0 = off)")
    a = ap.parse_args()
    run = Path(os.environ.get("RESEARCH_RUN_DIR", "."))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    tab = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    tables = {k: S.load_model_table(tab / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    store = CaseStore(a.data / "files")
    if a.cases:
        tr_idx = va_idx = a.cases
    else:
        sp = json.loads(a.split.read_text())  # type: ignore[union-attr]
        tr_idx, va_idx = sp["train"][: a.n_train], sp["val"][: a.n_val]
    img_tr, T_tr, tru_tr = load(store, tr_idx, dev)
    img_va, T_va, tru_va = load(store, va_idx, dev)
    fwd = AapmFanbeamForward(tables, exp_clip=a.exp_clip).to(dev)
    fbp = AapmFBP(device=dev) if a.arm == "recon_phys" else None

    def arm_loss(frac, img, T, sub):
        """Measurement-domain loss of the chosen label-free arm (no truth)."""
        if a.arm == "image_phys":  # approximate: spectrum-weighted linear attenuation image model
            return ((fwd.spectral.image_forward(frac) - img) ** 2).mean()
        if (
            a.arm == "recon_phys"
        ):  # official image formation: simulate T, -log, FBP, compare to images
            t = fwd(frac)
            sim = torch.stack([fbp(t[k], k, sparse=True) for k in S.KVPS], dim=1)
            return ((sim - img) ** 2).mean()
        return projection_loss(fwd, frac, T, sub)

    model = UNet(depth=a.depth, width=a.width, output=a.output).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def lr_factor(i: int) -> float:  # i = steps taken so far
        w = min(1.0, (i + 1) / a.warmup) if a.warmup else 1.0
        c = 0.5 * (1 + np.cos(np.pi * min(i, a.steps) / a.steps)) if a.cosine else 1.0
        return float(w * c)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor) if (a.cosine or a.warmup) else None
    val_sub = torch.arange(0, 256, 8, device=dev)  # fixed validation views (physics selection)
    hist: dict[str, list] = {
        "step": [],
        "train_loss": [],
        "grad_norm": [],
        "val_step": [],
        "val_proj": [],
        "val_label_mse": [],
    }
    best, best_state, t0 = float("inf"), None, time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, a.steps + 1):
        model.train()
        ix = torch.randperm(img_tr.shape[0], generator=gen)[: a.batch].to(dev)
        frac = pred(model, img_tr[ix], tru_tr[ix], a.oracle_occ)
        if a.arm == "supervised":
            loss = ((frac - tru_tr[ix]) ** 2).mean()
        else:
            sub = torch.randperm(256, generator=gen)[: a.views].to(dev)
            loss = arm_loss(frac, img_tr[ix], T_tr[ix], sub)
            if a.arm == "physics_feas":
                loss = loss + a.w_feas * feasibility(frac)
            for item in a.prior:
                name, w = item.split("=")
                loss = loss + float(w) * PRIORS[name](frac)
            if a.w_partition:
                loss = loss + a.w_partition * partition_penalty(frac, img_tr[ix])
        opt.zero_grad()
        loss.backward()
        gn = float(
            torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))
        )
        if not (np.isfinite(gn) and np.isfinite(loss.item())):
            raise RuntimeError(f"non-finite loss/grad at step {step}")
        if a.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        if sched:
            sched.step()
        hist["step"].append(step)
        hist["train_loss"].append(float(loss))
        hist["grad_norm"].append(gn)
        if step % a.val_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                fv = torch.cat(
                    [
                        pred(model, img_va[s : s + 2], tru_va[s : s + 2], a.oracle_occ)
                        for s in range(0, img_va.shape[0], 2)
                    ]
                )
                vp = float(
                    np.mean(
                        [
                            projection_loss(fwd, fv[s : s + 1], T_va[s : s + 1], val_sub).item()
                            for s in range(fv.shape[0])
                        ]
                    )
                )
                vl = float(
                    ((fv - tru_va) ** 2).mean()
                )  # logged for ALL arms, used for selection only by supervised
            hist["val_step"].append(step)
            hist["val_proj"].append(vp)
            hist["val_label_mse"].append(vl)
            crit = vl if a.arm == "supervised" else vp
            if crit < best:
                best, best_state = (
                    crit,
                    {k: v.detach().clone() for k, v in model.state_dict().items()},
                )
            print(
                f"step {step} loss {float(loss):.3e} gn {gn:.3e} "
                f"val_proj {vp:.3e} val_label {vl:.3e}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    runtime = time.time() - t0
    peak = (
        torch.cuda.max_memory_allocated() / 2**20
        if dev.type == "cuda"
        else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    res, pred_va = evaluate(model, fwd, img_va, T_va, tru_va, oracle=a.oracle_occ)
    zero_mse = float((tru_va**2).mean())  # MSE of predicting nothing
    res["zero_pred_label_mse"], res["best_val_label_mse"] = zero_mse, min(hist["val_label_mse"])
    res["collapsed_uniform"] = bool(res["pred_adipose_std_in_tissue"] < 0.05)
    res["collapsed_to_zero"] = bool(min(hist["val_label_mse"]) >= 0.99 * zero_mse)
    res.update(
        oracle_occupancy=a.oracle_occ,
        priors=a.prior,
        arm=a.arm,
        steps=a.steps,
        selection="val label MSE" if a.arm == "supervised" else "val projection loss",
        arm_loss_note="image_phys/recon_phys: own-domain loss; selection: val projection loss",
        runtime_s=runtime,
        peak_mem_mib=peak,
        device=str(dev),
        train_cases=tr_idx,
        val_cases=va_idx,
        params=int(sum(p.numel() for p in model.parameters())),
        args={k: str(v) for k, v in vars(a).items()},
    )
    (run / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (run / "history.json").write_text(json.dumps(hist))
    torch.save(model.state_dict(), run / "checkpoint.pt")
    np.savez_compressed(
        run / "val_pred.npz", pred=pred_va.astype(np.float32), truth=tru_va.cpu().numpy()
    )
    figures(run, model, fwd, img_va, T_va, tru_va, hist, a.oracle_occ)
    arts = [
        "results.json",
        "history.json",
        "checkpoint.pt",
        "val_pred.npz",
        "figures.png",
        "curves.png",
    ]
    with open(run / "artifacts.jsonl", "a") as f:
        for n in arts:
            f.write(json.dumps({"path": str((run / n).resolve()), "name": n}) + "\n")
    with open(run / "metrics.jsonl", "a") as f:
        for k in (
            "s1",
            "s2",
            "boundary_rmse",
            "proj_rmse_T",
            "null_dir_error_frac",
            "runtime_s",
            "peak_mem_mib",
        ):
            f.write(json.dumps({"name": k, "value": res[k]}) + "\n")
        for m, d in res["per_material"].items():
            f.write(json.dumps({"name": f"rmse_{m}", "value": d["rmse"]}) + "\n")
        f.write(json.dumps({"name": "max_grad_norm", "value": max(hist["grad_norm"])}) + "\n")
    print(json.dumps({k: res[k] for k in ("s1", "proj_rmse_T", "null_dir_error_frac")}, indent=1))
    return 0


def figures(run, model, fwd, img, T, tru, hist, oracle=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        f = pred(model, img[:1], tru[:1], oracle)
        tp = stack_kvp(fwd(f))[0].cpu().numpy()
    f, t0, im, tt = (
        f[0].cpu().numpy(),
        tru[0].cpu().numpy(),
        img[0].cpu().numpy(),
        T[0].cpu().numpy(),
    )
    fig, ax = plt.subplots(4, 4, figsize=(16, 14))
    for k in range(2):
        ax[0, k].imshow(im[k].T, cmap="gray", origin="lower")
        ax[0, k].set_title(f"input image {S.KVPS[k]} kVp [cm^-1]")
    for m, nm in enumerate(S.MATERIALS):
        ax[1, m].imshow(t0[m].T, vmin=0, vmax=1, origin="lower")
        ax[1, m].set_title(f"truth {nm}")
        ax[2, m].imshow(f[m].T, vmin=0, vmax=1, origin="lower")
        ax[2, m].set_title(f"pred {nm}")
        e = ax[3, m].imshow((f[m] - t0[m]).T, cmap="RdBu_r", vmin=-0.5, vmax=0.5, origin="lower")
        ax[3, m].set_title(f"error {nm}")
        fig.colorbar(e, ax=ax[3, m])
    for k in range(2):
        ax[0, 2 + k].imshow(tt[k] - tp[k], aspect="auto", cmap="RdBu_r")
        ax[0, 2 + k].set_title(f"T_ref - T_pred {S.KVPS[k]} kVp (views x bins)")
    ax[1, 3].imshow(tt[0], aspect="auto")
    ax[1, 3].set_title("T_ref 50 kVp")
    ax[2, 3].imshow(tp[0], aspect="auto")
    ax[2, 3].set_title("T_pred 50 kVp")
    ax[3, 3].axis("off")
    fig.savefig(run / "figures.png", dpi=80, bbox_inches="tight")
    fig2, a2 = plt.subplots(1, 3, figsize=(15, 3.5))
    a2[0].semilogy(hist["step"], hist["train_loss"])
    a2[0].set_title("train loss")
    a2[1].semilogy(hist["step"], hist["grad_norm"])
    a2[1].set_title("grad norm")
    a2[2].semilogy(hist["val_step"], hist["val_proj"], label="val projection")
    a2[2].semilogy(hist["val_step"], hist["val_label_mse"], label="val label MSE")
    a2[2].legend()
    fig2.savefig(run / "curves.png", dpi=80, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
