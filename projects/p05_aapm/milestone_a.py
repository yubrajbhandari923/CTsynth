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
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.metrics import partition_errors, per_material_errors, s1_mean_rmse
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


def feasibility(frac: torch.Tensor) -> torch.Tensor:
    return torch.relu(-frac).mean() + torch.relu(frac.sum(1) - 1).mean()


@torch.no_grad()
def evaluate(model, fwd, img, tr, tru, bs=2):
    model.eval()
    preds, pe = [], []
    for s in range(0, img.shape[0], bs):
        f = model(img[s : s + bs])
        preds.append(f.cpu())
        pe.append(((stack_kvp(fwd(f)) - tr[s : s + bs]) ** 2).mean(dim=(1, 2, 3)).cpu())
    p, t = torch.cat(preds).numpy().astype(np.float64), tru.cpu().numpy().astype(np.float64)
    e = p - t
    out = {
        "s1": s1_mean_rmse(t, p),
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
    ap.add_argument("--arm", choices=["supervised", "physics", "physics_feas"], required=True)
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--views", type=int, default=32, help="view positions per step (stochastic)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-feas", type=float, default=1.0)
    ap.add_argument("--output", default="softplus", choices=["linear", "softplus"])
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
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
    fwd = AapmFanbeamForward(tables).to(dev)
    model = UNet(depth=a.depth, width=a.width, output=a.output).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
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
        frac = model(img_tr[ix])
        if a.arm == "supervised":
            loss = ((frac - tru_tr[ix]) ** 2).mean()
        else:
            sub = torch.randperm(256, generator=gen)[: a.views].to(dev)
            loss = projection_loss(fwd, frac, T_tr[ix], sub)
            if a.arm == "physics_feas":
                loss = loss + a.w_feas * feasibility(frac)
        opt.zero_grad()
        loss.backward()
        gn = float(
            torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None))
        )
        if not (np.isfinite(gn) and np.isfinite(loss.item())):
            raise RuntimeError(f"non-finite loss/grad at step {step}")
        opt.step()
        hist["step"].append(step)
        hist["train_loss"].append(float(loss))
        hist["grad_norm"].append(gn)
        if step % a.val_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                fv = torch.cat([model(img_va[s : s + 2]) for s in range(0, img_va.shape[0], 2)])
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
    res, pred = evaluate(model, fwd, img_va, T_va, tru_va)
    res.update(
        arm=a.arm,
        steps=a.steps,
        selection="val label MSE" if a.arm == "supervised" else "val projection loss",
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
        run / "val_pred.npz", pred=pred.astype(np.float32), truth=tru_va.cpu().numpy()
    )
    figures(run, model, fwd, img_va, T_va, tru_va, hist)
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
        for k in ("s1", "proj_rmse_T", "null_dir_error_frac", "runtime_s", "peak_mem_mib"):
            f.write(json.dumps({"name": k, "value": res[k]}) + "\n")
        for m, d in res["per_material"].items():
            f.write(json.dumps({"name": f"rmse_{m}", "value": d["rmse"]}) + "\n")
        f.write(json.dumps({"name": "max_grad_norm", "value": max(hist["grad_norm"])}) + "\n")
    print(json.dumps({k: res[k] for k in ("s1", "proj_rmse_T", "null_dir_error_frac")}, indent=1))
    return 0


def figures(run, model, fwd, img, T, tru, hist):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        f = model(img[:1])
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
