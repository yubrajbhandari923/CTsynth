"""Stage 1 (EXP-036 pilot): converged per-case solves of the polychromatic projection problem.

No labels in any loss. Truth enters only (a) inits labelled ORACLE (truth + perturbation), (b) the
oracle occupancy S(x) used by arms A/B/D (the 'sum known' condition of F-043), (c) evaluation.
Arms: A  S*(a, 1-a, 0)          unknown a(x)        [solver positive control; floor from lesions]
      B  S*(a, 1-a-c, c)         unknown a(x), c(x); + L1 on c
      C  fractions (3 free) inside the image-derived support, no sum/calcification constraint
      D  arm A + bimodality prior;  A_img: arm A with image-derived support, sum=1 (no oracle S)
Optimiser: chunked L-BFGS on the FULL-view mean-square projection error (256 views x 2 kVp).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from stage0_profile import exact_null, smooth_field

from ctsynth import seeding
from ctsynth.aapm import priors as P
from ctsynth.aapm import schema as S
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.metrics import s1_mean_rmse
from ctsynth.aapm.physics_loss import stack_kvp
from ctsynth.aapm.solve import drop_zero_weight, error_bands, minimize, to_fractions

SC = 1e6  # loss scale: keeps L-BFGS curvature pairs away from float32 underflow
NCH = 8  # view chunks per evaluation (32 views each): bounds memory like training
AMP = 0.3  # oracle perturbation amplitude (adipose-fraction units)
LAM_CALC, LAM_BIMODAL = 5.6e-5, 1e-5  # inside the Stage-0 windows (RUN-0093), fixed, not swept

# (arm, init) in priority order; see EXP-036 amendment 2
RUNS = [
    ("A", "uniform"),
    ("A", "rough"),
    ("A", "image"),
    ("A", "smooth"),
    ("A", "uswap"),
    ("B", "uniform"),
    ("B", "rough"),
    ("B", "null"),
    ("C", "uniform"),
    ("C", "rough"),
    ("C", "null"),
    ("A_img", "uniform"),
    ("D", "uniform"),
]


class FullMSE(torch.autograd.Function):
    """Full-view loss*SC with gradient accumulated chunk by chunk (graph freed per chunk)."""

    @staticmethod
    def forward(ctx, x, build, fwd, T, extra):  # type: ignore[no-untyped-def]
        views = torch.arange(T.shape[2], device=T.device)
        total = 0.0
        with torch.enable_grad():
            xd = x.detach().requires_grad_(True)
            for c in range(NCH):
                sub = views[c::NCH]
                pred = stack_kvp(fwd(build(xd), sub))
                loss = ((pred - T[:, :, sub]) ** 2).sum() / T.numel()
                loss.backward()
                total += float(loss.detach())
                if not np.isfinite(total):  # EXP-036 amendment 3: name the first non-finite call
                    f = build(xd).detach()
                    raise FloatingPointError(
                        f"non-finite loss at chunk {c}: x[{x.min():.3g},{x.max():.3g}] "
                        f"frac[{f.min():.3g},{f.max():.3g}] pred_nonfinite="
                        f"{int((~torch.isfinite(pred)).sum())}/{pred.numel()} "
                        f"grad_nonfinite={int((~torch.isfinite(xd.grad)).sum())}"
                    )
            if extra is not None:
                pen = extra(build(xd))
                pen.backward()
                total += float(pen.detach())
        ctx.g = xd.grad.detach()
        return x.new_tensor(total * SC)

    @staticmethod
    def backward(ctx, go):  # type: ignore[no-untyped-def]
        return ctx.g * SC * go, None, None, None, None


def data_mse(fwd, frac, T) -> float:
    views = torch.arange(T.shape[2], device=T.device)
    tot = 0.0
    with torch.no_grad():
        for c in range(NCH):
            sub = views[c::NCH]
            tot += float(((stack_kvp(fwd(frac, sub)) - T[:, :, sub]) ** 2).sum())
    return tot / T.numel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--rank", choices=["low", "median", "high"], required=True)
    ap.add_argument("--max-iter", type=int, default=1500)
    ap.add_argument("--runs", nargs="*", help="ARM:INIT filters, e.g. A:uniform")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    # amendment 3: drop zero-weight energies (exact; stops 0*inf=NaN on negative paths)
    tables = {
        k: drop_zero_weight(S.load_model_table(tab / f"model_data_{k}kVp.npy", k)) for k in S.KVPS
    }
    store = CaseStore(a.data / "files")
    val = json.loads(a.split.read_text())["val"][:100]
    counts = sorted((int((store[i].phantom[2] > 1e-3).sum()), i) for i in val)
    n_calc, ci = counts[{"low": 0, "median": 50, "high": 99}[a.rank]]
    print(f"rank {a.rank}: case {ci} with {n_calc} calcified pixels", flush=True)
    c = store[ci]
    tru = torch.tensor(c.phantom[None], dtype=torch.float32, device=dev)
    T = torch.tensor(np.stack([c.transmission[50], c.transmission[80]])[None], device=dev)
    img = torch.tensor(np.stack([c.image[50], c.image[80]])[None], dtype=torch.float32, device=dev)
    Ssum = tru.sum(1).clamp(0, 1)  # oracle occupancy
    sup = Ssum > 1e-3
    m_img = (img >= 0.1).any(1)  # image-derived support (no labels)
    tis = (tru.sum(1) - 1).abs() <= 1e-3
    fwd = AapmFanbeamForward(tables, exp_clip=None).to(dev)
    nx = exact_null(tables)
    sw = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
    M = np.array(
        [
            [(0.5 * tables[k].spectrum * tables[k].mu_per_cm[m]).sum() for m in range(3)]
            for k in S.KVPS
        ]
    )
    g = seeding.rng(0, 36, {"low": 0, "median": 1, "high": 2}[a.rank])
    smooth = torch.tensor(smooth_field(tru.shape[2:], 20.0, g), dtype=torch.float32, device=dev)
    rough = torch.tensor(np.sign(g.standard_normal(tru.shape[2:])), dtype=torch.float32, device=dev)
    a_true = (tru[:, 0] / Ssum.clamp_min(1e-8)) * sup  # adipose share of occupancy
    c_true = (tru[:, 2] / Ssum.clamp_min(1e-8)) * sup
    pat = {"uswap": torch.ones_like(smooth), "smooth": smooth, "rough": rough}
    vec = lambda v: torch.tensor(v, dtype=torch.float32, device=dev).view(1, 3, 1, 1)  # noqa: E731

    def setup(arm):
        """build(x)->fractions, mask used, whether occupancy S is oracle, penalty."""
        if arm in ("A", "D"):
            build = lambda x: to_fractions(x, sup, "A") * Ssum[:, None]  # noqa: E731
        elif arm == "B":
            build = lambda x: to_fractions(x, sup, "B") * Ssum[:, None]  # noqa: E731
        elif arm == "A_img":
            build = lambda x: to_fractions(x, m_img, "A")  # noqa: E731
        else:
            build = lambda x: to_fractions(x, m_img, "C")  # noqa: E731
        pen = None
        if arm == "B":
            pen = lambda f: LAM_CALC * P.calc_l1(f.double(), tis).float()  # noqa: E731
        if arm == "D":
            pen = lambda f: LAM_BIMODAL * P.bimodal(f.double(), tis).float()  # noqa: E731
        return build, pen

    def init(arm, kind):
        mu = M[1]
        S_ = Ssum if arm in ("A", "B", "D") else torch.ones_like(Ssum)
        a0 = {"uniform": 0.5 * torch.ones_like(Ssum)}.get(kind)
        if kind == "image":
            a0 = ((mu[1] - img[:, 1] / S_.clamp_min(1e-8)) / (mu[1] - mu[0])).clamp(0, 1)
        if kind in pat:
            a0 = a_true + AMP * pat[kind]
        if arm in ("A", "D", "A_img"):
            return a0[:, None]
        if arm == "B":
            if kind == "null":  # oracle: truth + AMP*n projected on (a, c); sum is fixed by S
                return torch.stack([a_true + AMP * nx[0], c_true + AMP * nx[2]], 1)
            oracle = kind in pat  # non-oracle inits start with no calcification
            return torch.stack([a0, c_true if oracle else torch.zeros_like(c_true)], 1)
        # arm C: free fractions inside the image support
        if kind == "null":
            f0 = tru + AMP * vec(nx)
        elif kind in pat:
            f0 = tru + AMP * pat[kind][None, None] * vec([1.0, -1.0, 0.0])
        else:
            f0 = torch.stack([a0, 1 - a0, torch.zeros_like(a0)], 1)
        return f0 * m_img[:, None]

    def evaluate(arm, build, x):
        with torch.no_grad():
            f = build(x)
            fc = f.clone()
            if arm != "C":
                fc[:, 0] = torch.minimum(fc[:, 0].clamp_min(0), Ssum)  # clip adipose into [0, S]
                fc[:, 1] = f.sum(1) - fc[:, 0] - fc[:, 2]
            e = (f - tru)[0]
            err_a = ((f[:, 0] / f.sum(1).clamp_min(1e-8)) - a_true)[:, tis[0]]
            full_err = torch.zeros_like(a_true)
            full_err[tis] = err_a
            bands = error_bands(full_err, tis)
            e_t = e[:, tis[0]].double().cpu().numpy()
            out = {
                "s1_unclipped": s1_mean_rmse(tru.cpu().numpy(), f.cpu().numpy()),
                "s1_clipped": s1_mean_rmse(tru.cpu().numpy(), fc.cpu().numpy()),
                "mse_data": data_mse(fwd, f, T),
                "band_rms_adipose": bands,
                "adipose_err_mean_tissue": float(err_a.mean()),
                "swap_coord_mean": float((sw @ e_t).mean()),
                "null_coord_mean": float((nx @ e_t).mean()),
                "tissue_abs_sum_err": float((f.sum(1) - 1)[tis].abs().mean()),
                "calc_mass_off_truth_support": float(
                    f[:, 2][tis & (tru[:, 2] < 1e-3)].abs().mean()
                ),
            }
        return out

    done = []
    dest = Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / f"stage1_{a.rank}.json"
    # irreducible misspecification of the calcification=0 arms (truth is not in their model)
    with torch.no_grad():
        fA = to_fractions(a_true[:, None], sup, "A") * Ssum[:, None]
        floor_A = data_mse(fwd, fA, T)
        floor_truth = data_mse(fwd, tru, T)
    meta = {
        "case": ci,
        "n_calc_pixels": n_calc,
        "rank": a.rank,
        "floor_mse_truth": floor_truth,
        "floor_mse_armA_model": floor_A,
        "max_iter": a.max_iter,
        "AMP": AMP,
        "LAM_CALC": LAM_CALC,
        "LAM_BIMODAL": LAM_BIMODAL,
    }
    print(meta, flush=True)
    for arm, kind in RUNS:
        if a.runs and f"{arm}:{kind}" not in a.runs:
            continue
        build, pen = setup(arm)
        x0 = init(arm, kind).contiguous()

        def _val(x, b, p):
            with torch.no_grad():
                f = b(x)
                v = data_mse(fwd, f, T) + (float(p(f)) if p is not None else 0.0)
            return x.new_tensor(v * SC)

        loss_fn = lambda x, b=build, p=pen: (  # noqa: E731
            FullMSE.apply(x, b, fwd, T, p) if torch.is_grad_enabled() else _val(x, b, p)
        )

        t0 = time.time()
        r0 = evaluate(arm, build, x0)
        x, info = minimize(loss_fn, x0, max_iter=a.max_iter, abs_tol=1e-12 * SC)
        r1 = evaluate(arm, build, x)
        rec = {
            "arm": arm,
            "init": kind,
            "init_metrics": r0,
            "final_metrics": r1,
            "n_iter": info["n_iter"],
            "reason": info["reason"],
            "trace": [(i, v / SC) for i, v in info["trace"]],
            "seconds": time.time() - t0,
            "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0,
        }
        done.append(rec)
        dest.write_text(json.dumps({"meta": meta, "runs": done}, indent=1, default=float))
        print(
            arm,
            kind,
            "iters",
            info["n_iter"],
            info["reason"],
            f"mse {r0['mse_data']:.2e}->{r1['mse_data']:.2e}",
            f"s1 {r0['s1_clipped']:.4f}->{r1['s1_clipped']:.4f}",
            f"hf {r1['band_rms_adipose']['high']:.3f}",
            f"{rec['seconds']:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
