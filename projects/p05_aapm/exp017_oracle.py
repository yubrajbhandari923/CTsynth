"""EXP-017: ground-truth fractions -> fan-beam polychromatic forward vs the official transmissions.

Tolerances are the pre-registered first-order float32 error propagation (EXP-017 success
criteria), computed from geometry/tables/P only, never from the observed errors.
Usage: exp017_oracle.py --data <dataset dir with files/> [--n-other 40]
"""

import argparse
import json
import math
import os
import resource
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.aapm import schema as S
from ctsynth.aapm.fanbeam import fanbeam_project
from ctsynth.aapm.forward import AapmFanbeamForward
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.synthetic import line_integrals

U24 = 2.0**-24


def tolerance(
    tables, paths: np.ndarray, kvp: int, t: np.ndarray, n_layers: int = 512
) -> np.ndarray:
    """tol_T per ray. paths (3,V,D) float64 [cm] for this kVp's views; t = reference T (V,D)."""
    tb = tables[kvp]
    att = np.exp(-np.einsum("me,mvd->evd", tb.mu_per_cm, paths))  # (E,V,D)
    w = 0.5 * tb.spectrum[:, None, None]
    dp = math.sqrt(2 * n_layers) * U24 * paths  # (3,V,D)
    sens = np.einsum("me,evd->mvd", tb.mu_per_cm, w * att)  # S_m
    return 4 * (sens * dp).sum(0) + math.sqrt(len(tb.spectrum)) * 2 * U24 * t


def chord_vs_layered(phantom: np.ndarray, n_red: int) -> dict[str, float]:
    """Max/RMS difference of line integrals (cm): exact chord vs kit-layered, reduced geometry."""
    f = phantom.shape[-1] // n_red
    ph = phantom.reshape(3, n_red, f, n_red, f).mean((2, 4))
    g = S.GeometrySpec(n_pix=n_red, fov_cm=18.0, n_views=16, n_det=n_red * 2)
    ex = line_integrals(ph[None].astype(np.float64), g)[0]
    ly = fanbeam_project(torch.tensor(ph, dtype=torch.float64), g).numpy()
    d = ly - ex
    return {
        f"chord_vs_layered_n{n_red}_max_cm": float(np.abs(d).max()),
        f"chord_vs_layered_n{n_red}_rms_cm": float(np.sqrt((d**2).mean())),
        f"chord_vs_layered_n{n_red}_rel_rms": float(np.sqrt((d**2).mean() / (ex**2).mean())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n-other", type=int, default=40)
    a = ap.parse_args()
    run = Path(os.environ.get("RESEARCH_RUN_DIR", "."))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab_dir = Path(__file__).resolve().parents[2] / "data/aapm_kit_sample"
    tables = {k: S.load_model_table(tab_dir / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    store = CaseStore(a.data / "files")
    n = len(store)

    # kit cases located by content hash (not assumed to be the first ten)
    import hashlib

    with tarfile.open(a.data / "archives/starting_kit.tgz") as tf:
        m = next(x for x in tf.getmembers() if Path(x.name).name == "lowkVpImages.npy")
        import io

        kit_img = np.load(io.BytesIO(tf.extractfile(m).read()))  # type: ignore[union-attr]
        sino = {}
        for mat in ("Adipose", "Fibroglandular", "Calcification"):
            mm = next(x for x in tf.getmembers() if Path(x.name).name == f"Sinogram_{mat}.npy")
            sino[mat] = np.load(io.BytesIO(tf.extractfile(mm).read()))
    h = {
        hashlib.sha1(np.ascontiguousarray(store.arrays["lowkVpImages"][i]).tobytes()).hexdigest(): i
        for i in range(n)
    }
    kit_idx = [
        h[hashlib.sha1(np.ascontiguousarray(kit_img[j]).tobytes()).hexdigest()]
        for j in range(len(kit_img))
    ]
    others = [
        i for i in np.linspace(0, n - 1, a.n_other + len(kit_idx)).astype(int) if i not in kit_idx
    ][: a.n_other]
    cases = [(i, j) for j, i in enumerate(kit_idx)] + [(int(i), None) for i in others]

    fwd = AapmFanbeamForward(tables, dtype=torch.float32).to(dev)
    rows, viol_total, worst = [], 0, 0.0
    ang = {k: np.zeros(256) for k in S.KVPS}
    det = {k: np.zeros(1024) for k in S.KVPS}
    resid_keep: dict[str, np.ndarray] = {}
    t0 = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i, j in cases:
        c = store[i]
        x = torch.tensor(c.phantom[None], device=dev)
        with torch.no_grad():
            pred = fwd(x)
        for k in S.KVPS:
            ref = c.transmission[k].astype(np.float64)
            p = pred[k][0].double().cpu().numpy()
            err = p - ref
            paths = fanbeam_project(
                x[0].double(), views=torch.arange(0 if k == 80 else 1, 512, 2, device=dev)
            )
            pk = paths.cpu().numpy()
            tol = tolerance(tables, pk, k, ref)
            ratio = np.abs(err) / tol
            tissue = pk.sum(0) > 0
            rows.append(
                {
                    "case": i,
                    "kit_index": j,
                    "kvp": k,
                    "rmse": float(np.sqrt((err**2).mean())),
                    "mae": float(np.abs(err).mean()),
                    "max_abs": float(np.abs(err).max()),
                    "logT_rmse": float(np.sqrt(((np.log(p) - np.log(ref)) ** 2).mean())),
                    "rel_max_T_gt_1e-3": float((np.abs(err) / ref)[ref > 1e-3].max()),
                    "max_ratio_to_tol": float(ratio.max()),
                    "violations": int((ratio > 1).sum()),
                    "finite": bool(np.isfinite(p).all()),
                    "T_in_0_1": bool((p > 0).all() and (p <= 1 + 1e-6).all()),
                    "rmse_tissue_rays": float(np.sqrt((err[tissue] ** 2).mean())),
                    "rmse_air_rays": float(np.sqrt((err[~tissue] ** 2).mean()))
                    if (~tissue).any()
                    else None,
                }
            )
            viol_total += rows[-1]["violations"]
            worst = max(worst, rows[-1]["max_ratio_to_tol"])
            ang[k] += np.abs(err).mean(1)
            det[k] += np.abs(err).mean(0)
            if j == 3:
                resid_keep[f"resid_{k}"] = err.astype(np.float32)
        if j is not None:  # projector-only check against the kit sinograms (full interlaced views)
            lines = fwd.line_integrals(x, 80)[0].cpu().numpy()
            for mi, mat in enumerate(("Adipose", "Fibroglandular", "Calcification")):
                ref_p = sino[mat][j][::2]
                tolp = 4 * math.sqrt(1024) * U24 * float(ref_p.max())
                rows.append(
                    {
                        "case": i,
                        "kit_index": j,
                        "check": f"sinogram_{mat}_kvp80",
                        "max_abs_cm": float(np.abs(lines[mi] - ref_p).max()),
                        "tol_cm": tolp,
                    }
                )
    runtime = time.time() - t0
    peak = (
        torch.cuda.max_memory_allocated() / 2**20
        if dev.type == "cuda"
        else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )

    # float32 vs float64 on 3 cases (precision contribution)
    c32, c64 = (
        AapmFanbeamForward(tables, dtype=torch.float32).to(dev),
        AapmFanbeamForward(tables, dtype=torch.float64).to(dev),
    )
    dprec = 0.0
    for i, _ in cases[:3]:
        x = torch.tensor(store[i].phantom[None], device=dev)
        with torch.no_grad():
            a32, a64 = c32(x), c64(x.double())
        dprec = max(dprec, max(float((a32[k].double() - a64[k]).abs().max()) for k in S.KVPS))

    ph = store[kit_idx[3]].phantom.astype(np.float64)
    disc = {**chord_vs_layered(ph, 128), **chord_vs_layered(ph, 64)}
    sino_fail = [r for r in rows if "check" in r and r["max_abs_cm"] > r["tol_cm"]]
    summary = {
        "n_cases": len(cases),
        "n_kit_cases": len(kit_idx),
        "device": str(dev),
        "total_violating_rays": viol_total,
        "max_ratio_to_tol": worst,
        "max_abs_T_error": max(r["max_abs"] for r in rows if "max_abs" in r),
        "max_rmse_T": max(r["rmse"] for r in rows if "rmse" in r),
        "sinogram_check_failures": len(sino_fail),
        "max_sinogram_abs_cm": max(r["max_abs_cm"] for r in rows if "check" in r),
        "float32_vs_float64_max_abs_T": dprec,
        "runtime_s": runtime,
        "peak_mem_mib": peak,
        "all_finite": all(r["finite"] for r in rows if "finite" in r),
        "all_T_in_unit_interval": all(r["T_in_0_1"] for r in rows if "T_in_0_1" in r),
        **disc,
    }
    summary["pass"] = bool(
        viol_total == 0
        and not sino_fail
        and summary["all_finite"]
        and summary["all_T_in_unit_interval"]
    )
    (run / "exp017_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    (run / "exp017_rows.json").write_text(json.dumps(rows))
    np.savez_compressed(
        run / "exp017_profiles.npz",
        **{f"angle_{k}": ang[k] / len(cases) for k in S.KVPS},
        **{f"det_{k}": det[k] / len(cases) for k in S.KVPS},
        **resid_keep,
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
    for q, k in enumerate(S.KVPS):
        if f"resid_{k}" in resid_keep:
            im = ax[q].imshow(resid_keep[f"resid_{k}"], aspect="auto", cmap="RdBu_r")
            ax[q].set_title(f"kit case 3, {k} kVp: T_pred - T_ref")
            fig.colorbar(im, ax=ax[q])
    for k in S.KVPS:
        ax[2].semilogy(det[k] / len(cases), label=f"{k} kVp")
    ax[2].set_title("mean |err| vs detector bin")
    ax[2].legend()
    fig.savefig(run / "exp017_residuals.png", dpi=110, bbox_inches="tight")
    with open(run / "artifacts.jsonl", "a") as f:
        for nme in (
            "exp017_summary.json",
            "exp017_rows.json",
            "exp017_profiles.npz",
            "exp017_residuals.png",
        ):
            f.write(json.dumps({"path": str((run / nme).resolve()), "name": nme}) + "\n")
    with open(run / "metrics.jsonl", "a") as f:
        for key, v in summary.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                f.write(json.dumps({"name": key, "value": v}) + "\n")
        f.write(json.dumps({"name": "pass", "value": float(summary["pass"])}) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
