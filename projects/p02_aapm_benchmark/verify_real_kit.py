"""Retrospective verification of the real-kit consolidation findings (F-025, F-027, F-030, F-031).

DETERMINISTIC: no training, no randomness. Recomputes the classical-estimator ranking on real
AAPM starting-kit cases and the s1-vs-official-formula check, so those findings bind to evidence.

The kit sample cases are NOT in the repo (only the two small model tables are). Pass the extracted
starting_kit directory with --kit; the script refuses to invent numbers if it is absent.

  uv run python projects/p02_aapm_benchmark/verify_real_kit.py --kit <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ctsynth.aapm import classical as C
from ctsynth.aapm import metrics as M
from ctsynth.aapm import schema as S

TABLE_DIR = Path("data/aapm_kit_sample")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, required=True, help="extracted starting_kit directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-cases", type=int, default=3)
    a = ap.parse_args()
    if not (a.kit / "Phantom_Adipose.npy").exists():
        raise SystemExit(f"no kit data at {a.kit}; refusing to produce numbers without it")

    tables = {k: S.load_model_table(TABLE_DIR / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    mu = C.effective_mu(tables)
    n = a.n_cases

    def load(name: str) -> np.ndarray:
        return np.load(a.kit / f"{name}.npy", mmap_mode="r")[:n].astype(np.float64)

    truth = np.stack(
        [load("Phantom_Adipose"), load("Phantom_Fibroglandular"), load("Phantom_Calcification")],
        axis=1,
    )
    lo, hi = load("lowkVpImages"), load("highkVpImages")
    tissue = np.abs(truth.sum(axis=1) - 1.0) <= 1e-3

    def rmse(p: np.ndarray) -> float:
        return float(np.sqrt(((p - truth) ** 2).mean()))

    modes = ("unconstrained", "sum_to_one", "simplex")
    est = {m: C.decompose_lstsq(lo, hi, mu, mode=m) for m in modes}
    preds: dict[str, np.ndarray] = {
        "zero_prediction": np.zeros_like(truth),
        "unconstrained": est["unconstrained"],
        "unconstrained_clipped": np.clip(est["unconstrained"], 0.0, 1.0),
        "sum_to_one": est["sum_to_one"],
        "sum_to_one_clipped": np.clip(est["sum_to_one"], 0.0, 1.0),
        "simplex": est["simplex"],
    }

    # how far are the real images from the linear effective-mu model? (F-025)
    pl = (mu[0][:, None, None] * truth).sum(axis=1)
    ph = (mu[1][:, None, None] * truth).sum(axis=1)
    m = truth.sum(axis=1) > 0.5
    out: dict[str, object] = {
        "kind": "retrospective_verification",
        "deterministic": True,
        "n_cases": n,
        "kit_files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(TABLE_DIR.glob("*.npy"))
        },
        "linear_model_rel_residual_lo": float(np.abs(pl - lo)[m].mean() / np.abs(lo[m]).mean()),
        "linear_model_rel_residual_hi": float(np.abs(ph - hi)[m].mean() / np.abs(hi[m]).mean()),
        "sum_to_one_frac_outside_unit_range": float(
            np.mean((est["sum_to_one"] < 0) | (est["sum_to_one"] > 1))
        ),
        "sum_to_one_max_abs": float(np.abs(est["sum_to_one"]).max()),
        "rmse": {k: rmse(v) for k, v in preds.items()},
        "tissue_rmse": {
            k: float(np.sqrt((((v - truth).transpose(1, 0, 2, 3)[:, tissue]) ** 2).mean()))
            for k, v in preds.items()
        },
        "s1": {k: float(M.s1_mean_rmse(truth, v)) for k, v in preds.items()},
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "real_kit_verification.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    for k in preds:
        print(f"{k:24s} rmse {out['rmse'][k]:8.4f}  tissue {out['tissue_rmse'][k]:8.4f}")  # type: ignore[index]
    print(
        f"linear-model residual lo {out['linear_model_rel_residual_lo']:.4f} "
        f"hi {out['linear_model_rel_residual_hi']:.4f}; "
        f"sum_to_one outside [0,1]: {out['sum_to_one_frac_outside_unit_range']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
