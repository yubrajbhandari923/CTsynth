"""AAPM benchmark evaluation harness: metrics record, run manifest, runtime/memory, failure cases.

Built around the frozen metric functions in ctsynth.aapm.metrics. The one thing re-derived here
is the worst-ROI *location* (s2 returns only the value); it is cross-checked against s2.

Usage (no trained model needed):
  uv run python projects/p02_aapm_benchmark/evaluate.py configs/<cfg>.yaml \
      --data-dir DIR --out OUT [--split val] [--purpose development] [--methods zero_prediction ...]

Writes <OUT>/metrics.json (deterministic: same config + data + seed -> byte-identical),
<OUT>/run_meta.json (ctsynth.runmeta), <OUT>/manifest.json (sha256 of every input file),
<OUT>/timing.json (wall time and tracemalloc peak; NOT in metrics.json: it varies run to run),
and <OUT>/worst_case_<method>.npz (truth and prediction of the worst case by s1, for figures).

Config params understood (all optional): n_pix, n_views, n_det (geometry; defaults = real AAPM
sizes, checked against the file headers by CaseStore), roi (default 25), k_worst (default 3).

Memory: the whole requested split is materialised in RAM for the metric arrays. The loader itself is
streaming; this harness is not.

tracemalloc peak: counts Python-level allocations, including numpy array buffers (numpy reports
them to tracemalloc). It does NOT count torch tensors or other native memory outside that tracking.
Wall time is measured in an untraced run; tracemalloc overhead is excluded from it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ctsynth import runmeta
from ctsynth.aapm import classical as C
from ctsynth.aapm import metrics as M
from ctsynth.aapm import schema as S
from ctsynth.aapm.loader import CaseStore
from ctsynth.aapm.splits import split_indices
from ctsynth.config import load_config
from ctsynth.hashing import sha256_file

RECORD_VERSION = 1
BASE_LIMITATIONS: tuple[str, ...] = (
    "s1 and s2 are the official kit metrics on the stated split; s2 omits the last ROI position "
    "(kit loop) unless s2_full_scan is used.",
    "Map units are dimensionless volume fractions (inferred, not stated by the kit).",
    "Regimes come from the truth sum (air: sum <= 1e-3; tissue: |sum-1| <= 1e-3; band: otherwise); "
    "partition errors use the prediction's material sum.",
    "Failure cases come from the evaluated split; they are for diagnosis, not model selection.",
)

Array = S.Array


@dataclass(frozen=True)
class Inputs:
    """What a method may read: per-kVp images and transmission, stacked over the split's cases."""

    images: dict[int, Array]  # {50: (N, n, n), 80: (N, n, n)} float32, image[ix, iy]
    transmission: dict[int, Array]  # {50: (N, V//2, D), 80: ...} float32
    geometry: S.GeometrySpec
    tables: dict[int, S.ModelTable]  # committed model_data_{kVp}kVp.npy; source of the mu matrix


def _zero_prediction(inp: Inputs) -> Array:
    """Null reference: predicts no material anywhere. Sets the floor any method must beat."""
    n = inp.geometry.n_pix
    return np.zeros((inp.images[S.KVP_HIGH].shape[0], 3, n, n), dtype=np.float64)


def _classical(mode: C.Mode) -> Callable[[Inputs], Array]:
    """Per-pixel image-domain inversion (F-014); `mode` picks the constraint set."""

    def run(inp: Inputs) -> Array:
        mu = C.effective_mu(inp.tables)
        return C.decompose_lstsq(
            inp.images[S.KVP_LOW], inp.images[S.KVP_HIGH], mu, mode=mode
        ).astype(np.float64)

    return run


def _classical_tv(inp: Inputs) -> Array:
    """TV-regularised variant. lam is a FIXED default here; it must be selected on the
    validation split only (never on test) — see docs/aapm/classical-baseline.md."""
    mu = C.effective_mu(inp.tables)
    return C.decompose_tv(
        inp.images[S.KVP_LOW], inp.images[S.KVP_HIGH], mu, lam=TV_LAM, n_iter=TV_ITERS
    ).astype(np.float64)


TV_LAM = 1e-4  # validation-split choice on SYNTHETIC data only; re-select for real data
TV_ITERS = 300

# Registry of methods. A method maps Inputs -> prediction (N, 3, n, n) in canonical MATERIALS order.
METHODS: dict[str, Callable[[Inputs], Array]] = {
    "zero_prediction": _zero_prediction,
    "classical_unconstrained": _classical("unconstrained"),
    "classical_sum_to_one": _classical("sum_to_one"),
    "classical_simplex": _classical("simplex"),
    "classical_tv": _classical_tv,
}


def _as_stack(a: Any) -> Array:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError(f"expected (C, 3, H, W) with C >= 1, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values; refusing to report metrics for them")
    return arr


def _worst_roi(t: Array, r: Array, roi: int) -> dict[str, Any] | None:
    """Location and value of the ROI that attains s2 (same window set and skip rule as
    metrics.s2_worst_roi_rmse, default kit loop). None if every ROI is skipped or none fits."""
    h, w = t.shape[2], t.shape[3]
    nx, ny = h - roi, w - roi
    if nx <= 0 or ny <= 0:
        return None
    sq_sum = sliding_window_view((t - r) ** 2, (roi, roi), axis=(2, 3))[:, :, :nx, :ny].sum(
        axis=(1, 4, 5)
    )
    tmax = sliding_window_view(t.max(axis=1), (roi, roi), axis=(1, 2))[:, :nx, :ny].max(axis=(3, 4))
    keep = tmax > M.ROI_TRUTH_MAX_MIN
    if not keep.any():
        return None
    vals = np.where(keep, np.sqrt(sq_sum / (3 * roi * roi)), -np.inf)
    c, x, y = (int(v) for v in np.unravel_index(int(np.argmax(vals)), vals.shape))
    return {"case": c, "x0": x, "y0": y, "roi": roi, "value": float(vals[c, x, y])}


def _clean(o: Any) -> Any:
    """Plain Python for JSON: numpy scalars -> float/int; NaN/inf -> None (written as null)."""
    if isinstance(o, bool | np.bool_):
        return bool(o)
    if isinstance(o, float | np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, int | np.integer):
        return int(o)
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    return o


def evaluate_predictions(
    truth: Any,
    pred: Any,
    *,
    split_name: str,
    method_name: str,
    extra: dict[str, Any] | None = None,
    roi: int = M.ROI_SIZE,
    k_worst: int = 3,
) -> dict[str, Any]:
    """Machine-readable metrics record for one method on one split. Inputs (C, 3, H, W) in canonical
    material order (a (3, H, W) array is one case). `extra` keys are merged in; its 'limitations'
    list extends the base list. Counts are ints, other numbers floats, undefined values are null.
    """
    t, r = _as_stack(truth), _as_stack(pred)
    if t.shape != r.shape:
        raise ValueError(f"truth {t.shape} != pred {r.shape}")
    c = t.shape[0]
    part = M.partition_errors(t, r, S.PARTITION_TOL)
    per_case = [M.s1_mean_rmse(t[i : i + 1], r[i : i + 1]) for i in range(c)]
    order = sorted(range(c), key=lambda i: (-per_case[i], i))[: min(k_worst, c)]

    s2 = M.s2_worst_roi_rmse(t, r, roi=roi)
    worst_roi = _worst_roi(t, r, roi)
    if abs((worst_roi["value"] if worst_roi else 0.0) - s2) > 1e-12:
        raise RuntimeError("worst-ROI recorder disagrees with metrics.s2_worst_roi_rmse")

    rec: dict[str, Any] = {
        "record_version": RECORD_VERSION,
        "method_name": method_name,
        "split_name": split_name,
        "n_cases": c,
        "shapes": {"truth": list(t.shape), "pred": list(r.shape)},
        "s1": M.s1_mean_rmse(t, r),
        "s2": s2,
        "s2_full_scan": M.s2_worst_roi_rmse(t, r, roi=roi, include_last_position=True),
        "aggregate_rmse": M.aggregate_rmse(t, r),
        "per_material": M.per_material_errors(t, r),
        "partition": {
            "tissue": {"mean_abs": part["tissue_mean_abs"], "max_abs": part["tissue_max_abs"]},
            "air": {"mean_abs": part["air_mean_abs"], "max_abs": part["air_max_abs"]},
        },
        "regime_pixel_counts": {
            "tissue": int(part["n_tissue"]),
            "air": int(part["n_air"]),
            "band": int(part["n_band"]),
        },
        "boundary_rmse": M.boundary_rmse(t, r, S.PARTITION_TOL),
        "failure_cases": {
            "worst_cases_by_s1": [{"case": i, "s1": per_case[i]} for i in order],
            "worst_roi_s2": worst_roi,
        },
        "limitations": list(BASE_LIMITATIONS),
    }
    if extra:
        extra = dict(extra)
        rec["limitations"] += list(extra.pop("limitations", []))
        clash = set(extra) & set(rec)
        if clash:
            raise ValueError(f"extra would overwrite record keys {sorted(clash)}")
        rec.update(extra)
    return _clean(rec)


def _dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(_clean(obj), indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate registered AAPM methods on one split.")
    ap.add_argument("config", type=Path)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument(
        "--purpose",
        default="development",
        help="must be 'final_evaluation' for --split test (guard in ctsynth.aapm.splits)",
    )
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    a = ap.parse_args(argv)
    unknown = [m for m in a.methods if m not in METHODS]
    if unknown:
        ap.error(f"unknown methods {unknown}; registered: {sorted(METHODS)}")

    cfg = load_config(a.config)
    p = cfg.params
    geometry = S.GeometrySpec(
        n_pix=int(p.get("n_pix", S.REAL_GEOMETRY.n_pix)),
        fov_cm=S.REAL_GEOMETRY.fov_cm,
        n_views=int(p.get("n_views", S.REAL_GEOMETRY.n_views)),
        n_det=int(p.get("n_det", S.REAL_GEOMETRY.n_det)),
    )
    roi, k_worst = int(p.get("roi", M.ROI_SIZE)), int(p.get("k_worst", 3))

    table_dir = Path(p.get("table_dir", "data/aapm_kit_sample"))
    tables = {k: S.load_model_table(table_dir / f"model_data_{k}kVp.npy", k) for k in S.KVPS}

    store = CaseStore(a.data_dir, geometry)
    fr = (float(p.get("f_train", 0.8)), float(p.get("f_val", 0.1)), float(p.get("f_test", 0.1)))
    idx = split_indices(len(store), cfg.seed, a.split, purpose=a.purpose, fractions=fr)
    if idx.size == 0:
        raise SystemExit(f"split {a.split!r} is empty for N={len(store)}, seed={cfg.seed}")
    cases = list(store.iter_cases(idx))
    truth = np.stack([c.phantom for c in cases])
    inp = Inputs(
        images={k: np.stack([c.image[k] for c in cases]) for k in S.KVPS},
        transmission={k: np.stack([c.transmission[k] for c in cases]) for k in S.KVPS},
        geometry=geometry,
        tables=tables,
    )
    del cases

    input_files = [store.root / f"{stem}.npy" for stem in sorted(store.arrays)] + [a.config]
    manifest = {
        "inputs": [
            {"path": str(f.resolve()), "bytes": f.stat().st_size, "sha256": sha256_file(f)}
            for f in input_files
        ]
    }
    extra_lim = (
        f"Data directory {a.data_dir} is used as given; this harness does not check whether it "
        "holds AAPM or synthetic data."
    )

    a.out.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    timing: dict[str, Any] = {}
    for name in a.methods:
        fn = METHODS[name]
        t0 = time.perf_counter()
        pred = fn(inp)
        wall = time.perf_counter() - t0
        tracemalloc.start()
        try:
            pred_traced = fn(inp)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if not np.array_equal(pred, pred_traced):
            raise RuntimeError(f"method {name!r} is not deterministic for identical inputs")
        records[name] = evaluate_predictions(
            truth,
            pred,
            split_name=a.split,
            method_name=name,
            extra={"limitations": [extra_lim]},
            roi=roi,
            k_worst=k_worst,
        )
        timing[name] = {"wall_s": wall, "tracemalloc_peak_bytes": int(peak)}
        worst = records[name]["failure_cases"]["worst_cases_by_s1"][0]["case"]
        np.savez(
            a.out / f"worst_case_{name}.npz",
            case=worst,
            truth=_as_stack(truth)[worst],
            pred=_as_stack(pred)[worst],
        )

    metrics = {
        "experiment_id": cfg.experiment_id,
        "seed": cfg.seed,
        "split_name": a.split,
        "n_cases": int(idx.size),
        "case_indices": idx.tolist(),
        "methods": records,
        "not_computed": {
            m: "not requested via --methods" for m in sorted(METHODS) if m not in a.methods
        },
    }
    _dump(metrics, a.out / "metrics.json")
    _dump(timing, a.out / "timing.json")
    _dump(manifest, a.out / "manifest.json")
    runmeta.write(cfg, a.out)


if __name__ == "__main__":
    main()
