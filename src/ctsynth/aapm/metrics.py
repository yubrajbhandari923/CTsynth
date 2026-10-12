"""Official AAPM metrics and diagnostic metrics. Arrays are (C, 3, H, W) float, canonical material
order, dimensionless volume fractions; a (3, H, W) input is treated as C == 1. All reductions are in
float64. No normalisation anywhere.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ctsynth.aapm.schema import MATERIALS, Array

ROI_SIZE = 25
ROI_TRUTH_MAX_MIN = 0.01  # ROIs with truth.max() <= this are skipped


def _as4d(a: Array) -> Array:
    arr = np.asarray(a, dtype=np.float64)
    return arr[None] if arr.ndim == 3 else arr


def s1_mean_rmse(truth: Array, pred: Array) -> float:
    """Official s1 = (1/C) sum_c sqrt( sum_{m,x,y} (t-r)^2 / (3*H*W) ). Units: volume fraction."""
    t, r = _as4d(truth), _as4d(pred)
    sq = ((t - r) ** 2).sum(axis=(1, 2, 3))
    return float(np.mean(np.sqrt(sq / (3 * t.shape[2] * t.shape[3]))))


def s2_worst_roi_rmse(
    truth: Array,
    pred: Array,
    *,
    roi: int = ROI_SIZE,
    include_last_position: bool = False,
) -> float:
    """Official tie-break s2 = max over cases and ROI positions of
    sqrt( sum_{m, roi x roi} (t-r)^2 / (3*roi^2) ), over stride-1 windows whose truth max
    (over all 3 materials and the window) is > ROI_TRUTH_MAX_MIN; others are skipped.
    Positions x0, y0 in range(H - roi) (kit quirk:
    the last position H - roi is omitted) unless include_last_position=True
    (range(H - roi + 1)). Returns 0.0
    if every ROI is skipped. Units: volume fraction.
    """
    t, r = _as4d(truth), _as4d(pred)
    h, w = t.shape[2], t.shape[3]
    if h < roi or w < roi:
        return 0.0
    extra = 1 if include_last_position else 0
    nx, ny = h - roi + extra, w - roi + extra
    if nx <= 0 or ny <= 0:
        return 0.0
    sq = sliding_window_view((t - r) ** 2, (roi, roi), axis=(2, 3))[:, :, :nx, :ny]
    sq_sum = sq.sum(axis=(1, 4, 5))  # (C, nx, ny)
    tmax = sliding_window_view(t.max(axis=1), (roi, roi), axis=(1, 2))[:, :nx, :ny].max(axis=(3, 4))
    keep = tmax > ROI_TRUTH_MAX_MIN
    if not keep.any():
        return 0.0
    return float(np.sqrt(sq_sum[keep] / (3 * roi * roi)).max())


def per_material_errors(truth: Array, pred: Array) -> dict[str, dict[str, float]]:
    """{material: {'mae': mean|t-r|, 'rmse': sqrt(mean (t-r)^2)}} pooled over cases and pixels."""
    t, r = _as4d(truth), _as4d(pred)
    out: dict[str, dict[str, float]] = {}
    for m, name in enumerate(MATERIALS):
        d = t[:, m] - r[:, m]
        out[name] = {"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d**2).mean()))}
    return out


def aggregate_rmse(truth: Array, pred: Array) -> float:
    """sqrt(mean over ALL cases, materials, pixels of (t-r)^2). Differs from s1 (pooled)."""
    t, r = _as4d(truth), _as4d(pred)
    return float(np.sqrt(((t - r) ** 2).mean()))


def regime_masks(truth: Array, tol: float = 1e-3) -> tuple[Array, Array, Array]:
    """Boolean (C,H,W) masks from the TRUTH map sum S: (air: S <= tol, tissue: |S-1| <= tol,
    band: neither). Disjoint and exhaustive.
    """
    s = _as4d(truth).sum(axis=1)
    air = s <= tol
    tissue = np.abs(s - 1.0) <= tol
    return air, tissue, ~(air | tissue)


def partition_errors(truth: Array, pred: Array, tol: float = 1e-3) -> dict[str, float]:
    """Fraction-conservation error of the PREDICTION's material sum Sp, regimes from truth.

    keys: 'tissue_mean_abs' = mean |Sp-1| on tissue pixels, 'tissue_max_abs', 'air_mean_abs' =
    mean |Sp| on air pixels, 'air_max_abs', 'n_tissue', 'n_air', 'n_band' (counts as floats).
    Means/maxes are NaN when the regime is empty. Reported separately, never combined (F-013).
    """
    sp = _as4d(pred).sum(axis=1)
    air, tissue, band = regime_masks(truth, tol)
    e_tissue = np.abs(sp[tissue] - 1.0)
    e_air = np.abs(sp[air])

    def stat(v: Array, fn: str) -> float:
        if v.size == 0:
            return math.nan
        return float(v.mean() if fn == "mean" else v.max())

    return {
        "tissue_mean_abs": stat(e_tissue, "mean"),
        "tissue_max_abs": stat(e_tissue, "max"),
        "air_mean_abs": stat(e_air, "mean"),
        "air_max_abs": stat(e_air, "max"),
        "n_tissue": float(tissue.sum()),
        "n_air": float(air.sum()),
        "n_band": float(band.sum()),
    }


def boundary_rmse(truth: Array, pred: Array, tol: float = 1e-3) -> float:
    """sqrt( sum over band pixels and 3 materials of (t-r)^2 / (3 * n_band) ); NaN if n_band == 0.

    Band = regime_masks(truth, tol)[2] (truth sum strictly between air and tissue).
    """
    t, r = _as4d(truth), _as4d(pred)
    band = regime_masks(t, tol)[2]
    n_band = int(band.sum())
    if n_band == 0:
        return math.nan
    sq = ((t - r) ** 2).sum(axis=1)[band].sum()
    return float(math.sqrt(sq / (3 * n_band)))
