"""Official AAPM metrics and diagnostic metrics. Arrays are (C, 3, H, W) float, canonical material
order, dimensionless volume fractions; a (3, H, W) input is treated as C == 1. All reductions are in
float64. No normalisation anywhere.
"""

from __future__ import annotations

from ctsynth.aapm.schema import Array

ROI_SIZE = 25
ROI_TRUTH_MAX_MIN = 0.01  # ROIs with truth.max() <= this are skipped


def s1_mean_rmse(truth: Array, pred: Array) -> float:
    """Official s1 = (1/C) sum_c sqrt( sum_{m,x,y} (t-r)^2 / (3*H*W) ). Units: volume fraction."""
    raise NotImplementedError


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
    raise NotImplementedError


def per_material_errors(truth: Array, pred: Array) -> dict[str, dict[str, float]]:
    """{material: {'mae': mean|t-r|, 'rmse': sqrt(mean (t-r)^2)}} pooled over cases and pixels."""
    raise NotImplementedError


def aggregate_rmse(truth: Array, pred: Array) -> float:
    """sqrt(mean over ALL cases, materials, pixels of (t-r)^2). Differs from s1 (pooled)."""
    raise NotImplementedError


def regime_masks(truth: Array, tol: float = 1e-3) -> tuple[Array, Array, Array]:
    """Boolean (C,H,W) masks from the TRUTH map sum S: (air: S <= tol, tissue: |S-1| <= tol,
    band: neither). Disjoint and exhaustive.
    """
    raise NotImplementedError


def partition_errors(truth: Array, pred: Array, tol: float = 1e-3) -> dict[str, float]:
    """Fraction-conservation error of the PREDICTION's material sum Sp, regimes from truth.

    keys: 'tissue_mean_abs' = mean |Sp-1| on tissue pixels, 'tissue_max_abs', 'air_mean_abs' =
    mean |Sp| on air pixels, 'air_max_abs', 'n_tissue', 'n_air', 'n_band' (counts as floats).
    Means/maxes are NaN when the regime is empty. Reported separately, never combined (F-013).
    """
    raise NotImplementedError


def boundary_rmse(truth: Array, pred: Array, tol: float = 1e-3) -> float:
    """sqrt( sum over band pixels and 3 materials of (t-r)^2 / (3 * n_band) ); NaN if n_band == 0.

    Band = regime_masks(truth, tol)[2] (truth sum strictly between air and tissue).
    """
    raise NotImplementedError
