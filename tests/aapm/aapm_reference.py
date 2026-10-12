"""Independent numpy references for the AAPM adapter tests (nothing here imports ctsynth.aapm).

Self-checked by test_aapm_reference_sanity.py; these pass without any implementation.
"""

from __future__ import annotations

import math

import numpy as np

Arr = np.ndarray


def det_u(n_det: int, fov_cm: float, source_radius: float, sdd: float) -> Arr:
    """Detector bin centres u_j = -L/2 + (j+0.5) L/n_det, L = 2 tan(asin((fov/2)/R)) sdd (cm)."""
    length = 2.0 * math.tan(math.asin(fov_cm / 2.0 / source_radius)) * sdd
    return -length / 2.0 + (np.arange(n_det) + 0.5) * length / n_det


def ray_pixel_lengths(
    s: float, u: float, n_pix: int, fov_cm: float, source_radius: float, sdd: float
) -> Arr:
    """(n_pix, n_pix) exact length (cm) of one ray inside each pixel square [ix, iy].

    Source (R cos s, R sin s); detector point (R-sdd)(cos s, sin s) + u (-sin s, cos s); slab.
    """
    src = np.array([source_radius * math.cos(s), source_radius * math.sin(s)])
    cen = (source_radius - sdd) * np.array([math.cos(s), math.sin(s)])
    det = cen + u * np.array([-math.sin(s), math.cos(s)])
    d = det - src
    dx = fov_cm / n_pix
    edges = -fov_cm / 2.0 + dx * np.arange(n_pix + 1)
    lo, hi = edges[:-1], edges[1:]

    def interval(a: int) -> tuple[Arr, Arr]:
        if d[a] == 0.0:
            inside = (lo <= src[a]) & (src[a] <= hi)
            return (np.where(inside, -np.inf, np.inf), np.where(inside, np.inf, -np.inf))
        t1, t2 = (lo - src[a]) / d[a], (hi - src[a]) / d[a]
        return np.minimum(t1, t2), np.maximum(t1, t2)

    tx0, tx1 = interval(0)
    ty0, ty1 = interval(1)
    tmin = np.maximum.outer(tx0, ty0)
    tmax = np.minimum.outer(tx1, ty1)
    return np.maximum(0.0, tmax - tmin) * float(np.hypot(*d))


def ref_line_integrals(
    phantoms: Arr, n_pix: int, fov_cm: float, n_views: int, n_det: int, rad: float, sdd: float
) -> Arr:
    """P[case, material, view, bin] (cm), float64, full interlaced views, angle_k = 2 pi k / V."""
    n_case = phantoms.shape[0]
    p = np.zeros((n_case, 3, n_views, n_det))
    us = det_u(n_det, fov_cm, rad, sdd)
    for v in range(n_views):
        for j, u in enumerate(us):
            ch = ray_pixel_lengths(2 * math.pi * v / n_views, float(u), n_pix, fov_cm, rad, sdd)
            p[:, :, v, j] = np.einsum("nmxy,xy->nm", phantoms.astype(np.float64), ch)
    return p


def ref_transmission(p: Arr, spectrum: Arr, mu: Arr, kvp: int) -> Arr:
    """T = sum_E 0.5 s(E) exp(-sum_m mu_m(E) P_m) on the views of `kvp` (80 even, 50 odd)."""
    idx = np.arange(0 if kvp == 80 else 1, p.shape[2], 2)
    att = np.exp(-np.einsum("me,nmvd->nevd", mu, p[:, :, idx]))
    return np.einsum("e,nevd->nvd", 0.5 * spectrum, att)


def ref_s1(truth: Arr, pred: Arr) -> float:
    """Explicit loops: mean over cases of sqrt(sum (t-r)^2 / (3 H W))."""
    vals = []
    for c in range(truth.shape[0]):
        acc = 0.0
        for m in range(3):
            for x in range(truth.shape[2]):
                for y in range(truth.shape[3]):
                    acc += float(truth[c, m, x, y] - pred[c, m, x, y]) ** 2
        vals.append(math.sqrt(acc / (3 * truth.shape[2] * truth.shape[3])))
    return sum(vals) / len(vals)


def ref_s2(truth: Arr, pred: Arr, roi: int, include_last: bool = False) -> float:
    """Explicit loops s2; skip ROI if truth ROI max <= 0.01; positions range(H-roi[+1])."""
    worst = 0.0
    h = truth.shape[2]
    for c in range(truth.shape[0]):
        for x0 in range(h - roi + (1 if include_last else 0)):
            for y0 in range(h - roi + (1 if include_last else 0)):
                t = truth[c, :, x0 : x0 + roi, y0 : y0 + roi]
                if t.max() <= 0.01:
                    continue
                r = pred[c, :, x0 : x0 + roi, y0 : y0 + roi]
                worst = max(worst, math.sqrt(float(((t - r) ** 2).sum()) / (3 * roi * roi)))
    return worst
