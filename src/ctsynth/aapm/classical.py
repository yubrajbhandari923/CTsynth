"""Classical constrained image-domain material decomposition (task T-011).

Maps two reconstructed CT images (50 kVp "low", 80 kVp "high") to three volume-fraction maps
(adipose, fibroglandular, calcification) by per-pixel inversion of the 2x3 effective-attenuation
matrix (docs/aapm/classical-baseline.md, findings F-012, F-013).

!!! VALIDATED ON SYNTHETIC DATA ONLY. No AAPM case data exists locally; no claim about AAPM
performance follows from this module. Image-domain inversion is expected to be far worse than the
published transmission-domain winners (D-005, s1 = 6.8e-7). !!!

Two-regime partition (F-013): pixels are AIR (all three fractions 0) or TISSUE (fractions sum to 1);
boundary pixels (0 < sum < 1) exist and are NOT representable by either regime. Sum-to-one is
imposed only on pixels not declared air (see `AIR_CUTOFF_PER_CM`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ctsynth.aapm.schema import KVPS, Array, ModelTable

Mode = Literal["unconstrained", "sum_to_one", "simplex"]

# Air rule: a pixel is AIR iff BOTH images are below this value (1/cm). MODELLING CHOICE, not a fact
# from the data. Default = half the smallest tissue effective mu in F-012 (adipose at 80 kVp,
# 0.2193/cm), so a boundary pixel with tissue coverage below ~45% is declared air and above it
# tissue. On real data choose it on the VALIDATION split (e.g. the valley of the image histogram
# between the air peak and the tissue peak); never on test data. Pass None to disable (all tissue).
AIR_CUTOFF_PER_CM = 0.1
TV_EPS = 1e-2  # Huber smoothing of |grad| in the TV term (fraction units); makes TV differentiable


def effective_mu(tables: dict[int, ModelTable]) -> Array:
    """(2, 3) float64 matrix A[k, m] = sum_E 0.5 * s_k(E) * mu_m(E); rows (50, 80) kVp (F-012)."""
    return np.stack(
        [(0.5 * tables[k].spectrum * tables[k].mu_per_cm).sum(axis=1) for k in KVPS]
    ).astype(np.float64)


def air_mask(
    img_low: Array, img_high: Array, air_cutoff: float | None = AIR_CUTOFF_PER_CM
) -> Array:
    """Boolean mask, True where both images are below `air_cutoff` (1/cm); all False if None."""
    if air_cutoff is None:
        return np.zeros(np.shape(img_low), dtype=bool)
    return np.asarray((np.asarray(img_low) < air_cutoff) & (np.asarray(img_high) < air_cutoff))


def project_simplex(v: Array) -> Array:
    """Euclidean projection of each row of v (P, 3) onto {x >= 0, sum x = 1} (sort-based)."""
    u = -np.sort(-v, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    k = np.arange(1, v.shape[1] + 1)
    rho = (u - css / k > 0).sum(axis=1)  # >= 1 always (first term is positive)
    theta = css[np.arange(len(v)), rho - 1] / rho
    return np.maximum(v - theta[:, None], 0.0)


def decompose_lstsq(
    img_low: Array,
    img_high: Array,
    mu: Array,
    *,
    mode: Mode,
    air_cutoff: float | None = AIR_CUTOFF_PER_CM,
) -> Array:
    """Per-pixel inversion of b = mu @ x. Images (..., H, W) -> fractions (..., 3, H, W), float64.

    mode "unconstrained": min-norm pinv(mu) b. Expected to be BAD: it cannot see the 2x3 null
        direction (F-012), so the error is exactly parallel to it.
    mode "sum_to_one": exact 3x3 solve of [mu; 1 1 1] x = [b; 1] (cond ~1273, F-012).
    mode "simplex": Euclidean projection of the sum_to_one solution onto the probability simplex
        (NOT the weighted-least-squares optimum over the simplex; `decompose_tv` with lam=0 and a
        long run approaches that).
    Pixels declared air by `air_mask` are returned as exact zeros in every mode. Boundary pixels
    (0 < true sum < 1) above the cutoff are treated as tissue: the unit-sum row then conflicts with
    the measurement and the conflict is amplified by cond ~1273 (errors far larger than 1 - sum,
    with large negative values under sum_to_one); below the cutoff they are zeroed.
    Validated on synthetic data only.
    """
    mu = np.asarray(mu, dtype=np.float64)
    lo, hi = np.asarray(img_low, np.float64), np.asarray(img_high, np.float64)
    if lo.shape != hi.shape or lo.ndim < 2 or mu.shape != (2, 3):
        raise ValueError("images must share shape (..., H, W) and mu must be (2, 3)")
    air = air_mask(lo, hi, air_cutoff)
    b = np.stack([lo.ravel(), hi.ravel()], axis=1)  # (P, 2)
    if mode == "unconstrained":
        x = b @ np.linalg.pinv(mu).T
    elif mode in ("sum_to_one", "simplex"):
        full = np.vstack([mu, np.ones(3)])
        x = np.concatenate([b, np.ones((len(b), 1))], axis=1) @ np.linalg.inv(full).T
        if mode == "simplex":
            x = project_simplex(x)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    x[air.ravel()] = 0.0
    return np.moveaxis(x.reshape(*lo.shape, 3), -1, -3)


def _grad(u: Array) -> Array:
    """Forward differences with Neumann boundary: (3, H, W) -> (2, 3, H, W)."""
    g = np.zeros((2, *u.shape))
    g[0, :, :-1, :] = u[:, 1:, :] - u[:, :-1, :]
    g[1, :, :, :-1] = u[:, :, 1:] - u[:, :, :-1]
    return g


def _div_t(p: Array) -> Array:
    """Adjoint of `_grad` (i.e. -div): (2, 3, H, W) -> (3, H, W)."""
    out = np.zeros(p.shape[1:])
    out[:, :-1, :] -= p[0, :, :-1, :]
    out[:, 1:, :] += p[0, :, :-1, :]
    out[:, :, :-1] -= p[1, :, :, :-1]
    out[:, :, 1:] += p[1, :, :, :-1]
    return out


def _tv_and_grad(x: Array, eps: float) -> tuple[float, Array]:
    """Isotropic Huber-smoothed TV: sum_{pix,mat} sqrt(gx^2+gy^2+eps^2) - eps, and its gradient."""
    g = _grad(x)
    r = np.sqrt((g**2).sum(axis=0) + eps**2)
    return float((r - eps).sum()), _div_t(g / r)


def decompose_tv(
    img_low: Array,
    img_high: Array,
    mu: Array,
    *,
    lam: float,
    n_iter: int = 300,
    air_cutoff: float | None = AIR_CUTOFF_PER_CM,
    tv_eps: float = TV_EPS,
    history: list[float] | None = None,
) -> Array:
    """Minimise 0.5 sum_pix ||mu x - b||^2 + lam * TV_eps(x) over x with x >= 0, sum x = 1 on tissue
    pixels and x = 0 on air pixels (air rule as in `decompose_lstsq`). Images (H, W) or (N, H, W).

    Monotone FISTA (Beck & Teboulle 2009) on a convex problem: fixed step 1/L with
    L = sigma_max(mu)^2 + 8*lam/tv_eps, the projection onto the constraint set exact, and an
    objective-increase guard so the recorded objective never increases. Deterministic (no RNG).
    Warm-started at the closed-form `simplex` solution (so noiseless data stay exactly optimal).
    With cond ~1273 the data term barely constrains the adipose/fibroglandular direction, so
    n_iter is a budget, not a convergence proof; too-large lam collapses structure (observed on
    synthetic data) and the benefit is modest and lam-sensitive.
    `lam` is NOT tuned here: select it on the VALIDATION split only. If `history` is a list the
    objective after every iteration is appended (single-image input). Synthetic validation only.
    """
    mu = np.asarray(mu, dtype=np.float64)
    lo, hi = np.asarray(img_low, np.float64), np.asarray(img_high, np.float64)
    if lo.ndim == 3:
        return np.stack(
            [
                decompose_tv(a, c, mu, lam=lam, n_iter=n_iter, air_cutoff=air_cutoff, tv_eps=tv_eps)
                for a, c in zip(lo, hi, strict=True)
            ]
        )
    if lam < 0 or n_iter < 0 or lo.shape != hi.shape or lo.ndim != 2:
        raise ValueError("need lam >= 0, n_iter >= 0, 2-D images of equal shape")
    air = air_mask(lo, hi, air_cutoff)
    b = np.stack([lo, hi])  # (2, H, W)
    h, w = lo.shape

    def project(x: Array) -> Array:
        y = project_simplex(x.reshape(3, -1).T)
        y[air.ravel()] = 0.0
        return np.asarray(y.T.reshape(3, h, w))

    def objective(x: Array) -> float:
        r = np.tensordot(mu, x, axes=(1, 0)) - b
        return 0.5 * float((r**2).sum()) + lam * _tv_and_grad(x, tv_eps)[0]

    lip = np.linalg.norm(mu, 2) ** 2 + 8.0 * lam / tv_eps
    x = decompose_lstsq(lo, hi, mu, mode="simplex", air_cutoff=air_cutoff)
    y, t, fx = x.copy(), 1.0, objective(x)
    for _ in range(n_iter):
        grad = np.tensordot(mu.T, np.tensordot(mu, y, axes=(1, 0)) - b, axes=(1, 0))
        if lam > 0:
            grad = grad + lam * _tv_and_grad(y, tv_eps)[1]
        cand = project(y - grad / lip)
        fc = objective(cand)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        if fc <= fx:  # monotone step accepted
            y = cand + ((t - 1.0) / t_new) * (cand - x)
            x, fx = cand, fc
        else:  # reject; keep x, restart momentum
            y, t_new = x.copy(), 1.0
        t = t_new
        if history is not None:
            history.append(fx)
    return x
