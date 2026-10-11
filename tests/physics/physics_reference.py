"""Independent numpy reference code and synthetic fixtures for the forward-model tests.

Nothing here imports the implementation except the torch `build_*` helpers at the bottom (which
only construct objects). Self-checked by test_reference_sanity.py.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ctsynth.physics import MaterialBasis, ParallelBeamProjector2D, Spectrum

# ---------------------------------------------------------------- synthetic physics -------------


def kappa_pe(e: np.ndarray) -> np.ndarray:
    """Photoelectric-like toy mass attenuation 0.8*(E/50)^-3 cm^2/g (exact power law)."""
    return 0.8 * (np.asarray(e, dtype=float) / 50.0) ** -3.0


def kappa_cs(e: np.ndarray) -> np.ndarray:
    """Compton-like toy mass attenuation 0.2*(E/50)^-0.2 cm^2/g (exact power law)."""
    return 0.2 * (np.asarray(e, dtype=float) / 50.0) ** -0.2


REF_DENSITY = np.array([2.0, 1.0])  # g/cm^3, deliberately != 1 so f vs rho confusion is visible
EDGES = (25.0, 60.0, 90.0, 126.0)


def toy_arrays(n_energy: int = 12) -> dict[str, np.ndarray]:
    """Basis grid 10..150 keV (29 nodes); spectrum on n_energy points in 25..125 keV; 3 bins."""
    eg = np.linspace(10.0, 150.0, 29)
    e = np.linspace(25.0, 125.0, n_energy)
    phi = 1e5 * np.exp(-0.5 * ((e - 70.0) / 30.0) ** 2)
    trans = 1.0 - 0.6 * np.exp(-(e - 25.0) / 20.0)
    resp = np.stack(
        [((e >= lo) & (e < hi)).astype(float) for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True)]
    )
    return {
        "grid": eg,
        "kappa_grid": np.stack([kappa_pe(eg), kappa_cs(eg)]),
        "e": e,
        "phi": phi,
        "trans": trans,
        "resp": resp,
        "kappa_e": np.stack([kappa_pe(e), kappa_cs(e)]),  # closed form, NOT interpolated
        "w": resp * (phi * trans)[None, :],
    }


# ---------------------------------------------------------------- counts / solver --------------


def counts_ref(w: np.ndarray, kappa: np.ndarray, a: np.ndarray) -> np.ndarray:
    """lambda_b = sum_e W[b,e] exp(-sum_m A_m kappa[m,e]); a (n_mat,), returns (n_bin,)."""
    return w @ np.exp(-(a @ kappa))


def jac_ref(w: np.ndarray, kappa: np.ndarray, a: np.ndarray) -> np.ndarray:
    """d lambda_b / d A_m = -sum_e W[b,e] kappa[m,e] exp(-A.kappa[:,e]); (n_bin, n_mat)."""
    return -(w * np.exp(-(a @ kappa))[None, :]) @ kappa.T


def solve_lm(
    y: np.ndarray, w: np.ndarray, kappa: np.ndarray, a0: np.ndarray, iters: int = 500
) -> np.ndarray:
    """Levenberg-Marquardt on relative residuals (counts-y)/y; noiseless data."""
    a = a0.astype(float).copy()
    lam = 1e-3

    def resid(v: np.ndarray) -> np.ndarray:
        return (counts_ref(w, kappa, v) - y) / y

    r = resid(a)
    for _ in range(iters):
        jac = jac_ref(w, kappa, a) / y[:, None]
        h = jac.T @ jac
        g = jac.T @ r
        step = np.linalg.solve(h + lam * np.diag(np.diag(h)), -g)
        r_new = resid(a + step)
        if r_new @ r_new < r @ r:
            a, r, lam = a + step, r_new, lam / 3.0
        else:
            lam *= 3.0
        if r @ r < 1e-30:
            break
    return a


def level_set_a2(w1: np.ndarray, kappa: np.ndarray, a1: float, target: float) -> float:
    """Bisection for A_2 with counts(a1, A_2) == target (1 bin; monotone decreasing in A_2)."""
    lo, hi = 0.0, 200.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if counts_ref(w1, kappa, np.array([a1, mid]))[0] > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def loglog_interp(e: np.ndarray, grid: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """ln v linear in ln E between nodes (numpy scalar-free reference)."""
    return np.exp(np.interp(np.log(e), np.log(grid), np.log(vals)))


# ---------------------------------------------------------------- geometry ---------------------


def chord_rect(s: float, theta: float, xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    """Length of {s*n + t*d} inside the open rectangle (slab method). Parallel rays on/outside a
    slab boundary give 0 (edge rays are unspecified; tests avoid them)."""
    d = (math.cos(theta), math.sin(theta))
    p0 = (-s * math.sin(theta), s * math.cos(theta))
    t0, t1 = -math.inf, math.inf
    for p, dc, lo, hi in ((p0[0], d[0], xmin, xmax), (p0[1], d[1], ymin, ymax)):
        if abs(dc) < 1e-12:
            if p <= lo or p >= hi:
                return 0.0
        else:
            ta, tb = (lo - p) / dc, (hi - p) / dc
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    return max(0.0, t1 - t0)


def det_positions(n_det: int, spacing: float, offset: float = 0.0) -> np.ndarray:
    """s_j = (j-(n_det-1)/2)*spacing + offset."""
    return (np.arange(n_det) - (n_det - 1) / 2.0) * spacing + offset


def dense_matrix(
    nx: int, ny: int, dx: float, angles: np.ndarray, n_det: int, ds: float, off: float = 0.0
) -> np.ndarray:
    """Brute-force P of shape (n_view*n_det, ny*nx); row = v*n_det + j, column = iy*nx + ix."""
    s = det_positions(n_det, ds, off)
    mat = np.zeros((len(angles) * n_det, ny * nx))
    for v, th in enumerate(angles):
        for j in range(n_det):
            for iy in range(ny):
                for ix in range(nx):
                    xc, yc = (ix - (nx - 1) / 2) * dx, (iy - (ny - 1) / 2) * dx
                    mat[v * n_det + j, iy * nx + ix] = chord_rect(
                        s[j], th, xc - dx / 2, xc + dx / 2, yc - dx / 2, yc + dx / 2
                    )
    return mat


def disk_chord(s: np.ndarray, theta: float, radius: float, cx: float, cy: float) -> np.ndarray:
    """Analytic chord 2*sqrt(R^2 - (s - s_c)^2) of a disk; s_c = c . n(theta)."""
    sc = -cx * math.sin(theta) + cy * math.cos(theta)
    return 2.0 * np.sqrt(np.maximum(radius**2 - (s - sc) ** 2, 0.0))


def disk_image(
    nx: int, ny: int, dx: float, radius: float, cx: float, cy: float, val: float
) -> np.ndarray:
    """Pixel-centre rasterisation of a uniform disk."""
    xs = (np.arange(nx) - (nx - 1) / 2) * dx
    ys = (np.arange(ny) - (ny - 1) / 2) * dx
    xx, yy = np.meshgrid(xs, ys)
    return val * (((xx - cx) ** 2 + (yy - cy) ** 2) < radius**2).astype(float)


# ---------------------------------------------------------------- torch builders ---------------


def build_basis(dtype: torch.dtype = torch.float64) -> MaterialBasis:
    """Two-material toy basis ('pe', 'cs')."""
    a = toy_arrays()
    return MaterialBasis(
        names=("pe", "cs"),
        ref_density=torch.tensor(REF_DENSITY, dtype=dtype),
        energies=torch.tensor(a["grid"], dtype=dtype),
        kappa=torch.tensor(a["kappa_grid"], dtype=dtype),
    )


def build_spectrum(dtype: torch.dtype = torch.float64, n_energy: int = 12) -> Spectrum:
    """Toy 3-bin photon-counting spectrum."""
    a = toy_arrays(n_energy)
    t = lambda k: torch.tensor(a[k], dtype=dtype)  # noqa: E731
    return Spectrum(t("e"), t("phi"), t("trans"), t("resp"))


def build_projector(
    nx: int = 6,
    ny: int = 5,
    dx: float = 0.2,
    angles: tuple[float, ...] = (0.0, 0.7, 1.9),
    n_det: int = 7,
    ds: float = 0.2,
    off: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> ParallelBeamProjector2D:
    """Small non-square projector."""
    return ParallelBeamProjector2D(nx, ny, dx, torch.tensor(angles, dtype=dtype), n_det, ds, off)
