# ruff: noqa: E501
"""EXP-002 numerics: measurement designs, Jacobian/Fisher analysis, per-ray Poisson-MLE (LM).

Everything is float64 torch. Counts model (identical to ctsynth.physics, checked in tests):
lambda_b(A) = sum_e W[b,e] exp(-sum_m A_m kappa_m(E_e)),  W = R * fluence (filter already in
the SpekPy fluence; transmission=1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DT = torch.float64
WINDOWS_KEV = ((20.0, 50.0), (50.0, 70.0), (70.0, 90.0), (90.0, 120.5))  # last includes 120
# (exposure kVp keys, response kind, photon share of N_tot)
DESIGNS: dict[str, dict[str, object]] = {
    "M1": {"exposures": ("fluence_120",), "kind": "ei"},
    "M2": {"exposures": ("fluence_80", "fluence_120"), "kind": "ei"},
    "M4": {"exposures": ("fluence_120",), "kind": "pc4"},
}


@dataclass(frozen=True)
class Design:
    name: str
    W: torch.Tensor  # (n_meas, n_E) response*fluence, scaled so W.sum() == n_tot
    n_tot: float

    @property
    def n_meas(self) -> int:
        return int(self.W.shape[0])


def perturb_fluence(
    phi: np.ndarray, energies: np.ndarray, tilt: float, escale: float
) -> np.ndarray:
    """Spectrum tilt (1 + tilt*(E-Ebar)/Ebar, Ebar = photon-mean energy) and energy-scale
    phi'(E) = phi(E/(1+escale)) (linear interpolation, zero outside grid)."""
    ebar = float((phi * energies).sum() / phi.sum())
    shifted = np.interp(energies / (1.0 + escale), energies, phi, left=0.0, right=0.0)
    return np.clip(shifted * (1.0 + tilt * (energies - ebar) / ebar), 0.0, None)


def build_design(
    name: str,
    spectra: dict[str, np.ndarray],
    energies: np.ndarray,
    n_tot: float,
    tilt: float = 0.0,
    escale: float = 0.0,
) -> Design:
    """Rows R: 'ei' = E/Ebar_nominal (energy-integrating, weight prop. to E, normalised so the
    unattenuated expected value equals the photon count); 'pc4' = window indicators (weight 1 per
    photon). Perturbation is applied to the fluence only; R is always the nominal one. Fluence is
    then rescaled so the TOTAL unattenuated sum_b sum_e R*Phi equals n_tot (exposures share equally)."""
    spec = DESIGNS[name]
    exps: tuple[str, ...] = spec["exposures"]  # type: ignore[assignment]
    rows = []
    for key in exps:
        nominal = spectra[key]
        phi = perturb_fluence(nominal, energies, tilt, escale) if (tilt or escale) else nominal
        if spec["kind"] == "ei":
            ebar = float((nominal * energies).sum() / nominal.sum())
            r = (energies / ebar)[None, :]
        else:
            r = np.stack(
                [((energies >= lo) & (energies < hi)).astype(float) for lo, hi in WINDOWS_KEV]
            )
        w = r * phi[None, :]
        rows.append(w * (n_tot / len(exps) / w.sum()))
    return Design(name, torch.from_numpy(np.concatenate(rows, axis=0)).to(DT), float(n_tot))


def counts(A: torch.Tensor, W: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """A (N,k), W (nB,nE), kappa (k,nE) -> lambda (N,nB)."""
    return torch.exp(-A @ kappa) @ W.T


def jacobian(
    A: torch.Tensor, W: torch.Tensor, kappa: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic J[n,b,m] = d lambda_b / d A_m = -sum_e W[b,e] kappa[m,e] exp(-A.kappa); returns (lam, J)."""
    t = torch.exp(-A @ kappa)  # (N,nE)
    lam = t @ W.T
    J = -torch.einsum("ne,be,me->nbm", t, W, kappa)
    return lam, J


def fisher_analysis(
    A: torch.Tensor, W: torch.Tensor, kappa: torch.Tensor, rank_rtol: float = 1e-6
) -> dict[str, torch.Tensor]:
    """Per-ray weighted design D = diag(lambda^-1/2) J; singular values, rank, cond, CRLB std.

    rank = #singular values > rank_rtol * s_max. Non-identifiable rays (rank < k) get NaN cond/crlb.
    """
    lam, J = jacobian(A, W, kappa)
    D = J / lam.clamp_min(1e-300).sqrt()[..., None]
    k = A.shape[1]
    sv = torch.linalg.svdvals(D)  # (N, min(nB,k)) descending
    rank = (sv > rank_rtol * sv[:, :1]).sum(dim=1)
    ident = rank == k
    cond = torch.where(ident, sv[:, 0] / sv[:, -1], torch.full_like(sv[:, 0], float("nan")))
    F = D.transpose(1, 2) @ D
    crlb = torch.full((A.shape[0], k), float("nan"), dtype=DT)
    if bool(ident.any()):
        crlb[ident] = torch.linalg.inv(F[ident]).diagonal(dim1=1, dim2=2).sqrt()
    return {"sv": sv, "rank": rank, "ident": ident, "cond": cond, "crlb": crlb}


def lm_poisson(
    y: torch.Tensor,
    W: torch.Tensor,
    kappa: torch.Tensor,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-ray Poisson MLE of A (N,k) by Levenberg-Marquardt (Marquardt diag scaling), start A=0.

    NLL = sum_b (lam_b - y_b log lam_b); gradient J^T(1 - y/lam); Gauss-Newton Hessian J^T diag(1/lam) J.
    Converged when the Newton decrement g^T H^-1 g < tol (scale-free, NLL units). No positivity,
    no priors. Returns (A_hat, converged bool (N,)); non-converged rays are returned as-is, not dropped.
    """
    n, k = y.shape[0], kappa.shape[0]
    A = torch.zeros(n, k, dtype=DT)
    mu = torch.full((n,), 1e-3, dtype=DT)
    conv = torch.zeros(n, dtype=torch.bool)
    eye = torch.eye(k, dtype=DT)

    def nll(lam: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (lam - y * lam.clamp_min(1e-300).log()).sum(dim=1)

    for _ in range(max_iter):
        act = ~conv
        if not bool(act.any()):
            break
        idx = act.nonzero().squeeze(1)
        Aa, ya = A[idx], y[idx]
        lam, J = jacobian(Aa, W, kappa)
        lam_c = lam.clamp_min(1e-300)
        g = torch.einsum("nbm,nb->nm", J, 1.0 - ya / lam_c)
        H = torch.einsum("nbm,nb,nbj->nmj", J, 1.0 / lam_c, J)
        dH = H.diagonal(dim1=1, dim2=2).clamp_min(1e-300)
        try:
            newton = torch.linalg.solve(H + 1e-12 * dH[..., None] * eye, g[..., None]).squeeze(-1)
        except torch.linalg.LinAlgError:
            newton = torch.full_like(g, float("nan"))
        dec = (g * newton).sum(dim=1)
        done = dec < tol
        conv[idx[done]] = True
        m = mu[idx][:, None, None]
        try:
            step = torch.linalg.solve(H + m * dH[..., None] * eye, -g[..., None]).squeeze(-1)
        except torch.linalg.LinAlgError:
            step = torch.zeros_like(g)
        cand = Aa + step
        ok = torch.isfinite(cand).all(dim=1)
        cand = torch.where(ok[:, None], cand, Aa)
        better = ok & (nll(counts(cand, W, kappa), ya) < nll(lam, ya)) & ~done
        A[idx[better]] = cand[better]
        mu[idx] = torch.where(better, (mu[idx] / 3).clamp_min(1e-12), (mu[idx] * 4).clamp_max(1e12))
    return A, conv & torch.isfinite(A).all(dim=1)
