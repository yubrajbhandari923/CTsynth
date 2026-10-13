"""Retrospective verification of the Experiment A identifiability calculations (F-018, F-024).

DETERMINISTIC: no training, no randomness, no fitted parameters. Re-derives every headline number
from the committed CC-BY AAPM model tables and writes them as machine-readable metrics so the
findings can be bound to evidence instead of transcribed by hand.

Run:  uv run python projects/p03_unsupervised/verify_identifiability.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ctsynth.aapm import classical as C
from ctsynth.aapm import schema as S
from ctsynth.hashing import sha256_file

TABLE_DIR = Path(__file__).resolve().parents[2] / "data" / "aapm_kit_sample"


def transmission(x: np.ndarray, tables: dict[int, S.ModelTable]) -> np.ndarray:
    """Exact polychromatic AAPM model: T_k = sum_E 0.5 s_k(E) exp(-sum_m mu_m(E) x_m), I0 = 1."""
    return np.array(
        [
            float((0.5 * t.spectrum * np.exp(-(t.mu_per_cm * x[:, None]).sum(axis=0))).sum())
            for t in (tables[50], tables[80])
        ]
    )


def exact_decoy(
    x_true: np.ndarray, calc: float, tables: dict[int, S.ModelTable], guess: np.ndarray
) -> np.ndarray:
    """Hold calcification at `calc`; solve the 2x2 polychromatic system for the other two.

    Damped Newton on log-transmission residuals with the analytic Jacobian (numpy only: the
    original used scipy.optimize.fsolve, an undeclared dependency).
    """
    t_true = transmission(x_true, tables)
    ts = (tables[50], tables[80])

    def resid(ab: np.ndarray) -> np.ndarray:
        return np.log(transmission(np.array([ab[0], ab[1], calc]), tables) / t_true)

    def jac(ab: np.ndarray) -> np.ndarray:
        x = np.array([ab[0], ab[1], calc])
        rows = []
        for t in ts:
            w = 0.5 * t.spectrum * np.exp(-(t.mu_per_cm * x[:, None]).sum(axis=0))
            rows.append(-(t.mu_per_cm[:2] * w).sum(axis=1) / w.sum())
        return np.array(rows)

    ab = np.asarray(guess, dtype=float).copy()
    for _ in range(200):
        r = resid(ab)
        if np.abs(r).max() < 1e-15:
            break
        step = np.linalg.solve(jac(ab), -r)
        lam = 1.0
        while lam > 1e-8 and np.linalg.norm(resid(ab + lam * step)) > np.linalg.norm(r):
            lam /= 2
        ab = ab + lam * step
    else:
        raise RuntimeError(f"decoy solve did not converge (residual {np.abs(resid(ab)).max():.2e})")
    return np.array([ab[0], ab[1], calc])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    tables = {k: S.load_model_table(TABLE_DIR / f"model_data_{k}kVp.npy", k) for k in S.KVPS}
    mu = C.effective_mu(tables)
    null = np.linalg.svd(mu)[2][-1]
    b_full = np.vstack([mu, np.ones(3)])
    sv = np.linalg.svd(b_full, compute_uv=False)
    amp = np.linalg.svd(np.linalg.inv(b_full)[:, :2], compute_uv=False)[0]

    out: dict[str, object] = {
        "kind": "retrospective_verification",
        "deterministic": True,
        "source_tables": {p.name: sha256_file(p) for p in sorted(TABLE_DIR.glob("*.npy"))},
        "effective_mu_per_cm": mu.tolist(),
        "null_direction": null.tolist(),
        "adipose_fibro_angle_deg": float(
            np.degrees(
                np.arccos(
                    min(
                        1.0,
                        mu[:, 0] @ mu[:, 1] / np.linalg.norm(mu[:, 0]) / np.linalg.norm(mu[:, 1]),
                    )
                )
            )
        ),
        "sum_to_one_singular_values": sv.tolist(),
        "sum_to_one_cond_zero_thickness": float(sv[0] / sv[-1]),
        "sigma_max_Binv_measurement_block": float(amp),
        "decoys": [],
    }

    # F-018's original (physically unreachable: negative fibroglandular) and F-024's nonnegative one
    for label, x_true_, calc, guess in [
        ("f018_original_negative", np.array([8.0, 4.0, 0.05]), 0.10, np.array([12.0, 0.0])),
        ("f024_nonnegative", np.array([1.0, 14.0, 0.2]), 0.35, np.array([15.0, 0.4])),
    ]:
        t_true = transmission(x_true_, tables)
        xd = exact_decoy(x_true_, calc, tables, guess)
        td = transmission(xd, tables)
        out["decoys"].append(  # type: ignore[union-attr]
            {
                "label": label,
                "truth_cm": x_true_.tolist(),
                "decoy_cm": xd.tolist(),
                "nonnegative": bool((xd >= 0).all()),
                "separation_cm": float(np.linalg.norm(xd - x_true_)),
                "rel_err_T50": float(abs(td[0] - t_true[0]) / t_true[0]),
                "rel_err_T80": float(abs(td[1] - t_true[1]) / t_true[1]),
            }
        )

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "identifiability.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    for d in out["decoys"]:  # type: ignore[union-attr]
        print(
            f"{d['label']:24s} sep {d['separation_cm']:7.3f} cm  nonneg {d['nonnegative']!s:5s} "
            f"relerr {max(d['rel_err_T50'], d['rel_err_T80']):.2e}"
        )
    print(
        f"cond(zero-thickness) {out['sum_to_one_cond_zero_thickness']:.2f}  "
        f"sigma_max(B^-1) {amp:.1f}  angle {out['adipose_fibro_angle_deg']:.3f} deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
