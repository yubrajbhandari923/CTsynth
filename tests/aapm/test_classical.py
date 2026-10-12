"""Classical image-domain decomposition (T-011). SYNTHETIC DATA ONLY: nothing here says anything
about AAPM performance. Independent references: mu matrix from the model tables via np.load, SVD,
a brute-force simplex projection, and a closed-form conditioning expectation."""

import numpy as np
import pytest

from ctsynth.aapm import classical as C
from ctsynth.aapm import synthetic as Y
from ctsynth.aapm.schema import ModelTable
from ctsynth.aapm.synthetic import SyntheticConfig

# F-012 as published (5 s.f.); tests check the pipeline reproduces it, then use the computed one.
F012_MU = np.array([[0.26697, 0.31321, 3.01182], [0.21930, 0.24788, 1.64238]])
F012_NULL = np.array([-0.723, 0.691, -0.008])
F012_COND = 1273.0


def _mu(tables: dict[int, ModelTable]) -> np.ndarray:
    """Independent of classical.effective_mu: explicit loops."""
    a = np.zeros((2, 3))
    for i, k in enumerate((50, 80)):
        for m in range(3):
            a[i, m] = sum(
                0.5 * s * u for s, u in zip(tables[k].spectrum, tables[k].mu_per_cm[m], strict=True)
            )
    return a


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))  # sign-free: (anti)parallel
    return float(np.degrees(np.arccos(min(c, 1.0))))


def _data(tables: dict[int, ModelTable], with_air: bool = False):
    d = Y.generate(SyntheticConfig(n_cases=3, n_pix=16, n_views=16, n_det=24, seed=5), tables)
    x = d.phantoms.astype(np.float64)
    s = x.sum(axis=1)
    tis = np.abs(s - 1) <= 1e-3
    x = np.where(tis[:, None], x / np.where(s == 0, 1, s)[:, None], x)  # float32 sums are 1 +- 1e-7
    mask = np.ones_like(s, bool) if with_air else tis
    lo = np.einsum("nmxy,m->nxy", x, _mu(tables)[0])
    hi = np.einsum("nmxy,m->nxy", x, _mu(tables)[1])
    return x, s, mask, lo, hi


def test_mu_matches_f012(tables) -> None:
    a = _mu(tables)
    assert np.allclose(a, F012_MU, atol=6e-6)  # F-012 is rounded to 5 decimals (half-ulp 5e-6)
    assert np.allclose(C.effective_mu(tables), a, rtol=1e-12)
    sv = np.linalg.svd(np.vstack([a, np.ones(3)]), compute_uv=False)
    assert abs(sv[0] / sv[2] - F012_COND) < 2.0  # 4-digit published cond


def test_exact_recovery_sum_to_one(tables) -> None:
    x, _, tis, lo, hi = _data(tables)
    got = C.decompose_lstsq(lo, hi, _mu(tables), mode="sum_to_one", air_cutoff=None)
    err = np.abs(got - x).transpose(1, 0, 2, 3)[:, tis]
    # float64 solve: error ~ eps_mach*cond(1273)*|x| ~ 3e-13; the images are built in float64 here.
    assert err.max() < 1e-10


def test_unconstrained_error_is_null_direction(tables) -> None:
    x, _, tis, lo, hi = _data(tables)
    mu = _mu(tables)
    got = C.decompose_lstsq(lo, hi, mu, mode="unconstrained", air_cutoff=None)
    e = (got - x).transpose(1, 0, 2, 3)[:, tis].T  # (P, 3)
    e = e[np.linalg.norm(e, axis=1) > 1e-3]
    assert len(e) > 20 and np.linalg.norm(e, axis=1).max() > 0.05  # it genuinely fails
    null = np.linalg.svd(mu)[2][-1]  # independent null vector of the 2x3 matrix
    # pinv returns the row-space projection, so error = -(n.x) n exactly: only roundoff remains
    assert max(_angle_deg(v, null) for v in e) < 1e-6
    # and it is the F-012 direction: published to 3 digits, |dn|~5e-4 => <0.1 deg; allow 0.2
    assert _angle_deg(null, F012_NULL) < 0.2
    # measurements are unchanged by the error, i.e. the error is invisible to the data
    assert np.abs(e @ mu.T).max() < 1e-12


def test_conditioning_amplification(tables) -> None:
    mu = _mu(tables)
    b_full = np.vstack([mu, np.ones(3)])
    inv2 = np.linalg.inv(b_full)[:, :2]  # maps an image perturbation to a fraction error
    u, s, vt = np.linalg.svd(inv2)
    x = np.array([0.5, 0.5, 0.0])  # typical tissue pixel
    d = mu @ x
    eps = 1e-6  # relative perturbation of the measurement vector
    dlt = eps * np.linalg.norm(d) * vt[0]  # worst direction (top right-singular vector of B^-1)
    lo = np.full((2, 2), d[0] + dlt[0])
    hi = np.full((2, 2), d[1] + dlt[1])
    got = C.decompose_lstsq(lo, hi, mu, mode="sum_to_one", air_cutoff=None)[:, 0, 0]
    err = got - x
    # expectation derived, not fitted: |err| = |B^-1 dlt| = sigma_max(B^-1[:, :2]) * |dlt|
    assert np.linalg.norm(err) == pytest.approx(s[0] * np.linalg.norm(dlt), rel=1e-4)
    amp = (np.linalg.norm(err) / np.linalg.norm(x)) / eps
    cond = np.linalg.cond(b_full)
    # derived: amp = sigma_max(B^-1[:, :2]) * |d| / |x|. Bounded by cond(B); the exact sum row of
    # [d;1] is not perturbed, so |d| < |[d;1]| and amp sits below cond
    assert amp == pytest.approx(s[0] * np.linalg.norm(d) / np.linalg.norm(x), rel=1e-4)
    assert amp <= cond and amp > cond / 10  # observed ~0.14 * cond, i.e. same order as 1273
    # error is the adipose<->fibroglandular trade (F-012), not calcification
    assert abs(err[0] + err[1]) < 0.05 * np.linalg.norm(err) and abs(
        err[2]
    ) < 0.05 * np.linalg.norm(err)


def test_air_is_exact_zero_not_sum_one(tables) -> None:
    x, s, _, lo, hi = _data(tables, with_air=True)
    mu = _mu(tables)
    air = s <= 1e-3
    assert air.any()
    for mode in ("unconstrained", "sum_to_one", "simplex"):
        got = C.decompose_lstsq(lo, hi, mu, mode=mode)
        assert np.all(got.transpose(1, 0, 2, 3)[:, air] == 0.0), mode  # exact, not approx
    # without the rule, sum_to_one forces air to sum 1: the failure F-013 warns about
    forced = C.decompose_lstsq(lo, hi, mu, mode="sum_to_one", air_cutoff=None)
    assert np.allclose(forced.sum(axis=1)[air], 1.0)
    # tissue pixels away from the boundary are unaffected by the rule
    full = np.abs(s - 1) <= 1e-3
    got = C.decompose_lstsq(lo, hi, mu, mode="sum_to_one")
    assert np.abs(got - x).transpose(1, 0, 2, 3)[:, full].max() < 1e-10


def test_boundary_pixels_follow_cutoff_rule(tables) -> None:
    mu = _mu(tables)
    xt = np.array([0.0, 1.0, 0.0])
    cov = np.array([0.1, 0.3, 0.6, 0.9])  # tissue coverage of boundary pixels (true sum)
    lo = (cov * (mu[0] @ xt)).reshape(1, 4)  # a 1x4 image
    hi = (cov * (mu[1] @ xt)).reshape(1, 4)
    below = (lo < C.AIR_CUTOFF_PER_CM) & (hi < C.AIR_CUTOFF_PER_CM)
    assert below[0].tolist() == [True, True, False, False]  # cutoff 0.1 vs 80 kVp FG mu 0.248*cov
    got = C.decompose_lstsq(lo, hi, mu, mode="sum_to_one")  # (3, 1, 4)
    assert np.all(got[:, 0, :2] == 0.0)  # declared air -> exact zero
    assert np.allclose(got[:, 0, 2:].sum(axis=0), 1.0)  # declared tissue -> forced sum 1 ...
    # ... and the unit-sum mismatch is amplified by the ill-conditioning (F-012): error >> 1 - cov
    assert np.abs(got[:, 0, 2] - xt).max() > 10 * (1 - cov[2])
    simp = C.decompose_lstsq(lo, hi, mu, mode="simplex")
    assert simp.min() >= 0 and np.allclose(simp[:, 0, 2:].sum(axis=0), 1.0)


def test_simplex_matches_independent_projection(tables) -> None:
    mu = _mu(tables)
    rng = np.random.default_rng(0)
    x = rng.dirichlet([1, 1, 1], size=40)
    b = x @ mu.T + rng.normal(0, 3e-4, (40, 2))  # noise pushes sum_to_one solution off the simplex
    lo, hi = b[:, 0].reshape(1, 40), b[:, 1].reshape(1, 40)  # 1x40 image
    raw = C.decompose_lstsq(lo, hi, mu, mode="sum_to_one", air_cutoff=None)[:, 0].T
    assert (raw < 0).any(), "test must exercise the projection"
    got = C.decompose_lstsq(lo, hi, mu, mode="simplex", air_cutoff=None)[:, 0].T
    assert got.min() >= 0 and np.allclose(got.sum(axis=1), 1.0, atol=1e-12)

    def brute(v: np.ndarray) -> np.ndarray:  # min |p-v| over the simplex: try every active set
        best, bd = None, np.inf
        for mask in range(1, 8):
            idx = [i for i in range(3) if mask >> i & 1]
            p = np.zeros(3)
            p[idx] = v[idx] - (v[idx].sum() - 1) / len(idx)
            if p.min() >= 0 and (dd := np.linalg.norm(p - v)) < bd:
                best, bd = p, dd
        assert best is not None
        return best

    want = np.array([brute(v) for v in raw])
    assert np.abs(got - want).max() < 1e-12  # both exact in float64
    for mode in ("sum_to_one", "simplex"):  # air stays outside the simplex
        z = C.decompose_lstsq(np.zeros((1, 3)), np.zeros((1, 3)), mu, mode=mode)
        assert np.all(z == 0.0)


def _tv_problem(tables):
    mu = _mu(tables)
    n = 24
    x = np.zeros((3, n, n))
    yy, xx = np.mgrid[:n, :n]
    tissue = (yy - 12) ** 2 + (xx - 12) ** 2 <= 10**2
    fg = tissue & (xx > 11)
    x[0][tissue & ~fg] = 1.0
    x[1][fg] = 0.8
    x[0][fg] = 0.2  # piecewise-constant, simplex-valued, air outside
    b = np.tensordot(mu, x, axes=(1, 0))
    b = b + np.random.default_rng(1).normal(0, 1e-3, b.shape)  # fixed seed: deterministic
    return mu, x, b


def _rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def test_tv_improves_on_noisy_data(tables) -> None:
    mu, x, b = _tv_problem(tables)
    base = C.decompose_tv(b[0], b[1], mu, lam=0.0)
    r0 = _rmse(base, x)
    # lam from a coarse scan on THIS synthetic problem only (1e-3 collapses structure; 1e-6 ~ no-op)
    r1 = _rmse(C.decompose_tv(b[0], b[1], mu, lam=1e-4, n_iter=300), x)
    assert r1 < 0.8 * r0, (r0, r1)  # observed ~36% lower; >20% required


def test_tv_objective_monotone(tables) -> None:
    mu, _, b = _tv_problem(tables)
    h: list[float] = []
    C.decompose_tv(b[0], b[1], mu, lam=1e-4, n_iter=200, history=h)
    h_arr = np.array(h)
    assert len(h_arr) == 200 and np.all(np.diff(h_arr) <= 0.0)  # exactly non-increasing (guarded)
    assert h_arr[-1] < h_arr[0] - 1e-3  # and it actually descends


def test_tv_lam_zero_is_unregularised(tables) -> None:
    mu, _, b = _tv_problem(tables)
    zero = C.decompose_tv(b[0], b[1], mu, lam=0.0, n_iter=50)
    ref = C.decompose_lstsq(b[0], b[1], mu, mode="simplex")
    # warm start is the simplex solution; with lam=0 FISTA can only lower the data term
    assert _rmse(zero, ref) < 5e-2
    tiny = C.decompose_tv(b[0], b[1], mu, lam=1e-12, n_iter=50)
    assert np.abs(tiny - zero).max() < 1e-6  # continuity in lam at 0


def test_tv_noiseless_exact_start_stays_exact(tables) -> None:
    mu, x, _ = _tv_problem(tables)
    b = np.tensordot(mu, x, axes=(1, 0))
    got = C.decompose_tv(b[0], b[1], mu, lam=0.0, n_iter=20)
    assert np.abs(got - x).max() < 1e-9  # exact optimum of the data term is a fixed point


def test_determinism(tables) -> None:
    mu, _, b = _tv_problem(tables)
    a1 = C.decompose_tv(b[0], b[1], mu, lam=1e-4, n_iter=30)
    a2 = C.decompose_tv(b[0], b[1], mu, lam=1e-4, n_iter=30)
    assert a1.tobytes() == a2.tobytes()
    s1 = C.decompose_lstsq(b[0], b[1], mu, mode="simplex")
    s2 = C.decompose_lstsq(b[0], b[1], mu, mode="simplex")
    assert s1.tobytes() == s2.tobytes()


def test_bad_inputs(tables) -> None:
    mu = _mu(tables)
    with pytest.raises(ValueError):
        C.decompose_lstsq(np.zeros((2, 2)), np.zeros((2, 3)), mu, mode="simplex")
    with pytest.raises(ValueError):
        C.decompose_lstsq(np.zeros((2, 2)), np.zeros((2, 2)), mu, mode="nope")  # type: ignore[arg-type]
