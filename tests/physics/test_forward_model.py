"""Forward-model physics tests: T1, T2, T4, T5, T6, T7, T8, T12."""

import numpy as np
import physics_reference as ref
import torch

from ctsynth.physics import MaterialBasis, ParallelBeamProjector2D, SpectralForwardModel, Spectrum

F64 = torch.float64
NX, NY, DX = 6, 5, 0.2


def _ray_projector(dtype: torch.dtype = F64) -> ParallelBeamProjector2D:
    """One view along +x with detectors at the NY row centres: each ray crosses NX pixels."""
    return ParallelBeamProjector2D(NX, NY, DX, torch.tensor([0.0], dtype=dtype), NY, DX)


def _uniform(vals: tuple[float, float], dtype: torch.dtype = F64) -> torch.Tensor:
    return torch.stack([torch.full((NY, NX), v, dtype=dtype) for v in vals])


def _model(scatter: float | torch.Tensor | None = None) -> SpectralForwardModel:
    return SpectralForwardModel(ref.build_basis(), ref.build_spectrum(), _ray_projector(), scatter)


def _ref_counts_uniform(rho: tuple[float, float]) -> np.ndarray:
    """Closed-form expected counts (n_bin,) for a uniform field along a ray of length NX*DX."""
    a = ref.toy_arrays()
    return ref.counts_ref(a["w"], a["kappa_e"], np.array(rho) * NX * DX)


def test_t01_zero_path_equals_unattenuated() -> None:
    """Law: no material -> no attenuation -> counts_b = sum_e R[b,e] Phi[e] T[e] exactly.
    Reference: explicit python double loop over bins/energies. rtol 1e-13: sum reordering only."""
    a = ref.toy_arrays()
    want = np.array(
        [sum(a["resp"][b, e] * a["phi"][e] * a["trans"][e] for e in range(12)) for b in range(3)]
    )
    got = _model().expected_counts(torch.zeros(2, NY, NX, dtype=F64)).numpy()
    assert got.shape == (3, 1, NY)
    assert np.allclose(got, want[:, None, None] * np.ones((1, 1, NY)), rtol=1e-13, atol=0)


def test_t02_monoenergetic_beer_lambert() -> None:
    """Law: I = I0 exp(-sum_m kappa_m(E0) A_m) at one energy. Reference: closed-form power-law
    kappa at E0=61.7 keV (off-grid, so log-log interpolation must be exact for power laws) and
    A_m = rho_m * L with L = NX*DX. rtol 1e-12 (exp/log rounding)."""
    e0, n0, rho = 61.7, 1e6, (0.3, 0.9)
    spec = Spectrum(*(torch.tensor(v, dtype=F64) for v in ([e0], [n0], [1.0], [[1.0]])))
    model = SpectralForwardModel(ref.build_basis(), spec, _ray_projector())
    got = model.expected_counts(_uniform(rho)).numpy()
    expo = rho[0] * NX * DX * ref.kappa_pe(np.array(e0)) + rho[1] * NX * DX * ref.kappa_cs(
        np.array(e0)
    )
    assert np.allclose(got, n0 * np.exp(-expo), rtol=1e-12)


def test_t04_energy_response_hand_computed() -> None:
    """Counts follow lambda_b = sum_e R[b,e] Phi[e] T[e] exp(-A.kappa(E_e)). Hand case with two
    energies on grid nodes (40, 80 keV), R=[[E],[1,0]] (energy-integrating + counting-at-40),
    Phi=[1000,2000], T=[0.5,1]; reference: scalar python arithmetic. rtol 1e-13."""
    e = [40.0, 80.0]
    spec = Spectrum(
        *(
            torch.tensor(v, dtype=F64)
            for v in (e, [1000.0, 2000.0], [0.5, 1.0], [[40.0, 80.0], [1.0, 0.0]])
        )
    )
    model = SpectralForwardModel(ref.build_basis(), spec, _ray_projector())
    a = (0.2, 0.5)
    x = [a[0] * ref.kappa_pe(np.array(en)) + a[1] * ref.kappa_cs(np.array(en)) for en in e]
    l0 = 40 * 1000 * 0.5 * np.exp(-x[0]) + 80 * 2000 * 1.0 * np.exp(-x[1])
    l1 = 1000 * 0.5 * np.exp(-x[0])
    got = model.expected_counts_from_line_integrals(torch.tensor(a, dtype=F64).reshape(2, 1, 1))
    assert np.allclose(got.reshape(2).numpy(), [l0, l1], rtol=1e-13)


def test_t04_linearity_in_filtration_and_response() -> None:
    """Counts are linear in T, Phi and R: halving T halves every bin; scaling row b of R by c
    scales only bin b; a row equal to the sum of two rows gives the summed counts. Reference:
    the unmodified model's counts. rtol 1e-13 (exact scalings, rounding only)."""
    a = np.array([[[0.2, 1.0]], [[1.5, 4.0]]])  # (2,1,2) line integrals
    at = torch.tensor(a, dtype=F64)
    base_s = ref.build_spectrum()

    def ex(s: Spectrum) -> torch.Tensor:
        m = SpectralForwardModel(ref.build_basis(), s, _ray_projector())
        return m.expected_counts_from_line_integrals(at)

    base = ex(base_s)
    half = ex(
        Spectrum(base_s.energies, base_s.fluence, 0.5 * base_s.filter_transmission, base_s.response)
    )
    assert torch.allclose(half, 0.5 * base, rtol=1e-13, atol=0)
    r = base_s.response.clone()
    r[1] *= 3.0
    r = torch.cat([r, (base_s.response[0] + base_s.response[2])[None]])
    got = ex(Spectrum(base_s.energies, base_s.fluence, base_s.filter_transmission, r))
    assert torch.allclose(got[:3], torch.stack([base[0], 3 * base[1], base[2]]), rtol=1e-13, atol=0)
    assert torch.allclose(got[3], base[0] + base[2], rtol=1e-13, atol=0)


def test_t05_unit_rescaling_cm_to_mm() -> None:
    """Dimensional consistency: expressing lengths in mm (pixel size x10), densities in g/mm^3
    (rho/1000, ref_density/1000) and kappa in mm^2/g (x100) leaves the exponent kappa*rho*dl
    unchanged, so counts must be identical. Reference: the cm model. rtol 1e-12."""
    rho = _uniform((0.3, 0.9))
    cm = _model().expected_counts(rho)
    b = ref.build_basis()
    b_mm = MaterialBasis(b.names, b.ref_density / 1000.0, b.energies, b.kappa * 100.0)
    proj_mm = ParallelBeamProjector2D(NX, NY, DX * 10, torch.tensor([0.0], dtype=F64), NY, DX * 10)
    mm = SpectralForwardModel(b_mm, ref.build_spectrum(), proj_mm).expected_counts(rho / 1000.0)
    assert torch.allclose(mm, cm, rtol=1e-12, atol=0)


def test_t05_volume_fraction_is_not_mass_density() -> None:
    """rho_m = f_m * rho_ref_m (ref densities 2.0, 1.0 g/cm^3). Counts for f converted through
    the basis must match the closed-form reference using rho = f*ref, and must differ strongly
    (>5% in the lowest bin, where the pe exponent differs by x2) from wrongly feeding f as rho."""
    f = _uniform((0.5, 0.4))
    model = _model()
    rho = ref.build_basis().mass_density_from_volume_fraction(f)
    right = model.expected_counts(rho)[:, 0, 0].numpy()
    want = _ref_counts_uniform((0.5 * 2.0, 0.4 * 1.0))
    assert np.allclose(right, want, rtol=1e-11)
    wrong = model.expected_counts(f)[:, 0, 0].numpy()
    assert abs(wrong[0] / right[0] - 1.0) > 0.05


def test_t06_positive_finite_and_bounded() -> None:
    """Counts are finite, > 0 and <= unattenuated (exp(-x) <= 1) for physical rho >= 0 including
    heavy attenuation (pe exponent ~15 at 25 keV -> counts ~1e-2 > 0 in float64). Reference:
    the unattenuated sums; slack 1e-12 relative."""
    model = _model()
    unatt = ref.toy_arrays()["w"].sum(axis=1)[:, None, None]
    for rho in (
        _uniform((0.0, 0.0)),
        _uniform((0.3, 0.9)),
        _uniform((2.0, 1.0)),
        _uniform((2.0, 0.0)),
    ):
        c = model.expected_counts(rho).numpy()
        assert np.isfinite(c).all() and (c > 0).all()
        assert (c <= unatt * (1 + 1e-12)).all()


def test_t07_noiseless_is_deterministic_and_sampling_reproducible() -> None:
    """expected_counts uses no RNG: identical across calls, unaffected by global seed changes and
    leaves a generator's state untouched. sample_counts: same seed -> identical counts, different
    seed -> different counts (means ~1e4 so equality has probability ~0), global RNG untouched,
    non-negative integers. Exact (bitwise) comparisons."""
    model, rho = _model(), _uniform((0.2, 0.5))
    g = torch.Generator().manual_seed(5)
    state = g.get_state()
    torch.manual_seed(1)
    e1 = model.expected_counts(rho)
    torch.manual_seed(2)
    e2 = model.expected_counts(rho)
    assert torch.equal(e1, e2) and torch.equal(g.get_state(), state)
    gs = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
    a, b, c = (model.sample_counts(rho, gs(s)) for s in (11, 11, 12))
    assert torch.equal(a, b) and not torch.equal(a, c)
    glob = torch.get_rng_state()
    model.sample_counts(rho, gs(3))
    assert torch.equal(glob, torch.get_rng_state())
    assert (a >= 0).all() and torch.equal(a, a.round()) and a.dtype == F64 and not a.requires_grad


def test_t08_poisson_mean_and_variance() -> None:
    """Counts ~ Poisson(lambda). N = 400 draws x 5 identical rays = 2000 iid samples per bin
    (uniform field, one view -> all rays have the same lambda; lambda ~1e3..1e5). Seed fixed.
    Mean: z=(mean-lambda)/sqrt(lambda/N). Variance: Var(s^2) = (lambda+2 lambda^2)/N for Poisson
    (mu4-sigma^4 = lambda+2 lambda^2), z=(s^2-lambda)/sqrt((lambda+2lambda^2)/N). Both ~N(0,1)
    for N=2000; |z|<4.5 -> two-sided flake probability ~7e-6 per statistic (6 statistics)."""
    model, rho = _model(), _uniform((0.1, 0.3))
    lam = model.expected_counts(rho)[:, 0, 0].numpy()
    assert lam.min() > 500.0  # normal approximation of s^2 and of the mean is adequate
    g = torch.Generator().manual_seed(8008)
    draws = torch.stack([model.sample_counts(rho, g) for _ in range(400)])  # (400, bin, 1, NY)
    x = draws.permute(1, 0, 2, 3).reshape(3, -1).numpy()
    n = x.shape[1]
    assert n == 2000
    z_mean = (x.mean(axis=1) - lam) / np.sqrt(lam / n)
    z_var = (x.var(axis=1, ddof=1) - lam) / np.sqrt((lam + 2 * lam**2) / n)
    assert np.all(np.abs(z_mean) < 4.5) and np.all(np.abs(z_var) < 4.5)


def test_t08_scatter_adds_to_mean() -> None:
    """Scalar scatter s adds s*unattenuated_b to the expected counts (reference: spec definition,
    numpy). rtol 1e-13."""
    rho = _uniform((0.2, 0.5))
    base = _model().expected_counts(rho).numpy()
    unatt = ref.toy_arrays()["w"].sum(axis=1)[:, None, None]
    assert np.allclose(_model(0.02).expected_counts(rho).numpy(), base + 0.02 * unatt, rtol=1e-13)
    tens = torch.full((3, 1, NY), 7.5, dtype=F64)
    assert np.allclose(_model(tens).expected_counts(rho).numpy(), base + 7.5, rtol=1e-13)


def test_t12_recovery_of_line_integrals_from_noiseless_counts() -> None:
    """Identifiability: with >=2 bins of different energy weighting and 2 materials with distinct
    energy dependence, A is uniquely recoverable from noiseless counts. Model counts for three
    known A (g/cm^2) are inverted by an independent numpy Levenberg-Marquardt (physics_reference
    .solve_lm, closed-form kappa) from A0=(0.1,1). Tolerance rtol 1e-7: LM converges to a
    residual <1e-15; the condition number of the 3-bin Jacobian is O(10^2), model-vs-reference
    disagreement ~1e-12 -> expected error <<1e-7. Reference solver itself is validated in
    test_reference_sanity."""
    a = ref.toy_arrays()
    truths = np.array([[0.2, 2.0], [0.5, 8.0], [0.05, 0.5]])  # (3 rays, 2 materials)
    model = SpectralForwardModel(ref.build_basis(), ref.build_spectrum(), _ray_projector())
    y = model.expected_counts_from_line_integrals(
        torch.tensor(truths.T, dtype=F64).reshape(2, 1, 3)
    ).numpy()
    for i, truth in enumerate(truths):
        got = ref.solve_lm(y[:, 0, i], a["w"], a["kappa_e"], np.array([0.1, 1.0]))
        assert np.allclose(got, truth, rtol=1e-7), (truth, got)


def test_t12_jacobian_rank_and_one_bin_nonidentifiable() -> None:
    """3 bins/2 materials: counts Jacobian wrt A has rank 2 (sigma2/sigma1 > 1e-3 after row
    normalisation; reference sanity value is O(1e-1)). 1 bin/2 materials: Jacobian is 1x2 so
    rank 1, and two different A (A_cs differing by > 0.1 g/cm^2) give identical counts (reference:
    bisection level set in numpy). Count equality rtol 1e-12."""
    spec3 = ref.build_spectrum()
    basis = ref.build_basis()
    m3 = SpectralForwardModel(basis, spec3, _ray_projector())
    a0 = torch.tensor([0.3, 3.0], dtype=F64)
    f3 = lambda v: m3.expected_counts_from_line_integrals(v.reshape(2, 1, 1)).reshape(-1)  # noqa: E731
    jac = torch.autograd.functional.jacobian(f3, a0).numpy()
    sv = np.linalg.svd(jac / f3(a0).numpy()[:, None], compute_uv=False)
    assert sv[1] / sv[0] > 1e-3
    spec1 = Spectrum(spec3.energies, spec3.fluence, spec3.filter_transmission, spec3.response[:1])
    m1 = SpectralForwardModel(basis, spec1, _ray_projector())
    f1 = lambda v: m1.expected_counts_from_line_integrals(v.reshape(2, 1, 1)).reshape(-1)  # noqa: E731
    j1 = torch.autograd.functional.jacobian(f1, a0).numpy().reshape(1, 2)
    assert np.linalg.matrix_rank(j1) == 1
    a = ref.toy_arrays()
    a2 = ref.level_set_a2(a["w"][:1], a["kappa_e"], 0.1, float(f1(a0)))
    other = torch.tensor([0.1, a2], dtype=F64)
    assert abs(a2 - 3.0) > 0.1
    assert np.allclose(f1(other).numpy(), f1(a0).numpy(), rtol=1e-12)
