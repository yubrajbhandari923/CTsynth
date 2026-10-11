# Forward model spec (P01-T01, v0) — binding for `ctsynth.physics`; frozen tests in `tests/physics/`

Library review: no third-party projector/physics library is a dependency (ct_laboratory has no licence; LEAP, XrayPhysics, xcist are optional oracles only, tests must skip when absent). Reference implementation is plain torch.

## Units and fields
keV; cm; g/cm^3; cm^2/g. Material fields are **partial mass densities** `rho_m(x)` (g/cm^3), array `(n_material, ny, nx)`. Volume fraction `f_m` relates by `rho_m = f_m * rho_ref_m` (`MaterialBasis.mass_density_from_volume_fraction`); the model never accepts fractions. `mu(E,x) = sum_m rho_m(x) kappa_m(E)` (1/cm). Line integral `A_m = int rho_m dl` (g/cm^2); exponent `sum_m A_m kappa_m(E)` is dimensionless. The code treats units opaquely: consistent rescaling (mm, g/mm^3, mm^2/g) must give identical counts.

## MaterialBasis
`names`, `ref_density (n_mat,)`, `energies (n_E,)` strictly increasing, `kappa (n_mat, n_E)` > 0 finite. `kappa_at(E)`: ln kappa piecewise linear in ln E (exact for power laws); E outside `[energies[0], energies[-1]]` -> `ValueError`. Data vendoring (NIST/xraylib) is a separate task. All tensors share dtype (float32 or float64) and device; violations -> `ValueError`.

## Spectrum and counts
`energies (n_E,)`, `fluence (n_E,)` >= 0 = photons per detector element per unit exposure in the bin represented by `E_e` (already integrated over bin width; no dE factor anywhere), `filter_transmission (n_E,)` in [0,1], `response R (n_bin, n_E)` >= 0 (energy-integrating row: `R=E`; photon-counting row: indicator of a window; rows may overlap).
`lambda_b(A) = sum_e R[b,e] Phi[e] T[e] exp(-sum_m A_m kappa_m(E_e)) + scatter_b` — a plain sum over grid index. `kappa_m(E_e)` comes from `basis.kappa_at(spectrum.energies)` (all spectrum energies must be inside the basis grid). Result >= 0 and finite; extreme exponents underflow to exactly 0 (plus scatter), gradients stay finite. Scatter (minimal): `None`; float `s >= 0` -> `scatter_b = s * sum_e R Phi T` (uniform); tensor `(n_bin, n_view, n_det) >= 0` added verbatim.
Differentiable w.r.t. rho, `fluence`, `filter_transmission`, `response`, scatter tensor (no detach/in-place). `sample_counts(rho, generator)` = `torch.poisson(lambda, generator=generator)` (explicit generator only; detached; rate 0 -> 0). `expected_counts` uses no RNG.

## Projector (2D parallel beam, exact Siddon, pixel-constant, matrix-free)
Image `(.., ny, nx)`, pixel `(iy,ix)` = square of side `pixel_size`, centre `x=(ix-(nx-1)/2)*dx`, `y=(iy-(ny-1)/2)*dx` (origin at image centre, y increases with row index). View angle `th` (rad): `d=(cos th, sin th)`, `n=(-sin th, cos th)`. Detector `j` = line `{p : p.n = s_j}`, `s_j=(j-(n_det-1)/2)*det_spacing + det_offset`. `(P x)[v,j] = sum_pixels x * |line_vj ∩ pixel|` (exact chord length, no interpolation, no partial-volume blur: a 1-pixel line is seen at full value or not at all). Rays exactly on a pixel edge are **unspecified** (callers/tests avoid them). Sinogram `(.., n_view, n_det)`; leading dims are independent channels. `adjoint` is the exact transpose (`<Px,y> = <x,P^T y>` to ~1e-12 relative in float64). Uniform fields give `rho * analytic chord` exactly; discretisation error vs curved objects is O(pixel size).

## SpectralForwardModel(basis, spectrum, projector, scatter=None)
`line_integrals(rho)->(n_mat,n_view,n_det)`; `expected_counts_from_line_integrals(A)` for `A (n_mat, *S) -> (n_bin, *S)` (tensor scatter requires `S=(n_view,n_det)`); `expected_counts(rho)->(n_bin,n_view,n_det)`; `sample_counts(rho, generator)`; `.to(device, dtype)` on every component. `rho` dtype must equal model dtype else `ValueError`. Validation errors are `ValueError`. float64 default; float32 allowed (tests use stated looser tolerances); MPS float32 only; CUDA not required.

## Notes for Opus / implementer
- Tests construct objects positionally: `MaterialBasis(names, ref_density, energies, kappa)`, `Spectrum(energies, fluence, filter_transmission, response)`, `ParallelBeamProjector2D(nx, ny, pixel_size, angles, n_det, det_spacing, det_offset=0.0)`.
- Tensors with `requires_grad` may be passed in dataclass fields; implementations must keep the graph.
- Spectrum may have a single energy; n_view, n_det, n_bin, nx, ny may be 1.
