# Physics conventions — BINDING (implemented and tested in `ctsynth.physics`)

Status: fixed by T-001/T-002 (spec `docs/physics/forward-model-spec.md`, 69 frozen tests in `tests/physics/`).
Earlier draft items marked "?" are resolved here; anything still open is listed at the bottom.

## Units
| Quantity | Symbol | Unit |
|---|---|---|
| Photon energy | `E` | keV |
| Length, pixel size | | cm |
| Partial mass density (material field) | `ρ_m(x)` | g/cm³ |
| Mass attenuation coefficient | `κ_m(E) = (μ/ρ)_m` | cm²/g |
| Linear attenuation | `μ(E,x)` | 1/cm |
| Material line integral | `A_m` | g/cm² |
| Fluence per energy bin | `Φ(E)` | photons per detector element (bin-integrated) |

Units are *conventions, not enforced*: the code is unit-opaque, so any internally consistent rescaling
(e.g. mm with g/mm³ and mm²/g) gives identical counts. Test 5 (`test_units_rescaling`) checks this.

## Equations as implemented
1. **Mixture (idealised, additive):** `μ(E,x) = Σ_m ρ_m(x) κ_m(E)`.
   Partial mass densities add; no volume conservation and no mixing rules beyond additivity.
2. **Volume fraction vs mass density:** `ρ_m = f_m · ρ_ref,m`. The model accepts **mass densities only**;
   `MaterialBasis.mass_density_from_volume_fraction` is the only conversion, and a frozen test catches confusing the two.
3. **Line integral:** `A_m(view, det) = ∫ ρ_m dl` (g/cm²) — exact pixel chord lengths, so
   `exponent = Σ_m A_m κ_m(E)` is dimensionless.
4. **Polychromatic expected counts per detector bin b:**
   `λ_b = Σ_e R[b,e] · Φ[e] · T[e] · exp(−Σ_m A_m κ_m(E_e)) + s_b`
   A plain sum over the energy-grid index: `Φ` is already bin-integrated, so **no `dE` factor anywhere**.
   `R` is the detector response (row `∝ E` for energy-integrating; row = window indicator for photon counting;
   rows may overlap), `T` the source filtration transmission, `s_b` optional scatter.
5. **Noise:** `counts_b ~ Poisson(λ_b)`, drawn with an explicit `torch.Generator` only (never global RNG).
   `expected_counts` is deterministic and uses no RNG.
6. **Attenuation interpolation:** `ln κ` piecewise-linear in `ln E` (exact for power laws); energies outside
   the table grid raise `ValueError` — no extrapolation.
7. **Geometry (2D parallel beam):** image centred at the origin, pixel `(iy,ix)` centre
   `x=(ix−(nx−1)/2)·dx`, `y=(iy−(ny−1)/2)·dx`; view angle `θ` has direction `d=(cosθ,sinθ)`,
   normal `n=(−sinθ,cosθ)`; detector `j` is the line `{p : p·n = s_j}`,
   `s_j=(j−(n_det−1)/2)·det_spacing + det_offset`. Forward projection uses **exact chord lengths**
   (pixel-constant, no interpolation): a one-pixel structure is seen at full value or not at all.

## Array conventions
`ρ` is `(n_material, ny, nx)`; line integrals and sinograms are `(…, n_view, n_det)`; counts are `(n_bin, n_view, n_det)`;
spectra are `(n_energy,)`; `κ` is `(n_material, n_energy)`. float64 is the physics default; float32 is allowed with looser tolerances.

## Differentiability
Autograd w.r.t. `ρ`, fluence, filter transmission, response and tensor scatter. The adjoint is the exact
transpose of the forward projector (`⟨Px,y⟩=⟨x,Pᵀy⟩` to ~1e-12 relative in float64).
Verified by finite differences and `torch.autograd.gradcheck`.

## Deliberate simplifications (v0 — all out of scope, none "approximately handled")
Monochromatic-per-grid-point energy sampling (no intra-bin integration); no scatter physics (scatter is an
additive user-supplied term only); no detector blur, crosstalk, pile-up, afterglow or finite focal spot;
no beam-hardening *correction*; no polar/cone geometry; no partial-volume blur; no motion; no electronic noise
(Poisson only); additive mixtures without volume conservation; rays exactly on a pixel edge are unspecified.

## Numerical limits
Extreme attenuation underflows to exactly 0 counts (gradients stay finite). Negative line integrals are
unguarded (`exp` overflows) — physical inputs are non-negative. The projector stores a dense chord matrix
`O(n_view·n_det·ny·nx)`: fine for toy sizes, unsuitable for production volumes.

## Data provenance
Attenuation tables and spectra are **frozen versioned files** (`data/tables/*_v1.npz` + `provenance_v1.json`),
generated offline (xraylib 4.3.0, SpekPy 2.5.4); runtime never imports those libraries (D-001).
**Known limit (D-002):** the 1 keV grid under-resolves the iodine K-edge at 33.17 keV
(κ jumps 6.64 → 33.6 cm²/g between 33 and 34 keV); quantitative iodine-concentration claims require table v2 (T-006).
Absolute spectrum normalisation is UNVERIFIED, so experiments rescale fluence to a stated photon count per ray.

## Still open
Tissue-realistic material bases (ICRU 44/46 mixtures) and whether to model elemental ρ/Z_eff instead of
material pairs; 3D geometry; energy-bin integration within a bin; HU conventions (no HU in the code yet).
