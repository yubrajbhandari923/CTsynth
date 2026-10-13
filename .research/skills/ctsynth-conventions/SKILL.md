---
name: ctsynth-conventions
description: Use when writing or reviewing code that handles volumes, axes, units, materials, spectra or attenuation tables (AAPM kit vs paired-ct vs physics).
---

# ctsynth-conventions

Authoritative text: `docs/physics-conventions.md` (units, equations), `docs/data-contracts.md` (array layouts),
`docs/aapm/adapter-spec.md` (kit layout). This skill only says where they disagree on purpose.

**Three layouts, three libraries — never mix silently**
| Source | Layout | Where |
|---|---|---|
| AAPM kit | map `a[ix, iy]`, case `(M, ix, iy)`, fractions, μ in cm⁻¹, channels 0=50 kVp, 1=80 kVp | `ctsynth.aapm` |
| paired-ct | truth `(M, nz, ny, nx)` = `(iz, iy, ix)`, affine voxel→mm in a named frame, row 0 = +y | `ctsynth.paired` |
| physics | ρ in g/cm³ (torch), κ(E) from `data/tables/attenuation_v1.npz`, 1 keV grid | `ctsynth.physics` |
Convert AAPM↔paired only with `ctsynth.aapm.convert` (index relabel; it invents no affine). `physics.SpectralForwardModel` and
`aapm.AapmTorchForward` are different models on purpose — do not merge or substitute.

**Rules**
- Name units in identifiers (`_mm`, `_cm`, `_keV`) and state line integral (g/cm²) vs density (g/cm³) in docstrings.
- Frozen tables (`data/tables/*_v1.*`, `data/aapm_kit_sample/`) are never edited; regenerate only via `scripts/make_tables.py` to a new version.
- Randomness only via `ctsynth.seeding.rng(seed, *stream)`.
- Tests for any axis/orientation code use asymmetric shapes (distinct axis lengths, off-centre features) so a permutation changes the result.

**Verify**: `uv run pytest -m "physics or unit"`; for new conversions add a marker-pixel test like `tests/aapm/test_convert.py`.
**Record**: a changed convention is a decision (`research decision create`), not a code comment.
