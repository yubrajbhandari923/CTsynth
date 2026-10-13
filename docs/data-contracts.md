# Data contracts (v0 — specs only; no loader code until P01-T01 tests freeze)

Every dataset/run directory has `manifest.json` (or `run_meta.json`) with: schema version, producer commit, config hash, units, array shapes/dtypes.

| Object | Shape / dtype | Notes |
|---|---|---|
| `MaterialMap` | `(n_material, z, y, x)` float32 | fractions or densities (declared); basis names listed; units per conventions |
| `Spectrum` | `energies (E,) keV`, `weights (E,)` | normalised; includes source kVp, filtration, detector response flag |
| `Sinogram` | `(n_spectrum, view, row, col)` float32 | counts or line integrals (declared); geometry in manifest |
| `Image` | `(n_spectrum, z, y, x)` float32 | HU or μ (declared) + reconstruction kernel |

External-data adapters (e.g. AAPM DL-Spectral CT: 2D, 3 tissue maps, kV-switching 50/80 kVp — see `literature/aapm-dl-spectral-ct.md`) convert into these objects; they never define them.
Open: coordinate/geometry metadata schema, label vocabularies, versioning policy.

## Paired material/CT records (implemented, paired-ct/0.1)
Supersedes the v0 `MaterialMap` axis note for paired data: `src/ctsynth/paired/schema.py` (docstring is binding).
Truth `(n_material, nz, ny, nx)` = `(iz, iy, ix)`, semantics declared per material (volume fraction vs partial mass
density), affine voxel→mm in a named frame, unknown calibration declared in `unknown` (never defaulted), sha256 of every
array verified on load, split unit = `phantom_id`. Producers: mock (`paired/mock.py`), XCIST (`paired/xcist.py`,
docs/xcist.md). DukeSim: not implemented (docs/dukesim-request.md). The AAPM adapter's `[m, ix, iy]` order is a
different, older convention and must be transposed explicitly if ever converted.
