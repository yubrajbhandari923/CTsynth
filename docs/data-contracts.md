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
