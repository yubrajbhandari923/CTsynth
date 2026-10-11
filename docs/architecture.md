# Architecture

Layers: `ctsynth` core utilities → `physics` (spectra, attenuation, projector, forward model) →
(planned) `phantom` → `inference` (P01) → `eval`. Experiments live in `projects/`, not in the package.

## Package (`src/ctsynth/`)
| Module | Role |
|---|---|
| `seeding.py` | one root seed → independent streams via `SeedSequence` (`rng(seed, *stream)`) |
| `config.py` | YAML → frozen pydantic `ExperimentConfig` (`extra=forbid`); `experiment_id` = Research OS EXP id |
| `runmeta.py` | `run_meta.json`: config hash, git commit/dirty, versions — works without Research OS (HPC) |
| `device.py` | `cpu` default; `mps`/`cuda` opt-in, fail loudly if unavailable |
| `physics/materials.py` | `MaterialBasis`: κ(E) table, log-log interpolation, no extrapolation, mass-density conversion |
| `physics/spectrum.py` | `Spectrum`: fluence, filtration, detector response matrix R |
| `physics/projector.py` | `ParallelBeamProjector2D`: exact chord lengths, exact adjoint, autograd |
| `physics/forward.py` | `SpectralForwardModel`: ρ → line integrals → polychromatic counts → Poisson |

Physics equations and units: `docs/physics-conventions.md`. Interface contract: `docs/physics/forward-model-spec.md`.
69 frozen tests in `tests/physics/` define correct behaviour; they were written before the implementation and must not be edited by implementers.

## Data
Frozen versioned tables in `data/tables/` (`attenuation_v1.npz`, `spectra_v1.npz`, `provenance_v1.json`),
generated offline by `scripts/make_tables.py` using xraylib and SpekPy. **Runtime never imports those libraries** (D-001).
Regenerating produces byte-identical files; `tests/data/` checks hashes and NIST spot values.

## Experiments
`projects/p01_material_inference/exp002/` is the template: a `run_*.py` that writes `metrics.json` +
`run_meta.json` into `results/`, a separate `figures.py` depending only on that JSON (matplotlib is not a
project dependency), and tests in `tests/exp002/`. Runs are executed through `research run exec` on a clean
tree so results are citable.

## Decisions
- **D-001** No third-party physics/projector as a runtime dependency; `ctsynth.physics` is our own torch
  implementation. LEAP/XCIST are opt-in oracles; ct_laboratory is unlicensed and may only be read, never copied.
- **D-002** The v1 table under-resolves the iodine K-edge; absolute iodine-concentration claims are out of scope
  until table v2 (T-006).
- **Research OS owns experiment tracking**; we do not build a tracker. `runmeta.py` only covers HPC runs where
  `research run exec` cannot run.
- **pydantic + YAML, no Hydra**; **uv + `uv.lock`**; **src layout**; `mypy --strict` on `src/`; ruff lint+format.
- torch is the array type inside `physics` (autograd); numpy remains the contract for stored data.

## Known structural limits
Dense chord matrix is `O(n_view·n_det·ny·nx)` — toy sizes only, unsuitable for production volumes.
2D parallel beam only. No scatter/detector physics (see the simplification list in `physics-conventions.md`).
