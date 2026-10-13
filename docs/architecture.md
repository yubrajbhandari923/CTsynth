# Architecture

Package `src/ctsynth/` holds reusable, tested scientific code; `projects/pNN_*/` holds thin experiment drivers
(scripts that load a config, call the library, write `metrics.json`/`run_meta.json`, and have a separate `figures.py`);
`.research/` holds all research state. Matplotlib is not a project dependency (figures are post-processing on the JSON).

## Package map
| Module | Role | Torch? |
|---|---|---|
| `seeding` | one root seed → independent streams: `rng(seed, *stream)` (the only sanctioned randomness) | no |
| `config` | YAML → frozen pydantic `ExperimentConfig` (`extra=forbid`); `config_sha256` | no |
| `hashing` | `sha256_file`, dependency-free (array hashes: `paired.schema.npy_sha256`) | no |
| `runmeta` | `SOURCE_PATHS` + `dirty_source` (the single source-clean definition, D-008) and `run_meta.json` writer | no |
| `device` | `cpu` default; `mps`/`cuda` opt-in and fail loudly | optional |
| `physics/` | `MaterialBasis`, `Spectrum`, `ParallelBeamProjector2D`, `SpectralForwardModel` (ρ → line integrals → counts), autograd | yes |
| `aapm/` | AAPM DL-Spectral CT kit: `schema` (frozen conventions), `loader`, `splits`, `synthetic`, `classical`, `metrics`, `losses`, `torch_forward` (`AapmTorchForward`), `unet`, `train`, `convert` (↔ paired axes) | mixed |
| `paired/` | simulator-independent paired material/CT records (`schema` paired-ct/0.1), `mock`, analytic `phantoms` (+ `voxel_chords`), `xcist` adapter (gecatsim imported lazily; `environment_report`) | no |
| `hpc/` | `ctsynth` CLI: `preflight`, `postcheck`, `bundle-check`, `run-check`, `paired-check`, `xcist-check` — guards around the native Research OS HPC protocol | no |

Dependency direction (no cycles): `seeding`, `config`, `hashing` ← `runmeta`, `paired`, `aapm`, `hpc`; `physics` depends only on itself.
`physics` and `aapm` do **not** import each other: `physics.SpectralForwardModel` (parallel beam, ρ, `attenuation_v1`) and
`aapm.AapmTorchForward` (kit μ tables, image/path domain, no projector) are different models on purpose; do not merge them.
`paired` does not depend on `physics` or `aapm`; the drivers in `projects/` compose them.

## Drivers (`projects/`)
`p01_material_inference/exp002` (identifiability toy), `p02_aapm_benchmark` (evaluation harness, real-kit verification),
`p03_unsupervised` (B0–B4 ladder, identifiability verification), `p04_xcist` (dataset generator, mismatch ladder vs XCIST, HPC
replication). Each driver is run through `research run exec` on a committed tree. Dataset v1 generation/splitting for E1–E4 will move
into `ctsynth.paired` once T-032 fixes the split and power analysis.

## Data and conventions
Frozen versioned tables in `data/tables/` (`attenuation_v1.npz`, `spectra_v1.npz`, `provenance_v1.json`) generated offline by
`scripts/make_tables.py` (xraylib, SpekPy); runtime never imports those libraries (D-001); `tests/data/` checks hashes and NIST spot
values. `data/aapm_kit_sample/` holds the two CC-BY kit model tables; bulk AAPM data is never committed. Layouts and units:
`docs/data-contracts.md`, `docs/physics-conventions.md`; the AAPM `a[ix,iy]` vs paired `(iz,iy,ix)` difference is handled only by
`aapm.convert`.

## Tests (`uv run pytest`; CI = ruff, format, mypy, pytest)
Markers are assigned by location in `tests/conftest.py`: `unit` (default), `physics` (forward-model invariants, tests/physics),
`regression` (frozen tables, kit facts, driver outputs on tiny fixtures), `integration` (XCIST; whole file skipped without
gecatsim, run with `.venv-xcist/bin/python -m pytest tests/paired/test_xcist.py`), `hpc` (reserved, none). Passing tests show code
matches its spec; scientific claims need registered run evidence (D-008). Frozen physics tests were written before the
implementation and must not be edited by implementers.

## Limits
Dense chord matrix is O(n_view·n_det·ny·nx): toy sizes only. 2D parallel beam in `physics`; no fan-beam projector (T-021), no
scatter/detector physics beyond `physics-conventions.md`, no 3D, no anatomy/SDF code (vision only).
Decisions: `research decision list` (D-001 no third-party physics runtime dependency; D-010 native HPC; D-011 XCIST first).
