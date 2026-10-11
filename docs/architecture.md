# Architecture

Layers (only the first is code today):
`ctsynth` core utilities → (planned) `phantom` (material fields) → `physics` (spectra, attenuation, projector) → `inference` (P01) → `eval`.

## Present
| Module | Role |
|---|---|
| `seeding.py` | one root seed → independent streams via `SeedSequence` (`rng(seed, *stream)`) |
| `config.py` | YAML → frozen pydantic `ExperimentConfig` (`extra=forbid`); `experiment_id` = Research OS EXP id |
| `runmeta.py` | `run_meta.json`: config hash, git commit/dirty, versions, UTC time — works without Research OS (HPC) |
| `device.py` | `cpu` default; `mps`/`cuda` opt-in, fail loudly if unavailable |

## Decisions
- **Research OS owns experiment tracking** (experiments, runs, metrics, findings, checkpoints, task coordination). We do not build a tracker. `runmeta.py` only guarantees a self-describing record where `research run exec` cannot run (HPC).
- **pydantic + YAML, no Hydra** — enough for flat/nested validated configs; revisit if sweeps/composition are needed.
- **uv + `uv.lock`** for reproducible environments; torch is an optional extra so CI/laptop stay light.
- **src layout**, `mypy --strict` on `src/`, ruff lint+format.
- Package stays dependency-light; `numpy` is the array contract at module boundaries (torch internal only).
