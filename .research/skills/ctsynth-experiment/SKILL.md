---
name: ctsynth-experiment
description: Use before registering or running any CTsynth experiment (laptop or Duke SLURM): pre-registration rules, clean-tree runs, guards, validation sequence, evidence binding.
---

# ctsynth-experiment

CTsynth specifics on top of the generic `run-experiment`, `synthesize-experiment` and `hpc-setup` skills (read those for the
commands). Rules of record: `CLAUDE.md` (pre-registration, promotion gate), D-003, D-007, D-008.

**Before registering**: success criteria score every material/output, state the detection floor from a power analysis, are
not tied to significance at every condition, separate structural from empirical results, change one factor at a time. Never put
an experiment number in a title (create first, use the returned id). Name the config after the issued id.

**Sequence (runs must be citable: commit first)**
1. `research current`, `research hpc sync`, read what finished (CLAUDE.md rule).
2. Commit. The source tree is `runmeta.SOURCE_PATHS` (src, tests, configs, scripts, projects, pyproject.toml, uv.lock);
   outputs under `projects/**/results*` are not source. `uv run ctsynth preflight --repo . [--config configs/X.yaml]` refuses a dirty tree.
3. Cluster only: `uv run ctsynth bundle-check` (F-038), then `research hpc submit EXP-… --dry-run`, show CPUs/GPUs/mem/time to the human,
   `--yes` only after an explicit yes. Never resubmit an uncertain submission: `research hpc sync`.
4. Run with `research run exec EXP-… -- <cmd>`. Drivers write `run_meta.json` via `ctsynth.runmeta.write` and log metrics with
   `research.log_metric`; outputs stay outside tracked source.
5. After `hpc sync`: `ctsynth postcheck .research/runs/RUN-… --expect fp.json --require metrics.json`, then `research experiment check EXP-…`.
6. Interpret only after steps 1–5 pass: `research evidence bind` + `verify`, synthesis, independent review.

**Hooks** (registered in `.research/config.yaml`, run only after human approval): `source-clean`, `bundle-content` fire when a run is
created; `run-meta-citable` fires in `experiment check`; `lint-types`, `tests-fast` are run by hand with `research hook run NAME`.
Post-experiment steps do NOT run automatically when a SLURM job ends; `hpc sync` and `experiment check` are manual.

**Never**: tune on held-out data, loosen a tolerance to pass, call engineering tests scientific evidence, launch experiments from a refactor.
