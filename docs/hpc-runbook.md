# HPC runbook (Research OS native protocol)

Research OS 0.5.0 owns orchestration: `research hpc bundle|submit|status|sync|import|datasets`.
CTsynth adds two guards (`src/ctsynth/hpc/check.py`). History: the earlier CTsynth bundle/verify/import
code (RUN-0010) was removed once these guards and the native protocol were exercised on a real
cluster (EXP-007/RUN-0012, SLURM job 201820); see decision in Research OS.

1. `ctsynth preflight --repo . --config configs/X.yaml > fp.json` — refuses dirty/untracked
   src, tests, configs, scripts, projects, pyproject.toml, uv.lock. Record `git_commit`,
   `config_sha256`, `uv_lock_sha256` as experiment params.
2. `research hpc submit EXP-… --dry-run -- <cmd>` → show reservation to the owner → `--yes`.
   Use `python3` (cluster interpreter is 3.8); core CTsynth needs >=3.11 via uv (see EXP-008).
3. `research hpc sync` (idempotent; never resubmit an uncertain submission).
4. `ctsynth postcheck .research/runs/RUN-… --expect fp.json --require metrics.json`.
   The job must write `run_meta.json` (`ctsynth.runmeta.write`) into its registered outputs.

Bundles package the working tree, filtered by `hpc.bundle.exclude` in `.research/config.yaml`
(`.gitignore` does NOT apply). Check with the preview snippet in the Phase 1 notes before the first
bundle of a new tree. Native `checksums.json` travels inside the result archive, so it detects
accidental corruption, not a forger who rewrites both; compare `postcheck` output hashes against
values recorded elsewhere when that matters. `hpc bundle` has no dry-run and always creates a RUN.
