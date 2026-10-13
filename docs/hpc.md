# HPC workflow (Duke SLURM via Research OS)

HPC is a **run-only** environment, never a second development environment: develop and test on the laptop, commit, run on the
cluster, bring results back. Research OS owns orchestration (`research hpc doctor|bundle|submit|status|sync|import|datasets`);
CTsynth adds only the scientific guards below. History: the earlier CTsynth bundle/verify/import CLI was removed once the native
protocol worked on Duke (EXP-007/RUN-0012; decisions D-009 superseded by D-010). Cluster profile (paths, limits, `slurm.setup`)
lives in the machine-layer Research OS config, not in this repo.

## Sequence for a citable cluster run
1. Commit. "Source" is `ctsynth.runmeta.SOURCE_PATHS` (src, tests, configs, scripts, projects, pyproject.toml, uv.lock); the
   same definition is used by run metadata and by preflight, so local and HPC decisions agree. Outputs under
   `projects/**/results*` are not source.
2. `uv run ctsynth preflight --repo . [--config configs/X.yaml]` — refuses a dirty or untracked source tree; prints the
   fingerprints (`git_commit`, `uv_lock_sha256`, and with `--config` `experiment_id`, `config_sha256`) to record as experiment params.
3. `uv run ctsynth bundle-check` — what `research hpc bundle` would package. Bundles are the working tree filtered by
   `hpc.bundle.exclude` in `.research/config.yaml` (**`.gitignore` does not apply**, F-038); the check fails on environments, data,
   result directories or an oversize bundle. `hpc bundle` itself has no dry-run and always creates a RUN.
4. `research hpc submit EXP-… --cluster duke --dry-run -- <cmd>`; show the CPUs/GPUs/memory/wall-time to the owner; re-run with
   `--yes` only after an explicit yes. Use `python3` for stdlib job scripts (cluster interpreter is 3.8); core CTsynth needs
   Python >= 3.11 via uv.
5. `research hpc sync` (idempotent; never resubmit an uncertain submission). Nothing is interpreted until someone runs sync.
6. `uv run ctsynth postcheck .research/runs/RUN-… --expect fp.json --require metrics.json` — identity (commit, config, lock),
   citability (`git_dirty_source` must be exactly `false`) and required outputs. The job must write `run_meta.json`
   (`ctsynth.runmeta.write`) into its registered outputs.
7. `research experiment check EXP-…`, then `research evidence bind` / `verify`, synthesis, independent review.

Post-experiment steps do **not** run automatically when a SLURM job finishes; steps 5-7 are manual. Native `checksums.json`
travels inside the result archive, so it detects accidental corruption, not a forger who rewrites both.

## Guards (`uv run ctsynth …`)
`preflight`, `postcheck`, `bundle-check`, `run-check` (registered `run_meta.json` files of an experiment are citable),
`paired-check DIR…` (paired-ct records load, every sha256 verifies), `xcist-check` (optional XCIST stack usable). Exit 0 = ok,
1 = failed check (reasons on stderr), 2 = usage. Registered Research OS hooks that call them: `docs/research-os.md`.

## XCIST on the cluster
`scripts/hpc_setup_xcist_env.sh` builds an isolated Python 3.11 + gecatsim env; `scripts/hpc_xcist_replicate.sh` runs the
replication. Both read the env location from `scripts/xcist_env_path.sh`: `CTSYNTH_XCIST_ENV`, default
`$HOME/research-os/projects/CTsynth/env`. Details and quirks: `docs/xcist.md`.
`scripts/hpc_smoke.py` and `scripts/hpc_env_probe.py` are frozen as run (RUN-0012/0013): their hashes are recorded, do not edit them.

## Datasets
Datasets live on the cluster; the filesystem is the registry (`research hpc datasets list|check`). Never re-download something
that verifies and never adopt an undescribed directory without the owner. The AAPM DL-Spectral CT acquisition plan and its
Zenodo md5s are in `docs/aapm/hpc-acquisition.md` (not executed).

Laptop tests never establish GPU/HPC behaviour; HPC-specific claims need an HPC run record.
