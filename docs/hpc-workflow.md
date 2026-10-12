# HPC workflow

HPC is a **run-only** environment, never a second development environment.

1. Develop and test on the laptop; commit. Record the commit hash.
2. `ctsynth preflight` (clean source), then `research hpc submit` — Research OS bundles the working tree (see `docs/hpc-runbook.md`). **No edits on HPC**; a dirty source tree makes results non-citable (`git_dirty_source` in `run_meta.json`).
3. Run via a config file from `configs/`; the run writes `run_meta.json` + outputs to its own directory.
4. `research hpc sync` imports results and registers metrics/artifacts; `ctsynth postcheck` verifies identity and required outputs. Bind numbers to findings (`research evidence bind`).
5. Findings from HPC runs cite experiment ID + commit + `run_meta.json` hash. Large outputs stay on disk; register paths only.

Laptop tests do not validate GPU/HPC behaviour; HPC-specific claims require an HPC run record. Cluster profile (paths, limits, `slurm.setup`) lives in the machine-layer Research OS config; EXP-007/008 verified a CPU smoke job on Duke.
