# HPC workflow

HPC is a **run-only** environment, never a second development environment.

1. Develop and test on the laptop; commit. Record the commit hash.
2. Transfer a clean checkout at that commit (`git archive`/clone of a tag) and `uv sync --locked`. **No edits on HPC**; a dirty tree makes results non-citable (`git_dirty` in `run_meta.json`).
3. Run via a config file from `configs/`; the run writes `run_meta.json` + outputs to its own directory.
4. Transfer the output directory back. On the laptop: register with Research OS (`research sync`, `research artifact`, `research run`/`metric`), bind numbers to findings (`research evidence bind`).
5. Findings from HPC runs cite experiment ID + commit + `run_meta.json` hash. Large outputs stay on disk; register paths only.

Laptop tests do not validate GPU/HPC behaviour; HPC-specific claims require an HPC run record. SLURM setup lines go in `.research/config.yaml` (`backends.slurm.setup`); none configured yet (cluster details unknown — open question).
