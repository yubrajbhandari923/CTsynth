# CTsynth — agent rules

Read `AGENTS.md` (Research OS commands) first; start with `.research/bin/research current`.
Scope, physics and contracts: `docs/`. Active project: `projects/p01_material_inference/`.

## Roles
- **Opus** owns scientific interpretation, findings, synthesis, `docs/scientific-decisions.md`.
- **Sonnet** owns technical specs (interfaces, data contracts) and *substantive test design* (physics invariants, tolerances with rationale). Tests are frozen before Haiku implements.
- **Haiku** implements bounded tasks and iterates against frozen tests / a stated objective. Never edits frozen tests or specs; escalates after repeated failure.
- Humans (Yubraj) decide what to investigate; discuss science with GPT Astra. Laptop is authoritative; HPC runs experiments only.

## Coordination
- One **coordinator** (the main session) integrates shared files: `pyproject.toml`, `uv.lock`, `src/ctsynth/__init__.py`, `docs/architecture.md`, `docs/data-contracts.md`, `.research/`.
- Every delegated task states **owned files** in its brief; touch nothing else. Parallel agents never share a file.
- Claim work with `research task start`; record notes/results in Research OS, not chat.

## Evidence rules
- Passing tests show code matches its spec, not that a scientific claim is true. Conclusions need run evidence: `research evidence bind` numbers to RUN/metric records.
- Every finding links an experiment ID **and** a git commit (`git_commit` in `run_meta.json`; dirty tree = not citable).
- Findings stay *preliminary* until reviewed by someone other than the author.
- Literature/implementation claims must be verifiable: cite DOI/arXiv/Zenodo ID and mark each as read-in-full / abstract-only / unverified (`docs/literature/`). Never invent a citation, CLI command, or number.
- Laptop tests never establish GPU/HPC validation. HPC results enter only as transferred outputs + `run_meta.json` (see `docs/hpc-workflow.md`); no code is developed on HPC.
- Never tune on held-out data; never loosen a tolerance to pass.

## Code
`uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` must pass. Randomness only via `ctsynth.seeding.rng`. CPU-first; MPS/CUDA opt-in via config `device`. Units and conventions: `docs/physics-conventions.md`. Keep the package small; use Research OS for experiment tracking rather than building one.
