# CTsynth

Editable, physics-grounded virtual CT patient populations. A patient is represented as
**anatomy + material fields + pathology + optional physiology**; CT acquisition/reconstruction then
produces synthetic measurements and images. See `docs/vision.md`.

Active subproject: **P01 material inference from dual-energy / spectral CT** (`projects/p01_material_inference/`).

## Reproduce the smoke test
```bash
uv sync                      # creates .venv from uv.lock (CPU only; `uv sync --extra torch` for torch)
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
.research/bin/research current   # Research OS state (no install needed)
```

## Map
| Path | Purpose |
|---|---|
| `src/ctsynth/` | package: seeding, config, run metadata, device |
| `configs/` | YAML experiment configs (validated by `ctsynth.config`) |
| `docs/` | vision, architecture, decisions, conventions, contracts, HPC, `literature/` |
| `projects/p01_material_inference/` | hypotheses, tasks, project notes |
| `.research/` | Research OS memory (questions, experiments, runs, findings) — do not hand-edit |
| `CLAUDE.md`, `AGENTS.md` | agent rules (AGENTS.md Research OS block is auto-managed) |
