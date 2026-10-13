# CTsynth

Editable, physics-grounded virtual CT patient populations: **anatomy + material fields + pathology + optional physiology**, mapped
through a spectral CT forward model to synthetic measurements. Vision and non-goals: `docs/vision.md`. Active line of work:
material inference on XCIST paired data (`projects/p04_xcist/`, experiments E1–E4); earlier: AAPM benchmark (p02/p03), identifiability (p01).

## Quick start
```bash
uv sync                                  # core env from uv.lock (torch via the dev group)
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
.research/bin/research current           # Research OS state (no install needed)
```
Optional XCIST env: `docs/xcist.md`. Cluster runs: `docs/hpc.md`.

## Where things are
| Path | Purpose |
|---|---|
| `src/ctsynth/` | tested library: physics, aapm, paired, provenance and HPC guards (`docs/architecture.md`) |
| `projects/pNN_*/` | experiment drivers (thin scripts) and their figures |
| `configs/` | one YAML per registered experiment id |
| `tests/` | `unit`, `physics`, `regression`, `integration` markers |
| `data/` | frozen attenuation/spectrum tables and kit model tables (bulk data is never committed) |
| `docs/` | `architecture`, `physics-conventions`, `data-contracts`, `hpc`, `xcist`, `research-os`, `vision`, specs, `literature/` |
| `.research/` | Research OS state (questions, experiments, runs, findings, decisions, hooks, skills): use the CLI, do not hand-edit |
| `AGENTS.md`, `CLAUDE.md` | agent rules (the `AGENTS.md` Research OS block is managed) |

## Working rules in one paragraph
Research state lives in Research OS, not in Markdown. Runs are registered and made from a committed tree; numbers enter findings through
`research evidence bind`; a finding is `supported` only after independent review (D-008). Laptop tests never validate HPC/GPU behaviour.
Start with `docs/research-os.md`.
