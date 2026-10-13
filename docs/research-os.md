# Working with Research OS in CTsynth

Research OS (1.0.0, `.research/bin/research`, no install) is the system of record for questions, experiments, runs, findings,
decisions, plans, tasks and checkpoints. This file says only what is CTsynth-specific; generic procedure lives in the Research OS
skills (`research skill list`) and `AGENTS.md`. Do not keep a second research ledger in Markdown.

## Session start
`research current` → `research hpc sync` → read what finished (`research status`, `experiment show`, `task show`) → `research task context T-…`.
The next scientific action is the latest checkpoint's *Next step*; `research task next` only returns the lowest ready ID.

## Model allocation
Active strategy `sonnet_first` (`research strategy show`): balanced model (Sonnet) coordinates and implements, fast (Haiku) takes
checkable mechanical work, deep (Opus) is for scientific synthesis. Orchestration execution (`orchestration.enabled`) is **off**;
routing is advisory (`research strategy route T-…`). Turning it on is a human decision (contracts, frozen tests,
`orchestrate verify`).

## Hooks (`research hook list`)
Registered in `.research/config.yaml`; **a hook runs only after a human approves its exact command** (`research hook approve NAME`).
Required hooks that are enabled but unapproved *block* what they gate, so the required ones ship disabled:
| Hook | Stage | Fires | Command | State |
|---|---|---|---|---|
| `source-clean` | pre_experiment | when `research run exec` creates a run | `ctsynth preflight --repo .` | required, disabled |
| `bundle-content` | pre_experiment | same | `ctsynth bundle-check --repo .` | required, disabled |
| `lint-types` | post_implementation | `research hook run lint-types` (by hand) | ruff check, ruff format --check, mypy | required, disabled |
| `tests-fast` | post_implementation | by hand | `pytest -m "not integration and not hpc"` | required, disabled |
| `run-meta-citable` | post_experiment | in `research experiment check` | `ctsynth run-check` | advisory, enabled, unapproved |
To adopt: `research hook enable NAME && research hook approve NAME`. `post_implementation` only fires automatically inside
`orchestrate verify`; `preflight` only on orchestrate dispatch (none registered). Expensive integration tests (XCIST) and scientific
evaluation are deliberately not hooks.

## Skills (scoped, not always-on)
`ctsynth-xcist` (XCIST/paired data), `ctsynth-experiment` (CTsynth rules on top of `run-experiment`/`hpc-setup`),
`ctsynth-conventions` (axes, units, tables). `research skill show NAME`.

## Experiment lifecycle
question → pre-registered experiment (criteria in `CLAUDE.md`) → implementation + tests (`uv run pytest`) → commit → guards
(`docs/hpc.md`) → `research run exec` / `hpc submit` → `hpc sync` → `experiment check` → `evidence bind` + `verify` → synthesis →
independent review → Scientist Guide refresh (`research guide status`) → checkpoint. Drivers report with `research.log_metric` /
`research.register_artifact` and `ctsynth.runmeta.write`; they never reimplement run management. Operational task completion is not a
supported finding (D-008): engineering tests show code matches its spec, not that a scientific claim holds.
