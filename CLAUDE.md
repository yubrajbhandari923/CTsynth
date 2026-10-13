# CTsynth — agent rules

Read `AGENTS.md` (Research OS commands) first; start with `.research/bin/research current`.
Scope, physics and contracts: `docs/`. Research state lives in Research OS, not in chat or Markdown history.
Before ANY research task or dependent experiment: `research current`, then `research hpc sync` (imports finished
jobs), then inspect what completed since the last checkpoint (`research status`, `research experiment show`,
`research task show`) and synthesize/review it first. Never launch an experiment that depends on a result you have
not read from its run record. Active work: `projects/p04_xcist/` (XCIST pairs); earlier: `projects/p01_material_inference/`.
`research task next` picks the lowest ready ID, not the most important task: the next scientific action is the latest
checkpoint's Next step. Re-preview bundle contents (`hpc.bundle.exclude`) before EVERY `hpc submit` (F-038).
XCIST/paired data: `docs/xcist.md`; planned experiments: `docs/xcist-experiments.md` (registered as EXP-013..016).

## Roles (Research OS 1.0.0, strategy `sonnet_first`; `research strategy show`)
Levels map to models: deep = Opus, balanced = Sonnet, fast = Haiku. Routing is by task kind, not label (`research strategy route T-…`).
- **Sonnet (balanced)** is the coordinator: planning, verification, unchecked work, technical specs (interfaces, data contracts),
  *substantive test design* (physics invariants, tolerances with rationale), and independent review. Tests are frozen before Haiku implements.
- **Haiku (fast)** implements checkable/mechanical tasks against frozen tests / a stated objective (retries: 3, then escalates to Sonnet).
  Never edits frozen tests or specs.
- **Opus (deep)** only for scientific interpretation, findings, synthesis, `docs/scientific-decisions.md`, and scientific-assumption calls.
- Humans (Yubraj) decide what to investigate; discuss science with GPT Astra. Laptop is authoritative; HPC runs experiments only.
- Delegated task protocol: `research task context T-…` (effective spec: files, criteria, checks, hooks), then
  `research orchestrate verify T-… --tier LEVEL --record` before handing back. Project hooks (`research hook list`) run only after human approval.
- MCP covers the core loop only; use the CLI for orchestrate, hooks, strategy, tuning and most `hpc` commands.
- Tuning: `research tuning declare|trial|select|test` (no longer under `orchestrate`); held-out test is evaluated once.

## Tooling and delegation
Verified available (2026-10-11): Ponytail 5.1.0 (enabled, `full` active; `/ponytail lite|full|ultra`,
skills `ponytail:ponytail-review|-audit|-debt`), pyright-lsp, research-desk (reference-lookup MCP:
`verify_reference`, `lookup_reference`, `search_works`), parallel (web_search/web_fetch MCP), zeroguess,
Google Drive/Gmail/Calendar + Claude Docs MCP. **No project or global `skills` CLI entries, no `.claude/`
dir in this repo** — all skills come from plugins. Research OS CLI: `.research/bin/research` (no install).

Use before working, not for show: `reference-lookup` to verify any citation before it enters `docs/literature/`;
`parallel` web_search for literature/dataset facts; `ponytail:ponytail-review` on a substantial diff.

Model routing follows the active strategy (above); one job per agent, explicit owned files. Let Haiku loop implement → test → fix
on its own; escalate only when a failure is architectural or scientific, never for routine debugging.
Avoid duplicate context loading, redundant subagents and repeated literature searches: check `docs/literature/`
and `research search` before dispatching a new search agent. Prefer the cheapest mode that does the job.
Record decisions and evidence in Research OS; do not log routine implementation iterations.

## Coordination
- One **coordinator** (the main session) integrates shared files: `pyproject.toml`, `uv.lock`, `src/ctsynth/__init__.py`, `docs/architecture.md`, `docs/data-contracts.md`, `.research/`.
- Every delegated task states **owned files** in its brief; touch nothing else. Parallel agents never share a file.
- Claim work with `research task start`; record notes/results in Research OS, not chat.

## Pre-registration rules (learned from EXP-002, D-003)
Before an experiment is registered, its success criteria must:
- **Score every material/output**, not a single favoured one. An iodine-only criterion hid a large water bias.
- **State the detection floor** from a power analysis *before* running. A "no effect" result is meaningless unless the
  experiment could have detected the effect. Prefer a deterministic noise-free calculation as the primary quantity and
  use Monte-Carlo only to establish detectability.
- **Not require significance at every condition** (e.g. both dose levels) unless that is the intended hypothesis;
  the noisiest condition otherwise dictates the verdict.
- Distinguish **structural** results (forced by arithmetic, e.g. rank <= min(n_meas, k)) from **empirical** ones.
  Never report arithmetic as evidence.
- Compare like with like: changing two factors at once (design *and* basis size) confounds them.

## Promotion gate (D-L) — a finding may be marked `supported` only with all three
1. **Evidence matching the claim type**: an empirical claim needs `research evidence bind` + `verify`
   against a registered run; a mathematical claim needs an independently checked derivation, recorded as
   such (a deterministic verification run counts and is the preferred form — see EXP-005/RUN-0007).
2. **Independent review** by an agent that did not author it, with authority to reject. A superseding
   finding is itself authored and needs its own review.
3. **Explicit scope**: synthetic vs real data, which split, which regime, and the tolerance any
   threshold-dependent number was computed at.
Run provenance: `git_dirty_source` (not `git_dirty`) is the citability signal; experiment outputs live
outside tracked source (`projects/*/results*/` is gitignored) so a run never dirties its own source tree.
Never put an experiment number in an experiment title — create the record first and use the id it returns (D-007).

## Evidence rules
- Passing tests show code matches its spec, not that a scientific claim is true. Conclusions need run evidence: `research evidence bind` numbers to RUN/metric records.
- Every finding links an experiment ID **and** a git commit (`git_commit` in `run_meta.json`; dirty tree = not citable).
- Findings stay *preliminary* until reviewed by someone other than the author. The reviewer's job is to BREAK the
  claim, not confirm it; a review that finds nothing must say what it tried. Superseding findings are themselves
  authored, so they need their own independent review.
- A self-consistency check (estimator agreeing with its own Fisher information) is **not** validation of physical
  fidelity: a wrong table or spectrum passes it equally. Say which one a check actually is.
- State the **units and the scope** of every number: line integral (g/cm²) vs density (g/cm³), which material,
  which rays, which cell, median vs universal.
- Literature/implementation claims must be verifiable: cite DOI/arXiv/Zenodo ID and mark each as read-in-full / abstract-only / unverified (`docs/literature/`). Never invent a citation, CLI command, or number.
- Laptop tests never establish GPU/HPC validation. HPC results enter only as transferred outputs + `run_meta.json` (see `docs/hpc-workflow.md`); no code is developed on HPC.
- Never tune on held-out data; never loosen a tolerance to pass.

## Code
`uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` must pass. Randomness only via `ctsynth.seeding.rng`. CPU-first; MPS/CUDA opt-in via config `device`. Units and conventions: `docs/physics-conventions.md`. Keep the package small; use Research OS for experiment tracking rather than building one.
