# Architecture and workflow audit (Phase 2)

Date 2026-10-13 · Research OS 1.0.0, schema 3, strategy `sonnet_first` · base commit `952a99d` · author: Sonnet (coordinator).
Audit only: no code, record, dataset, HPC or hook was changed. Research state is in Research OS (PLAN-004, T-043..T-060, the
Phase 2 note and checkpoint); this file holds only the inventory, diagrams and tables that do not fit there.
Labels: **[V]** verified by command/read in this session · **[S]** suspicion, not checked · **[P]** preference.

## 1. Executive assessment
The package core is small (29 modules, 3.3k lines `src/`), typed (mypy strict passes), lint-clean, and its layering is sound:
`physics` and `paired` do not import `aapm`; `aapm` does not import `physics` **[V, import graph]**. The mess is not in `src/`;
it is in four places: (a) **provenance guards disagree with each other**, (b) **orchestration is configured but switched off**,
(c) **project scripts carry the duplicated boilerplate and the untested science**, (d) **docs/records contain stale or contradictory
statements**. Tests pass (ruff, mypy, pytest all green **[V]**), which says the code matches its specs, not that anything is scientifically valid.

## 2. Inventory and architecture map (all [V] from import graph + reads)

| Area | Files | Status | Used by |
|---|---|---|---|
| `ctsynth` core: `seeding, config, device, hashing, runmeta` | 5 | live | everything |
| `ctsynth.physics` (parallel-beam torch forward model; D-001) | 4 | live, **dense chord matrix, toy sizes only** | `p01/exp002`, `p04/compare` |
| `ctsynth.aapm` (AAPM schema, classical, U-Net, losses, own torch forward) | 10 | live, image-domain only | `p02`, `p03` |
| `ctsynth.paired` (paired-ct/0.1 schema, mock, phantoms, XCIST wrapper) | 4 | live, the target data contract | `p04` |
| `ctsynth.hpc` (`preflight`/`postcheck` CLI `ctsynth`) | 3 | live | manual, by docs only |
| `projects/p01..p04` (experiment scripts, figures) | 17 | mixed: run records exist for p01/p03/p04 | tests (p01,p02,p03 only) |
| `scripts/` | 7 | 2 frozen-by-hash (`hpc_smoke`, `hpc_env_probe`), 3 XCIST-HPC, `make_tables`, `fetch_aapm` | HPC / offline |
| `tests/` | ~40 files (physics, aapm, paired, hpc, data, p01–p03 drivers) | live | CI |
| `docs/` | 25 md | see §5 | — |

Dependency graph (arrows = imports; no cycles found **[V]**):

```
seeding   device   config ◄── hashing ◄── runmeta
   ▲                  ▲          ▲  ▲
   │                  └── hpc/check ┘  └── paired/schema
physics/{materials ◄ spectrum, projector ◄ forward}        paired/{schema ◄ mock, xcist; phantoms}
   ▲                                                          ▲
projects/p01/exp002          aapm/{schema ◄ classical, loader, metrics, splits, synthetic, torch_forward ◄ losses ◄ train; unet}
projects/p04/{generate, compare ──► physics + paired}              ▲
                                                projects/p02/evaluate, p03/run_exp004
```
Two independent forward models coexist: `physics.SpectralForwardModel` (parallel-beam, ρ-based, κ(E) table `attenuation_v1`) and
`aapm.AapmTorchForward` (kit tables, image- and path-domain, no projector). This is **legitimate** (different tables, geometry, inputs), not duplication:
do not merge. The cost is two attenuation/spectrum sources of truth with no cross-check test **[V: no test imports both]**.

Planned/absent **[V]**: fan-beam projector (T-021), 3D anything, XCAT/SDF/organ-label code (none exists; `docs/vision.md` only), DukeSim (spec only),
material inversion on XCIST data (E1–E4 registered, 0 runs, no code), dataset v1 and split file (T-032).

## 3. Findings

### P0 — correctness / provenance
**A1 [V] Three different "is the tree clean?" definitions.** `runmeta.py:51` checks `src tests configs`; `hpc/check.py` preflight refuses dirty
`src tests configs scripts projects pyproject.toml uv.lock` (docs/hpc-runbook.md). Experiment scripts live in `projects/`, so a local run from a dirty
`projects/` tree records `git_dirty_source=false`, which D-008 treats as the citability signal. Fix: one `source_paths` tuple in one module, used by both, with a test that touches a file in each path.
**A2 [V] Orchestration is off.** `orchestration.enabled: false` (effective config); the `sonnet_first` strategy I set earlier routes (`strategy route T-032` works) but nothing is contracted, frozen or verified. Hooks: none. No task has a check, contract or parent. Usage: none recorded. So today none of the 1.0 delegation guarantees apply.
**A3 [V] Experiment code is largely untested and some results were produced outside `research run exec`.** No tests for `p04/generate.py, compare.py, hpc_replicate.py, official_example.py, p02 verify_real_kit, p03 verify_identifiability`. EXP-004 was run by Haiku outside `run exec` (D-007). F-037/F-036 rest on p04 scripts with no test. Not wrong, but not protected.
**A4 [V] Hard-coded cluster paths in committed scripts**: `/home/yb107/research-os/projects/CTsynth/env` in `scripts/hpc_setup_xcist_env.sh:5` and `hpc_xcist_replicate.sh:7`. Also `module load gcc/11.2.0`. Works; breaks silently for any other user/cluster. Preserve the behaviour (RUN-0022 hash-bound) but read the path from `$RESEARCH_PROJECT_DIR`-style env or one config line in a *new* script version.
**A5 [V] Committed outputs contradict the documented rule.** 20 files under `projects/p01_material_inference/exp002/results/` are tracked although `.gitignore` says `projects/*/results*/` is never tracked and D-008 says outputs live outside source. They are the cited evidence for EXP-002 (artifact hashes), so **do not delete**; instead mark as an explicit curated exception and exclude from the dirty check.
**A6 [V] Three AAPM-vs-paired axis conventions**: aapm `image[ix, iy]` (`aapm/schema.py:43`) vs paired `(iz, iy, ix)`; documented in data-contracts.md but no converter and no test that fails if someone mixes them. Silent transpose is the classic CT bug. Add a one-direction converter with a round-trip test before any E-experiment consumes both.
**A7 [S] Physics cross-check missing**: nothing compares `attenuation_v1` (physics) with the AAPM kit mu tables for a shared material/energy. F-037 shows spectrum spec is the largest mismatch (2.4–3.2%), so a unit-level test of agreement belongs in the suite. Needs one run to decide tolerance (human/Opus call).

### P1 — structure and duplication
| # | Where | What | Same behaviour? | Canonical / action |
|---|---|---|---|---|
| D1 | `hashing.sha256_file`, `p02/evaluate.py:223 sha256_file`, `p03/run_exp004.py:171 _sha256`, inline `hashlib` in `verify_real_kit.py:74`, `verify_identifiability.py:63`, `hpc_replicate.py:29`, `official_example.py:49`, `compare_replication.py:20`, `make_tables.py:63` | whole-file sha256 reimplemented 9× | `sha256_file` streams; most others `read_bytes` (identical digest) | use `ctsynth.hashing.sha256_file`. **Keep** `hpc_smoke.py`/`env_probe.py` (hash-frozen) and `paired.schema.npy_sha256` (hashes array content, different semantics) and `make_tables` only if stdlib-only is required (it is offline; leave) |
| D2 | `hashing` imports `config.ExperimentConfig` | leaf utility depends on a model | n/a | move `config_sha256` to `config.py`; `hashing` becomes dependency-free |
| D3 | `p04/compare_replication.py` vs `hpc_replicate.py` | split by laptop/HPC by design | different roles | **keep**; both lack tests |
| D4 | `p0x/figures.py` ×4 | per-project figure scripts, matplotlib not a dependency | similar plumbing, different content | do not merge; share a 20-line `_savefig` only if a 5th appears **[P]** |
| D5 | `p01 exp002/estimator.py` vs `aapm/classical.py` | two estimators | **different problems** (identifiability toy vs kit-constrained NNLS) | keep both; do not merge |
| D6 | `ctsynth.hpc` (`ctsynth preflight|postcheck`) vs Research OS `hpc submit/sync`, `run check`, hooks | CTsynth guard on top of native protocol (D-010) | partial overlap only in "required outputs" | keep, expose as **hooks** (preflight / post_experiment), not new CLI |
| D7 | `runmeta.py` | "covers HPC where `run exec` cannot" | superseded in part: native HPC runs now register runs/provenance | decide: keep for config hash + dirty-source; drop software/platform fields Research OS already records **[S, verify `research run show`]** |
| D8 | three `.venv*` / bundle dirs, `.mypy_cache` etc. | not source | — | T-042 is a human decision; do not auto-clean |

Dead code **[V]** (no non-test caller): `seeding.seed_everything` (0 refs anywhere) — candidate for removal; `device.pick_device` (tests only), `aapm.synthetic.write_synthetic_dir`, `aapm.splits.case_split`, `aapm.schema.{kvp_of_view, deinterlace, validate_dataset}`, `paired.mock.make_mock_record` — these are tested API for the not-yet-run real-data path, **keep** until T-017/T-022 are decided. Only `seed_everything` is a deletion candidate, and it is rule-adjacent ("randomness only via `ctsynth.seeding.rng`"), so deleting it is a human nod.

**Config:** `configs/*.yaml` (4) feed `ExperimentConfig` for p01/p02/p03 only; p04 takes values from module constants (`generate.py: BASE, DX, XN, N`), i.e. dataset parameters live in code, not a config → unreproducible-by-config. Add `configs/p04_*.yaml` when dataset v1 is generated (T-032), not before.
**Entry points:** one console script `ctsynth`; everything else is `python projects/...` with `PYTHONPATH` set by hand (`hpc_xcist_replicate.sh`) and `# type: ignore[import-not-found]` for `from generate import ...` in `hpc_replicate.py`. `projects/` is not a package and not under mypy. **Acceptable** while experiments are scripts; becomes a problem at E1–E4 scale. Proposed fix: a small `ctsynth.xcist_data` module (dataset generation + split + loader) in `src/` and keep `projects/p04` as thin drivers.

### P1 — Documentation audit
| Doc | Verdict |
|---|---|
| `AGENTS.md` | authoritative (managed block), keep untouched |
| `CLAUDE.md` | authoritative project rules; reduce after Phase 3 (duplicates AGENTS.md + D-003/D-008 text) **[P]** |
| `docs/physics-conventions.md`, `data-contracts.md`, `physics/forward-model-spec.md`, `aapm/adapter-spec.md`, `aapm/unsupervised-spec.md`, `aapm/classical-baseline.md`, `xcist.md` | keep (scientific/technical specs; code-adjacent) |
| `docs/scientific-decisions.md` | **stale**: assumption A3 "XCAT/DukeSim not available…", open question 1 "download AAPM kit?" (done), decisions D-A..D-D duplicate Research OS D-0xx under a *different numbering*. Replace by a 10-line pointer to `research decision list` + the still-true non-goals; archive the rest |
| `docs/architecture.md` | mostly accurate; its "Decisions" list duplicates D-001/D-002; "planned phantom" layer is now `paired`. Rewrite once, from this audit's §2 |
| `docs/hpc-workflow.md` + `docs/hpc-runbook.md` | overlapping (11+21 lines); merge into one |
| `docs/aapm/hpc-handoff.md`, `hpc-acquisition.md` | **obsolete** (describe unimplemented "Prompt 4" bundle CLI; D-009 `ctsynth prepare|verify|export|import` was removed in `6a98348`, yet D-009 is still `active` and several docs/skills still reference it — `grep` hits in `evaluate.py`, `run_exp004.py`, `run_exp002.py`). Supersede D-009 by D-010; archive hand-off docs |
| `docs/dukesim-request.md`, `xcist-experiments.md` | `xcist-experiments.md` duplicates EXP-013..016 text (should link, not copy); `dukesim-request` is a draft request to a third party: keep as is |
| `docs/vision.md`, `README.md` | README "Active subproject P01" and Map are stale (p04 is active; lists `projects/p01` as hypotheses/tasks) |
| `docs/SCIENTIST_GUIDE.md` | Research OS-managed narrative; keep |
| `docs/literature/` | keep; verify-before-cite rule already stated |
Target set (≈9): README, AGENTS.md, CLAUDE.md, architecture.md, physics-conventions.md, data-contracts.md, one `hpc.md`, `xcist.md`, `SCIENTIST_GUIDE.md` + literature/ and component specs.

## 4. Scientific-software and reproducibility
Strong **[V]**: frozen versioned tables with provenance and NIST spot tests; `rng(seed, *stream)` only; schema sha256 verified on load; split unit = phantom; mypy-strict core; reference-implementation tests (`tests/*/…_reference.py`) separate from the code under test.
Gaps:
- **Test taxonomy is implicit.** Everything is a unit/regression test in one `pytest` run. Proposed markers: `unit` (default, <1 s), `physics` (invariants: adjoint, energy conservation, monotonicity, unit round-trips), `regression` (pinned numbers with tolerance and rationale), `integration` (needs `.venv-xcist`/gecatsim; skipped otherwise), `hpc` (never in CI). Scientific evidence stays in Research OS runs, never in pytest.
- **No test for XCIST wrapper behaviour beyond mock-level** (`tests/paired/test_xcist.py` runs without gecatsim **[S]**: verify). An opt-in smoke test that checks geometry (affine, rays, orientation) against a known disk phantom would catch the highest-risk CT bug (axis flips).
- Units are documented (`physics-conventions.md`) but there is no machine-checked unit tag; use naming (`_cm`, `_mm`, `_keV`) and a conventions test that greps public signatures **[P]**.
- `run_meta.json` is written by only some scripts. Define one required-fields list and check it in a `post_experiment` hook.
- Determinism: HPC vs laptop agreement was established for one XCIST cell (F-036, preliminary, awaiting review). Learning code on GPU is untested for determinism.
- Dataset/provenance: `hpc datasets list` shows `belauri-geography` complete and `datasets` "no manifest" — **inspect, do not adopt** without the user's say. CTsynth's AAPM data is local only (`data/aapm`, gitignored); no manifest convention.

## 5. Current workflow (as practised, reconstructed from records)
```
Human + GPT Astra idea ─► Opus drafts Q/EXP/task (titles sometimes wrong, D-007) ─► Sonnet/Haiku implement (by label, no contract)
   ─► pytest/ruff/mypy by hand (CI only on push) ─► experiment script run: `research run exec` (laptop) | manual bundle → hpc submit
   ─► `hpc sync` ─► `ctsynth postcheck` (manual) ─► Opus writes findings (mostly preliminary; many "awaiting review")
   ─► checkpoint (CP-009 notes backlog of unreviewed findings) ─► next step chosen by human from checkpoint text
```
Observed frictions: PLAN-002 "AAPM benchmark" holds 31 tasks including all XCIST/E1–E4 work (T-032..T-042) — wrong container, so `research current` prints misleading "7/31"; PLAN-001 has 4 stale `todo` tasks (T-005/6/8/9) that earlier work superseded **[S: confirm with human]**; 14 findings await independent review (T-033, T-037) — the promotion gate is the actual bottleneck; F-038 (a bundle shipped `.venv-xcist`) was a missing pre-submit check — exactly what a `preflight` hook automates; Opus authored everything (`agent:opus` on every record) so independence of review depends on a different model being routed.

## 6. Research OS 1.0 capability matrix
Legend: **C** configured and used · **A** available, not configured · **P** needs CTsynth implementation · **U** unsupported.

| Need | Status | How |
|---|---|---|
| Model allocation | **C** (just set) but inert (A2) | `strategy sonnet_first`; set `orchestration.enabled: true` (human nod: changes behaviour of `task`/`verify`) |
| Per-task routing | A | `strategy route T-…`; classes checked/unchecked/mechanical/scientific/planning/verification |
| Task contracts (files, criteria, checks) | A | `task create --check/--acceptance`, `orchestrate contract/freeze/pin` |
| Parent/child decomposition | A | `orchestrate delegate PARENT` |
| Bounded retries/escalation | A | strategy attempts fast:3, balanced:2 → human |
| Independent verification | A | `orchestrate verify T --tier L --record`; findings `review` |
| Hooks (preflight / post_implementation / pre_experiment / post_experiment) | A, **none registered**, need human approval | §7 |
| Usage accounting | A, empty | `orchestrate dispatch --headless` records; interactive sessions don't, so **no measured savings exist yet** |
| Dataset discovery | C | `hpc datasets list/check`; no CTsynth registry needed |
| HPC submit/sync/doctor | C | keep; `doctor` is green |
| Evidence binding, `experiment check` | C partially | `experiment check EXP-010` passes; bind coverage for older findings is "evidence-gap records" |
| Scientist Guide, checkpoints, search | C | keep |
| Project skills | C: 9, all generic Research OS skills; **no CTsynth-specific skill** | §7 |
| Autonomous post-experiment pipeline / scheduler | **U** | `sync` + hook + human; document the manual sequence |
| MCP for orchestrate/hook/strategy/tuning | **U** (CLI only) | use CLI |
| XCIST/physics/geometry validators | **P** | project CLI/library + hook |

## 7. Research OS plan for CTsynth (proposals only; nothing activated)
**Strategy:** keep `sonnet_first` unmodified. Routing rules (all via `strategy route`, no override unless noted):
| Work | Level | Gate |
|---|---|---|
| Deterministic transforms with frozen tests (hash dedup, converters, markers) | fast (Haiku) | `orchestrate verify --tier fast --record`, 3 attempts → Sonnet |
| Sonnet-authored contract + tests for XCIST/data loader, split file, power analysis spec | balanced | tests frozen (`orchestrate freeze`) before implementation |
| Architecture, doc consolidation, integration of shared files (pyproject, `__init__`, `.research`) | balanced, coordinator only | human review |
| Independent review of findings (T-033, T-037) | a *different* level than author | `finding review`; Opus only when the claim is scientific |
| Scientific synthesis/interpretation, pre-registration wording | deep (Opus) | at experiment boundaries only |
| Parameter tuning with explicit objective | fast, `research tuning declare…` | validation split only; test once |
Override only if observed failures justify it (record why).

**Hooks** (to be *proposed* with `hook add`; humans approve; none created now). Fast = under ~20 s, deterministic.
| Stage | Command (sketch) | Timeout | Failure |
|---|---|---|---|
| `post_implementation` | `uv run ruff check . && uv run ruff format --check . && uv run mypy` | 120 s | blocking |
| `post_implementation` | `uv run pytest -x -q -m "not integration and not hpc"` | 300 s | blocking |
| `preflight` / `pre_experiment` | `uv run ctsynth preflight --repo . --config <cfg>` (extended per A1) | 60 s | blocking |
| `pre_experiment` (HPC) | bundle-content check: fail if staged tree contains `.venv*`, `data/`, `*.pt` (F-038) | 60 s | blocking |
| `post_experiment` | `ctsynth postcheck` + required metrics/`run_meta.json` fields | 60 s | blocking |
| `post_experiment` | `research experiment check EXP-…` | 60 s | **advisory** (needs human interpretation) |
Not hooks: the XCIST integration tests (need `.venv-xcist`; on-demand `-m integration`), any experiment, any scientific threshold.

**Project-specific tools** (decision rule: library if called from code, CLI if a human/agent runs it repeatedly, hook if it must gate, skill if it is judgment):
| Tool | Form |
|---|---|
| Unified source-clean check (A1) | library fn in `runmeta`, used by `preflight` + `runmeta`; hook calls CLI |
| Paired-record validator (`load_record` already verifies hashes) | CLI `ctsynth paired check DIR` thin wrapper, used by post_experiment hook |
| XCIST environment smoke (`gecatsim` import, library path, one tiny sim) | CLI `ctsynth xcist smoke`; on-demand |
| AAPM↔paired converter + axis test | library |
| Geometry/units sanity (affine, spacing, orientation vs known disk) | pytest `physics`-marked tests |
| Run-record comparison | **existing** `research run compare` / metrics; no wrapper |
| Dataset inspection | **existing** `research hpc datasets`; filesystem is the registry |

**Skills** (new, each ≤ ~40 lines, loaded by `task --skill`/plan, never always-on): `ctsynth-xcist` (geometry, libgomp quirks, env, F-036/F-037 gotchas), `ctsynth-experiment` (CTsynth specifics of `run-experiment`: preflight → dry-run → run → postcheck → bind → synthesize, pre-registration rules from CLAUDE.md), `ctsynth-physics-conventions` (units/axes pointer). Do **not** create skills for generic Research OS procedures (already present). Move the CLAUDE.md pre-registration/promotion text into the experiment skill later, keeping CLAUDE.md short **[P]**.

**Experiment lifecycle** (per E-experiment): `scientific-handoff` → register EXP with pre-registration criteria and detection floor (task T-032 first) → `post_implementation` hooks on code → `preflight` hook → `run exec` / `hpc submit --dry-run` (human confirms reservation) → `hpc sync` → post_experiment hooks → `experiment check` → `evidence bind` + `verify` → synthesis by Opus → independent review by non-author → Scientist Guide refresh → checkpoint. Failed runs stay registered. Operational task completion ≠ supported finding (D-008).

**Context conventions:** session start = `research current` + `task context T-…`; handoff = `research report --full --check`; permanent knowledge goes to notes/decisions, not CLAUDE.md. Savings: unmeasured; the only measurable plan is to enable headless dispatch for Haiku tasks so `usage summary` has data.

## 8. Target architecture
```
src/ctsynth/
  core/        seeding, config(+config_sha256), device, hashing, provenance (source-clean, runmeta)   # no deps on science
  physics/     unchanged (materials, spectrum, projector, forward)
  aapm/        unchanged; + convert.py (aapm↔paired axes)
  paired/      schema, mock, phantoms, xcist(env + run)  ; + dataset.py (v1 generator, split, loader)  <- from projects/p04
  inference/   (planned, only when E1 starts) classical, mlp/unet wrappers sharing aapm.losses
  hpc/         preflight/postcheck (thin, wraps Research OS)
projects/pNN/  thin drivers: parse config → call library → write metrics via `research` API → figures
configs/       one YAML per registered EXP-id (p04 included)
tests/         mirrors src; markers unit|physics|regression|integration|hpc
docs/          ≈9 permanent files (see §3)
```
Preserve unchanged: `physics/*`, frozen tables and their scripts, `aapm/*` APIs, `paired/schema.py`, HPC native protocol and bundle excludes, `hpc_smoke.py`/`hpc_env_probe.py` (hash-frozen), seeding, all `.research/` history. Refactor: `hashing`/`runmeta`/`check` provenance trio, p04 scripts → library, boilerplate sha256s, docs. Do **not** introduce: a plugin registry, a config framework, a custom tracker, or a dataset registry.
The `core/` rename is optional **[P]**; skip it if it only churns imports (use modules in place).

## 9. Risks and decisions needed from you
1. Enable `orchestration.enabled: true`? (activates contracts/verify; reversible.) — recommended.
2. Approve the proposed hooks (§7); approval is yours (`hook approve`).
3. Supersede D-009 (stale, contradicts D-010) and the Opus-authored `scientific-decisions.md` text — Opus-owned file; I recommend a pointer rewrite.
4. T-042 cleanup of `.venv-xcist` copies on the cluster/local bundles (already open).
5. Whether to relocate T-032..T-042 into a new PLAN-005 "XCIST E1–E4" (records are history; this is a re-parenting, not deletion) and close/abandon stale PLAN-001 tasks.
6. Adopt/ignore `shared-data/datasets` (no manifest) — nothing done.
7. Unverified here: whether `tests/paired/test_xcist.py` needs gecatsim; `runmeta` redundancy vs `run show`; usage accounting cost.

## 10. Phase 3 backlog
Tasks T-043..T-060 under PLAN-004 (`research task list`); acceptance criteria and dependencies are on the tasks (`research task context T-…`).
Task IDs: T-043 source-clean unification (A1) · T-044 sha256 dedup (D1) · T-045 seed_everything decision (human) · T-046 pytest markers ·
T-047 AAPM↔paired converter (A6) · T-048 HPC script paths (A4) · T-049 docs consolidation · T-050 supersede D-009 · T-051 attenuation cross-check (A7) ·
T-052 plan hygiene (human) · T-053 orchestration pilot (A2) · T-054 hooks proposal (human approves) · T-055 config_sha256 move (D2) ·
T-056 p04 → library · T-057 README/architecture refresh · T-058 CTsynth skills · T-059 xcist smoke tool · T-060 exp002 results exception (A5).
Recommended order: T-043 → T-054 (human approves hooks) → T-053 (enable orchestration, pilot on T-044) → T-044/T-055/T-046/T-045 →
T-047 → T-060 → T-049/T-050 → T-057 → T-058 → T-051 → T-048 → T-059; T-056 only after T-032 (E1–E4 prerequisite) is settled; T-052 any time (human).
**Do not let Phase 3 block T-032/T-033/T-037:** the scientific next step in CP-009 (dataset v1, F-037 review) is independent of everything above except that T-043 should land before the next run.
