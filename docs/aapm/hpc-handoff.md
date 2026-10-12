# HPC experiment-bundle requirements (specification only — Prompt 4 not implemented)

What an HPC run must carry so its results are citable under the same rules as a laptop run.
Nothing here is implemented; no SLURM system exists and no cluster has been contacted.

## The bundle
| Field | Source today | Status |
|---|---|---|
| Git source commit | `runmeta.collect()["git_commit"]` | ✅ exists |
| Source-tree cleanliness | `git_dirty_source` (src/tests/configs only) — **the citability signal**, not `git_dirty` | ✅ exists (F-017/F-029) |
| Resolved config + seed | `ExperimentConfig` (frozen, `extra=forbid`) + `config_sha256` | ✅ exists |
| Dataset provenance | `manifest.json` sha256 per input; Zenodo md5 for AAPM (`scripts/fetch_aapm.py` verifies and deletes on mismatch) | ✅ exists |
| Environment lock | `uv.lock` | ✅ exists; **not yet recorded in run_meta** |
| Experiment/run ids | Research OS `EXP-*` / `RUN-*` | ✅ exists; see D-007 (never put a number in a title) |
| Output schema | per-experiment `metrics.json` + `run_meta.json` + `manifest.json` | ✅ convention established |
| Metric bindings | `research metric add` → `evidence bind` → `evidence verify` | ✅ exists |
| Import validation | — | ❌ **missing**: nothing validates a transferred output dir before registration |
| Immutable evidence vs mutable interpretation | outputs gitignored; findings in Research OS | ✅ by convention |

## Gaps to close in Prompt 4 (minimal)
1. **Record `uv.lock` hash in `run_meta.json`** — one line in `runmeta.collect()`; without it the environment is not pinned in the evidence.
2. **An import validator**: given a transferred output dir, check `run_meta.json` exists, `git_dirty_source` is false, the commit is an ancestor of the current branch, every `manifest.json` hash matches, and the metrics keys match the experiment's declared metrics. Refuse to register otherwise.
3. **Distinguish historical from repeated runs**: a re-run of an experiment must get a new `RUN-*` and a `--label` saying what changed (already the practice: RUN-0003→0006 are relabelled attempts; RUN-0008 is the recorded ladder).

## Inherited rules (non-negotiable)
- HPC is **run-only**; no code is developed there (`docs/hpc-workflow.md`).
- A run whose `git_dirty_source` is true is **not citable**.
- Laptop results never substitute for HPC results, and reduced-resolution results are **not** comparable to the published leaderboard (s1 divides by 3·512²).
- Numbers enter findings through `evidence bind`, never by transcription.
