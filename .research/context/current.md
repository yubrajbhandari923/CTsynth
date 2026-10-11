# Research context — CTsynth

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~850 tokens.
> Generated 2026-10-11T19:33:44-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

## Latest checkpoint

**CP-001** Checkpoint — 2026-10-11 (2026-10-11)

## Current baseline

_none set_

## Active & open questions

- Q-003 [investigating] P01-H2: Does material-map error depend more on spectral separation (kVp pair / photon-counting bins) and noise than on the inference architecture?
- Q-001 [investigating] P01: Can material maps be inferred from dual-energy/spectral CT measurements well enough to seed editable virtual patients?
- Q-002 [open] P01-H1: Does a physics-consistency (differentiable polychromatic forward model) loss beat image-domain baselines when the test-time forward model is mismatched from the training one (no inverse crime)?
- Q-004 [open] P01-H3: Do models trained on procedural phantoms generalise to held-out material compositions (e.g., unseen iodine concentrations)?

## Established findings (what we believe)

_none_

## Failed directions (do not repeat without a new reason)

_none_

## Current experiments

- EXP-002 [proposed] Spectral identifiability toy: bins x basis size x spectrum mismatch — runs 0

## Active plans

- PLAN-001 P01 foundation: spectral forward model and baselines — 3/7 tasks done · objective: Validated spectral forward model + classical baseline decomposition on laptop, then HPC-ready configs
  - next ready: T-005 Classical baseline: image-domain two-material decomposition on toy phantom (`research task context T-005`)
  - running: T-007 EXP-002: identifiability toy experiment (designs x basis size x mismatch) (claimed by opus)
  - BLOCKED: T-004 Read AAPM starting_kit.tgz (103 MB) for spectra/geometry/metric (needs user OK to download)

## Active decisions

- D-002 (2026-10-11) EXP-002 runs on the 1 keV table as-is; iodine K-edge (33.17 keV) is under-resolved and the shared forward/estimator table makes this a shared-model limitation, not a forward/estimator mismatch. Iodine ABSOLUTE accuracy claims are therefore out of scope for EXP-002; identifiability/conditioning/noise-amplification claims remain valid because they depend on the spectral contrast actually present in the model used. A K-edge-resolved table (v2, fine grid near edges) is required before any quantitative iodine-concentration claim.
- D-001 (2026-10-11) No third-party physics/projector is a runtime dependency; ctsynth.physics is a small torch reference implementation. Real attenuation tables enter as frozen versioned npz generated offline with xraylib; LEAP/XCIST are opt-in test oracles; ct_laboratory (no license) is conceptual reference only.

## Blockers

_none_

## Project skills

- 8 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md
