# Research context — CTsynth

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~3472 tokens.
> Generated 2026-10-30T11:01:25-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

Git: `feature/xcat-simulator` @ `863565eec9` (uncommitted changes)

## Latest checkpoint

**CP-019** EXP-040 validated: corrected 10-patient closed loop, pitch-1.4 cause, F-050 supported (scoped) (2026-10-30)
- **Understanding:** All 10 test patients pass guarded closed loop (v2 == v1). Pitch-1.4 failure of arm D = FiLM extrapolation of a constant-1 pitch input (validation-patient override reproduces it). C-P1 gap mostly image restoration. Skin/blood not recovered by arm C. F-050 reviewed twice (R1 needs_work, R2 supported, confidence medium).
- **Current problem:** Evidence partitions are declared but no structured split metadata exists; RUN-0118 whole-tree git_dirty (unrelated files); conditioning encoding not repaired (needs retraining)
- **Next step:** No new experiment required; optional: centred-conditioning retrain as a separate decision

## Current baseline

_none set_

## Active & open questions

- Q-009 [investigating] P09-Q1: Can a 3D U-Net recover XCAT-derived physical material concentration maps (g/cm3 per elemental composition) from simulated monoenergetic COPD CT of unseen patients, such that the predicted phantom reproduces the input CT through LEAP and transfers to a different photon energy?
- Q-002 [investigating] P01-H1: Does a physics-consistency (differentiable polychromatic forward model) loss beat image-domain baselines when the test-time forward model is mismatched from the training one (no inverse crime)?
- Q-006 [investigating] P02-Q6: Can a network recover physically meaningful material maps from spectral measurement consistency ALONE (no paired labels)? Prior analysis (F-014) says NO for the AAPM 2-measurement/3-material case: the null direction is invisible to the data, so a pure consistency loss is minimised by a one-parameter family containing the truth and infinitely many wrong maps. The scientific content is therefore WHICH added prior selects the truth, and whether it does so for physical reasons or by accident.
- … 6 more (`research question list`)

## Established findings (what we believe)

- F-050 [supported, medium] Physical-concentration U-Net reproduces simulated CT at 70 and unseen 80 keV and beats 70-keV voxelwise inverses, mostly by image restoration (simulated, 10 test patients) — evidence: EXP-040, RUN-0117, RUN-0113, RUN-0118
- F-020 [supported, medium] HELLWEGE 2026 IS NOT A PRECEDENT FOR OUR PROBLEM (full arXiv PDF text read). Their task is 2 basis materials (soft tissue, calcification) from 2 spectra: SQUARE and determined by counting, so pixelwise identifiability is automatic and never discussed in the paper. Our AAPM task is 3 materials from 2 spectra: underdetermined, and in their Eq. 6 objective L = (1/|D|)(1/M) sum_i [||p_LE - phat_LE||^2 + ||p_HE - phat_HE||^2] (unweighted MSE on log line integrals, no I0, no Poisson term, NO regulariser, no nonnegativity, no sum-to-one, no TV, no entropy, no softmax) our F-014 null direction would be EXACTLY FLAT. The paper reports no ablations at all; its gain over MBIR (VMI RMSE 5.83 vs 13.07 HU at 50 keV) comes from the network and training, not from the loss. The authors state the inverse crime explicitly (Sec 4.2: same polychromatic model generates and decodes) and run no mismatch experiment. Early stopping uses projection-domain validation loss with no ground truth, which is a genuinely label-free protocol worth copying. Unverified: energy-bin count, final activation, how 3+ AAPM tissue classes were merged into 2, whether data were noisy, and Table 1 mean-vs-median. — evidence: Q-006
- F-021 [supported, medium] PENG 2024 IS NOT A LABEL-FREE COMPARATOR. Loss L = lambda*||A[x1;x2] - [mu_H;mu_L]||^2 + L_GAN, image-domain on linear attenuation coefficients with a LINEAR 2x2 per-pixel matrix A from ROI means: no spectrum, no projector, no Poisson term. Because A is square and invertible the fidelity minimum is exactly the noisy direct inversion, so the GAN term supplies ALL the noise suppression (SD -91-97%). The discriminator requires an UNPAIRED reference set of clean material maps built by thresholding SECT/high-energy images from other patients, so the method carries a learned prior fitted to label-like data: it is 'unpaired-reference adversarial', not label-free. Evaluation ground truth uses the same thresholding recipe, so the reference prior is in-distribution with the evaluation - a self-favouring protocol. Like Hellwege it is 2-from-2 (determined, no sum-to-one row) and so says nothing about our 3-from-2 null space. Usable only as an unpaired-reference-prior baseline on a 2-material subproblem, and must be labelled as such. Unverified: published lambda (supplement not obtained; 1e4/1e2 from arXiv v1 only), exact GAN loss form, hyperparameter protocol; no code released. — evidence: Q-006
- … 30 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

- F-048 [preliminary] EXP-036 engineering failure: arm C jobs RUN-0098/99/100 crashed with NaN because zero-spectrum-weight energies (1.25-3.25 keV, mu up to 1.5e3 / 3.2e4 cm^-1) overflow exp(-mu.P) to inf for slightly negative paths and 0*inf=NaN; fixed by dropping zero-weight energies (exact on finite paths, test added); arms A/B unaffected. Any other free-fraction solver on this forward model needs the same guard.
- F-047 [preliminary] EXP-036 arms C (three free fractions) and A_img (image-derived support, sum=1) did not test sum-free identifiability: the image-derived support cannot hold partial-volume pixels (truth masked to the support already has data MSE 1.6e-5 vs ~1e-10 unmasked), every C init plateaus at 2.7-2.8e-6 and A_img at 4.7-4.9e-6; their reported band RMS (1e5-1e6; 1.5) are metric artefacts. Bimodality prior (arm D, lambda 1e-5) gave no gain over arm A (s1 0.087-0.100 vs 0.014-0.048).
- … 3 more (`research finding list --kind failure`)

## Current experiments

- EXP-034 [needs_review] Acquisition identifiability theory: SECT vs DECT vs PCCT Jacobian rank, singular values, Fisher information — runs 2
- EXP-033 [needs_review] EXP-024 anomaly diagnostics: occ_softmax seed 2 and oracle collapse — runs 5
- EXP-026 [ready] Image-domain vs projection-domain vs reconstruction-domain consistency — runs 0

## Active plans

- PLAN-009 P09 Physical-phantom inversion: composition-concentration targets, baselines, U-Net, closed-loop LEAP and 70->80 keV test — 7/7 tasks done · objective: Predict XCAT-derived per-voxel composition mass concentrations (g/cm3) from cached simulated COPD CT and validate by passing the predicted phantom through LEAP at the input and a different energy.
- PLAN-008 P08 COPD XCAT noisy-CT simulator and inverse U-Net (staged simulate-then-train, node004 only) — 7/7 tasks done · objective: Reproducible, resource-efficient pipeline: XCAT -> LEAP projections -> Poisson noise / controlled acquisition and kernel variants -> compact CT cache -> 3D U-Net recovering organ-group labels; compare A clean, B noisy, C varied, D conditioned on identical patient splits.
- PLAN-007 P07 TrueCT data visualization and LEAP reconstruction pilot (no pairing, no DukeSim) — 2/4 tasks done · objective: Open TrueCT reference/COPD phantoms, reference recon and projections as NIfTI/PNG for 3D Slicer; reconstruct projections with LEAP on DRCC GPU. Data conversion pilot, not a hypothesis test.
  - running: T-075 TrueCT NIfTI, preview and LEAP reconstruction pilot (claimed by claude-code) — STALE, idle 23.2h
  - awaiting verification: T-077 XCAT phantom -> mu(70 keV) -> LEAP projection -> LEAP helical FBP pipeline (reference, COPD 22728, COPD 21535)

## Needs attention (verification)

- [verification failed] T-053 Pilot orchestration: enable, contract and verify one Haiku task (A2): no deterministic verifier ran (no task checks and no approved post_implementation hooks): this is not a pass
- [reviewer disagrees] F-012 AAPM 3-material conditioning, computed from the official starting-kit spectra and mu tables (md5-verified kit, 10 sample cases): the spectrum-weighted effective attenuation matrix is [[0.26697,0.31321,3.01182],[0.21930,0.24788,1.64238]] 1/cm for (adipose,fibroglandular,calcification) at 50/80 kVp. Adipose and fibroglandular are spectrally near-degenerate: column cosine 0.999834, i.e. 1.04 degrees apart. The unconstrained 2x3 null direction is [-0.723,+0.691,-0.008], confirming F-010's prediction concretely: adipose can be traded against fibroglandular with essentially no measurement change. Appending the sum-to-one row makes the 3x3 system full rank (singular values 3.697, 1.176, 0.00290) but POORLY CONDITIONED at cond=1273. Therefore the partition constraint restores identifiability but error in the adipose/fibroglandular direction is amplified ~1000x; exact recovery in the noiseless challenge data is possible, while any model mismatch or interpolation error is magnified by that factor. Image-domain linear inversion is expected to be far worse than the published model-based winners (6.8e-7).: reviewer3: needs work — RECOMPUTED from model_data npy: mu matrix, col angle 1.043deg, SVs 3.697/1.176/0.0029, cond 1272.9 all reproduce. BUT it is the zero-thi
- [reviewer disagrees] F-039 EXP-024 NEGATIVE RESULT (synthetic AAPM, validation, noise-free): no tested output parameterisation or soft-partition penalty is shown to improve physics-only label-free s1 over the EXP-023 baseline: claude-deep: needs work — Tried: recomputed every arm's 3-seed mean s1 and proj RMSE from results.json (RUN-0047-49,64-72,74-79,83-85,88-90); applied the rule u
- [reviewer disagrees] F-040 RUN-0066 (occ_softmax seed 2, s1 0.0914) is a genuine structured label-free solution; why it arose is unresolved: claude-deep: needs work — R2 wording only: title says 'the projection loss did not select it' but the corrected statement says selection is unresolved (within-a
- [reviewer disagrees] F-041 Exact F-012 null direction is ~330x less visible to the projection loss than the pure A<->F swap; n = swap + calcification (-0.78%) + tissue-sum change (-3.1%) per unit: claude-deep: needs work — R2 delta review (RUN-0091 diag.json). Numbers reproduce: M@n=(-1.4e-15,-2.2e-16); null profile 4.03e-6 @0.1, 1.61e-5 @0.4 (linear, sym
- [reviewer disagrees] F-042 Softmax-output runs (3-channel, oracle-S) end as dead-gradient one-hot states in most seeds; cause not isolated; converged-oracle identifiability untested: claude-deep: needs work — R2 wording only: title 'saturate ... and lose all gradient within ~100 steps' overgeneralises - RUN-0088 ~261 steps, RUN-0089/90 (orac
- [reviewer disagrees] F-045 EXP-036 pilot (3 validation cases, noiseless, oracle occupancy S(x), calcification=0 arm A): per-case full-view L-BFGS reaches projection MSE 1.7e-9 (557-calcified-pixel case; 2.4e-8 and 3.2e-8 at 1062 and 1441) from five different inits to the same solution, clipped s1 0.31->0.014-0.048, but the >1/8 cycles/px adipose-fraction error stays 0.21/0.35/0.42 (criterion 0.05): convergence fixes the bulk map and does not recover the spatially varying A/F error.: research-balanced-reviewer: needs work — Numbers verified against the six JSONs. Major: hf computed on unclipped a while s1_clipped clips (unclipped s1 0.209/0.
- [reviewer disagrees] F-046 EXP-036 pilot: no non-oracle (arm, init) was converged-and-recovered. B from uniform is cap-limited (MSE 5-6e-8, loss falling ~11% per 200 it at it1000, residual mostly low-frequency 0.14), B rough stalls at 4e-8 with its oracle swap error unchanged, B with oracle null-pattern init recovers (s1 0.011-0.015, hf 0.004-0.007): nearly invisible perturbations persist, visible ones are removed, consistent in ordering with Stage 0. Whether B-from-uniform is information- or solver-limited is not decided.: research-balanced-reviewer: needs work — lambda*L1 figure was B-null only; B null not converged; B rough stall could be a flat valley; A positive control failed

## Active decisions

- D-012 (2026-10-13) D-009 is superseded: the CTsynth 'ctsynth prepare|verify|export|import|report' laptop<->HPC CLI no longer exists (removed in 6a98348 when the Research OS native HPC protocol replaced it, D-010). The current workflow is docs/hpc.md: ctsynth preflight/postcheck/bundle-check/run-check guards around research hpc. Phase 3 conventions recorded with it: (1) ONE definition of 'source' for citability, runmeta.SOURCE_PATHS (src tests configs scripts projects pyproject.toml uv.lock), shared by run metadata and preflight; projects/**/results* are outputs; unknown git state is None and never citable; (2) AAPM<->paired axis conversion only through ctsynth.aapm.convert (index relabel, no invented affine); (3) hooks are registered but unapproved/disabled until a human enables them; (4) orchestration stays disabled.
- D-011 (2026-10-12) XCIST is the first independent CT forward-data producer; DukeSim deferred

## Blockers

- [checkpoint] CP-019: Evidence partitions are declared but no structured split metadata exists; RUN-0118 whole-tree git_dirty (unrelated files); conditioning encoding not repaired (needs retraining)

## Project skills

- 15 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md
