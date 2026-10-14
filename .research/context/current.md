# Research context — CTsynth

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~3229 tokens.
> Generated 2026-10-14T04:36:22-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

Git: `master` @ `f145ebee75`

## Latest checkpoint

**CP-012** Milestone A: AAPM end-to-end physics pipeline demonstrated (gradient flow yes, label-free recovery no) (2026-10-13)
- **Understanding:** Verified: official projector port equals the exact chord and the kit sinograms (EXP-017: 0 violating rays of 50 cases x 2 kVp, max error ratio 0.30 of the pre-derived tolerance); gradients match finite differences to <2e-6 incl. real 512^2 (EXP-018); supervised smoke passed on 2nd run after a documented NaN/undertraining failure (EXP-019); three-arm demo EXP-020: supervised s1 0.029, physics-only s1 0.392 with 99.7% of error along the F-012 null direction and a collapsed uniform adipose/fibro prediction, physics+sum<=1 hinge s1 0.504. Dataset (1000 cases) acquired and validated (EXP-029), no duplicates, the 10 kit cases are not in it, ALL cases share one support mask (EXP-031), frozen 800/100/100 split in configs/aapm_split_v1.json. Single seed, 8 val cases: descriptive. No finding has been independently reviewed.
- **Current problem:** Physics-only arm was optimisation-limited (projection RMSE 4.3e-3 vs supervised 2.6e-3). Fixed single support mask limits Exp 8. No independent review done yet; findings not yet registered.
- **Next step:** EXP-021 classical baselines (validation only) and EXP-022 supervised U-Net on the full 800-case train split; then EXP-023 physics-only with 3 seeds.

## Current baseline

_none set_

## Active & open questions

- Q-002 [investigating] P01-H1: Does a physics-consistency (differentiable polychromatic forward model) loss beat image-domain baselines when the test-time forward model is mismatched from the training one (no inverse crime)?
- Q-006 [investigating] P02-Q6: Can a network recover physically meaningful material maps from spectral measurement consistency ALONE (no paired labels)? Prior analysis (F-014) says NO for the AAPM 2-measurement/3-material case: the null direction is invisible to the data, so a pure consistency loss is minimised by a one-parameter family containing the truth and infinitely many wrong maps. The scientific content is therefore WHICH added prior selects the truth, and whether it does so for physical reasons or by accident.
- Q-005 [investigating] P01-Q5: On the AAPM DL-Spectral CT benchmark, how do a constrained classical estimator and a small supervised U-Net compare for 3-material decomposition, and what makes the 3-from-2 problem well posed?
- … 5 more (`research question list`)

## Established findings (what we believe)

- F-020 [supported, medium] HELLWEGE 2026 IS NOT A PRECEDENT FOR OUR PROBLEM (full arXiv PDF text read). Their task is 2 basis materials (soft tissue, calcification) from 2 spectra: SQUARE and determined by counting, so pixelwise identifiability is automatic and never discussed in the paper. Our AAPM task is 3 materials from 2 spectra: underdetermined, and in their Eq. 6 objective L = (1/|D|)(1/M) sum_i [||p_LE - phat_LE||^2 + ||p_HE - phat_HE||^2] (unweighted MSE on log line integrals, no I0, no Poisson term, NO regulariser, no nonnegativity, no sum-to-one, no TV, no entropy, no softmax) our F-014 null direction would be EXACTLY FLAT. The paper reports no ablations at all; its gain over MBIR (VMI RMSE 5.83 vs 13.07 HU at 50 keV) comes from the network and training, not from the loss. The authors state the inverse crime explicitly (Sec 4.2: same polychromatic model generates and decodes) and run no mismatch experiment. Early stopping uses projection-domain validation loss with no ground truth, which is a genuinely label-free protocol worth copying. Unverified: energy-bin count, final activation, how 3+ AAPM tissue classes were merged into 2, whether data were noisy, and Table 1 mean-vs-median. — evidence: Q-006
- F-021 [supported, medium] PENG 2024 IS NOT A LABEL-FREE COMPARATOR. Loss L = lambda*||A[x1;x2] - [mu_H;mu_L]||^2 + L_GAN, image-domain on linear attenuation coefficients with a LINEAR 2x2 per-pixel matrix A from ROI means: no spectrum, no projector, no Poisson term. Because A is square and invertible the fidelity minimum is exactly the noisy direct inversion, so the GAN term supplies ALL the noise suppression (SD -91-97%). The discriminator requires an UNPAIRED reference set of clean material maps built by thresholding SECT/high-energy images from other patients, so the method carries a learned prior fitted to label-like data: it is 'unpaired-reference adversarial', not label-free. Evaluation ground truth uses the same thresholding recipe, so the reference prior is in-distribution with the evaluation - a self-favouring protocol. Like Hellwege it is 2-from-2 (determined, no sum-to-one row) and so says nothing about our 3-from-2 null space. Usable only as an unpaired-reference-prior baseline on a 2-material subproblem, and must be labelled as such. Unverified: published lambda (supplement not obtained; 1e4/1e2 from arXiv v1 only), exact GAN loss form, hyperparameter protocol; no code released. — evidence: Q-006
- F-014 [supported, medium] F-012's null space is now demonstrated OPERATIONALLY, not just algebraically (synthetic data, src/ctsynth/aapm/classical.py + tests/aapm/test_classical.py). Unconstrained per-pixel inversion of the 2x3 AAPM system genuinely fails (max per-pixel error > 0.05 in volume fraction) and its error vector is parallel to the SVD null direction to <1e-6 deg; that independently-computed null vector lies 0.024 deg from the published F-012 vector [-0.723,+0.691,-0.008]. Crucially the error is INVISIBLE to the data: |e @ mu^T| < 1e-12, i.e. the wrong answer reproduces the measurements exactly. Adding the sum-to-one row restores a unique solution (exact recovery to ~1e-10 in float64 on noiseless synthetic tissue pixels) but amplifies measurement perturbation: the derived worst-direction amplification sigma_max(B^-1[:,:2])*|d|/|x| = 181 for a typical 50/50 adipose/fibroglandular pixel, matched to rel 1e-4, which is 0.14x cond(B)=1273 (below cond because the appended sum row is exact and unperturbed). — evidence: Q-005
- … 22 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

- F-038 [preliminary] PROCESS: four HPC bundles (RUN-0019..RUN-0022) accidentally included the local .venv-xcist

## Current experiments

- EXP-025 [running] Physics plus prior ablations — runs 2
- EXP-024 [proposed] Occupancy-aware output parameterisations — runs 0
- EXP-027 [proposed] Physics mismatch and calibration sensitivity — runs 0

## Active plans

- PLAN-005 P05 AAPM end-to-end physics pipeline and benchmark suite — 8/10 tasks done · objective: Milestone A: verified differentiable fan-beam AAPM pipeline and label-free demo. Milestone B: reproducible AAPM benchmark (classical, supervised, physics-only, priors, hybrid) on the full 1000-case dataset
  - next ready: T-062 Polychromatic AAPM forward: fractions -> kVp-switched transmissions (`research task context T-062`)
- PLAN-004 P04 Phase 2 audit and Phase 3 refactoring backlog — 13/18 tasks done · objective: Execute the audit roadmap (docs/architecture-workflow-audit.md) without changing scientific results, the HPC baseline or history
  - BLOCKED: T-051 Cross-check attenuation_v1 vs AAPM kit mu tables with stated tolerance (A7) — Not well-defined: attenuation_v1 is water/iodine/calcium, the AAPM kit is adipose/fibroglandular/calcification (composition of kit calcification unstated); no shared material to compare except possibly calcium. Needs a human/Opus scoping decision before any tolerance exists
  - BLOCKED: T-052 Plan hygiene: re-parent XCIST E1-E4 tasks, resolve stale PLAN-001 tasks — Human decision only: re-parenting is impossible in Research OS 1.0 (documented in PLAN-002 context); PLAN-001 tasks T-005/6/8/9 reviewed and left open. Close/abandon them only if you decide they are superseded. Not a blocker for any experiment.
  - BLOCKED: T-053 Pilot orchestration: enable, contract and verify one Haiku task (A2) — Out of scope by owner instruction for Phase 3 (do not enable orchestration). Re-open only on an explicit human decision
  - BLOCKED: T-054 Propose and register CTsynth hooks (not approved) — Hooks registered; awaiting human 'research hook enable/approve' (see docs/research-os.md)
  - BLOCKED: T-056 Move p04 dataset generation, split and loader into src (with tests) — Partial work done (voxel_chords moved, drivers testable). Dataset v1 generator/split/loader depends on T-032 (unresolved scientific design)
- PLAN-003 P03 label-free physics-consistent decomposition (B0-B4 ladder) — 2/2 tasks done · objective: Implement and fairly compare B0 classical constrained, B1 regularized classical, B2 supervised U-Net, B3 label-free physics-consistent U-Net, B4 physics+priors; ablate priors individually; test identifiability, information, noise, mismatch, generalization

## Needs attention (verification)

- [verification failed] T-053 Pilot orchestration: enable, contract and verify one Haiku task (A2): no deterministic verifier ran (no task checks and no approved post_implementation hooks): this is not a pass
- [reviewer disagrees] F-012 AAPM 3-material conditioning, computed from the official starting-kit spectra and mu tables (md5-verified kit, 10 sample cases): the spectrum-weighted effective attenuation matrix is [[0.26697,0.31321,3.01182],[0.21930,0.24788,1.64238]] 1/cm for (adipose,fibroglandular,calcification) at 50/80 kVp. Adipose and fibroglandular are spectrally near-degenerate: column cosine 0.999834, i.e. 1.04 degrees apart. The unconstrained 2x3 null direction is [-0.723,+0.691,-0.008], confirming F-010's prediction concretely: adipose can be traded against fibroglandular with essentially no measurement change. Appending the sum-to-one row makes the 3x3 system full rank (singular values 3.697, 1.176, 0.00290) but POORLY CONDITIONED at cond=1273. Therefore the partition constraint restores identifiability but error in the adipose/fibroglandular direction is amplified ~1000x; exact recovery in the noiseless challenge data is possible, while any model mismatch or interpolation error is magnified by that factor. Image-domain linear inversion is expected to be far worse than the published model-based winners (6.8e-7).: reviewer3: needs work — RECOMPUTED from model_data npy: mu matrix, col angle 1.043deg, SVs 3.697/1.176/0.0029, cond 1272.9 all reproduce. BUT it is the zero-thi
- [awaiting review] F-037 CORRECTED: spectrum specification is the largest CTsynth-vs-XCIST projection mismatch (about 2.4-3.2%, binning-convention dependent): Agent finding awaiting an independent review
- [awaiting review] F-036 XCIST on Duke SLURM reproduces the laptop reference cell to float32 precision: Agent finding awaiting an independent review
- [awaiting review] F-034 Native Research OS HPC protocol completes a minimal CPU job on Duke SLURM (smoke test): Agent finding awaiting an independent review

## Active decisions

- D-012 (2026-10-13) D-009 is superseded: the CTsynth 'ctsynth prepare|verify|export|import|report' laptop<->HPC CLI no longer exists (removed in 6a98348 when the Research OS native HPC protocol replaced it, D-010). The current workflow is docs/hpc.md: ctsynth preflight/postcheck/bundle-check/run-check guards around research hpc. Phase 3 conventions recorded with it: (1) ONE definition of 'source' for citability, runmeta.SOURCE_PATHS (src tests configs scripts projects pyproject.toml uv.lock), shared by run metadata and preflight; projects/**/results* are outputs; unknown git state is None and never citable; (2) AAPM<->paired axis conversion only through ctsynth.aapm.convert (index relabel, no invented affine); (3) hooks are registered but unapproved/disabled until a human enables them; (4) orchestration stays disabled.
- D-011 (2026-10-12) XCIST is the first independent CT forward-data producer; DukeSim deferred

## Blockers

- [checkpoint] CP-012: Physics-only arm was optimisation-limited (projection RMSE 4.3e-3 vs supervised 2.6e-3). Fixed single support mask limits Exp 8. No independent review done yet; findings not yet registered.

## Project skills

- 14 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md
