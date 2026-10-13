# Research context — CTsynth

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2587 tokens.
> Generated 2026-10-13T00:33:32-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

Git: `master` @ `45c3fe99f6` (uncommitted changes)

## Latest checkpoint

**CP-010** Phase 2 architecture audit complete (design only; no code changed) (2026-10-13)
- **Understanding:** Scientific state unchanged since CP-009. Audit done: see docs/architecture-workflow-audit.md and the pinned Phase 2 note. Strategy sonnet_first is set but orchestration.enabled=false; no hooks registered; PLAN-004 holds the Phase 3 backlog T-043..T-060.
- **Current problem:** Provenance guards disagree (T-043); orchestration inert (T-053); hooks need human approval (T-054); 14 findings still await independent review (T-033, T-037).
- **Next step:** Human: decide the listed items (enable orchestration, approve hooks, T-042, T-052). Scientific line unchanged: T-032 then T-039, T-038, T-033 in parallel. Land T-043 before the next experiment run. Re-preview bundle before every hpc submit.

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

- EXP-015 [proposed] E3 supervised pretraining plus physics-consistent adaptation under changed acquisition — runs 0
- EXP-014 [proposed] E2 physics-consistent self-supervised material inference on XCIST measurements — runs 0
- EXP-016 [proposed] E4 identifiability and prior comparison on XCIST data (deterministic first) — runs 0

## Active plans

- PLAN-004 P04 Phase 2 audit and Phase 3 refactoring backlog — 1/18 tasks done · objective: Execute the audit roadmap (docs/architecture-workflow-audit.md) without changing scientific results, the HPC baseline or history
  - next ready: T-044 Replace duplicated sha256 helpers with ctsynth.hashing.sha256_file (D1) (`research task context T-044`)
- PLAN-003 P03 label-free physics-consistent decomposition (B0-B4 ladder) — 2/2 tasks done · objective: Implement and fairly compare B0 classical constrained, B1 regularized classical, B2 supervised U-Net, B3 label-free physics-consistent U-Net, B4 physics+priors; ablate priors individually; test identifiability, information, noise, mismatch, generalization
- PLAN-002 P02 AAPM benchmark: adapter, classical and supervised baselines — 7/31 tasks done · objective: Reproducible benchmark on AAPM DL-Spectral CT: adapter with explicit conventions, deterministic splits, streaming loader, constrained classical baseline, small U-Net, honest evaluation
  - next ready: T-017 P1.1 Acquire + validate the real AAPM 1000-case dataset (`research task context T-017`)

## Needs attention (verification)

- [reviewer disagrees] F-012 AAPM 3-material conditioning, computed from the official starting-kit spectra and mu tables (md5-verified kit, 10 sample cases): the spectrum-weighted effective attenuation matrix is [[0.26697,0.31321,3.01182],[0.21930,0.24788,1.64238]] 1/cm for (adipose,fibroglandular,calcification) at 50/80 kVp. Adipose and fibroglandular are spectrally near-degenerate: column cosine 0.999834, i.e. 1.04 degrees apart. The unconstrained 2x3 null direction is [-0.723,+0.691,-0.008], confirming F-010's prediction concretely: adipose can be traded against fibroglandular with essentially no measurement change. Appending the sum-to-one row makes the 3x3 system full rank (singular values 3.697, 1.176, 0.00290) but POORLY CONDITIONED at cond=1273. Therefore the partition constraint restores identifiability but error in the adipose/fibroglandular direction is amplified ~1000x; exact recovery in the noiseless challenge data is possible, while any model mismatch or interpolation error is magnified by that factor. Image-domain linear inversion is expected to be far worse than the published model-based winners (6.8e-7).: reviewer3: needs work — RECOMPUTED from model_data npy: mu matrix, col angle 1.043deg, SVs 3.697/1.176/0.0029, cond 1272.9 all reproduce. BUT it is the zero-thi
- [awaiting review] F-037 CORRECTED: spectrum specification is the largest CTsynth-vs-XCIST projection mismatch (about 2.4-3.2%, binning-convention dependent): Agent finding awaiting an independent review
- [awaiting review] F-036 XCIST on Duke SLURM reproduces the laptop reference cell to float32 precision: Agent finding awaiting an independent review
- [awaiting review] F-034 Native Research OS HPC protocol completes a minimal CPU job on Duke SLURM (smoke test): Agent finding awaiting an independent review

## Active decisions

- D-011 (2026-10-12) XCIST is the first independent CT forward-data producer; DukeSim deferred
- D-010 (2026-10-12) Adopt Research OS 0.5.0 native HPC protocol; CTsynth keeps thin preflight/postcheck guards

## Blockers

- [checkpoint] CP-010: Provenance guards disagree (T-043); orchestration inert (T-053); hooks need human approval (T-054); 14 findings still await independent review (T-033, T-037).

## Project skills

- 9 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md
