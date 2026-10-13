# Research context — CTsynth

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2878 tokens.
> Generated 2026-10-12T23:11:08-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

Git: `master` @ `07d902868f`

## Latest checkpoint

**CP-007** Native HPC adopted; XCIST integrated; forward-mismatch ladder; E1-E4 registered (2026-10-12)
- **Understanding:** Research OS 0.5.0 native HPC works on Duke SLURM (RUN-0012; D-010); legacy CTsynth orchestration replaced by ctsynth.hpc.check guards. XCIST 1.6.9 installed (laptop + HPC after a CRLF line-ending patch of the hash-verified sdist build script), official example reproduced (RUN-0015), adapter conventions verified against an analytic reference, paired-ct/0.1 schema + 18-record dataset v0 (RUN-0017), staged mismatch ladder (RUN-0023, F-037, preliminary): XCIST-ingredient floor 0.26-0.47% (voxel chords), spectrum specification the largest term (~2.4-3.2%, convention-sensitive), attenuation tables ~0.1-0.2%, detector <=0.25% relative. HPC replication of one cell matches laptop to 9.7e-6 in -ln units (RUN-0022, F-036). F-035 was superseded after review found a binning bug. NOT done: photon-counting, scatter, learning experiments (E1-E4 registered as proposed, splits declared in docs/xcist-experiments.md), DukeSim.
- **Next step:** T-032 dataset v1 + split file + power analysis, then E1/E4; T-033 review of F-037.

## Current baseline

_none set_

## Active & open questions

- Q-006 [investigating] P02-Q6: Can a network recover physically meaningful material maps from spectral measurement consistency ALONE (no paired labels)? Prior analysis (F-014) says NO for the AAPM 2-measurement/3-material case: the null direction is invisible to the data, so a pure consistency loss is minimised by a one-parameter family containing the truth and infinitely many wrong maps. The scientific content is therefore WHICH added prior selects the truth, and whether it does so for physical reasons or by accident.
- Q-005 [investigating] P01-Q5: On the AAPM DL-Spectral CT benchmark, how do a constrained classical estimator and a small supervised U-Net compare for 3-material decomposition, and what makes the 3-from-2 problem well posed?
- Q-003 [investigating] P01-H2: Does material-map error depend more on spectral separation (kVp pair / photon-counting bins) and noise than on the inference architecture?
- … 3 more (`research question list`)

## Established findings (what we believe)

- F-021 [supported, medium] PENG 2024 IS NOT A LABEL-FREE COMPARATOR. Loss L = lambda*||A[x1;x2] - [mu_H;mu_L]||^2 + L_GAN, image-domain on linear attenuation coefficients with a LINEAR 2x2 per-pixel matrix A from ROI means: no spectrum, no projector, no Poisson term. Because A is square and invertible the fidelity minimum is exactly the noisy direct inversion, so the GAN term supplies ALL the noise suppression (SD -91-97%). The discriminator requires an UNPAIRED reference set of clean material maps built by thresholding SECT/high-energy images from other patients, so the method carries a learned prior fitted to label-like data: it is 'unpaired-reference adversarial', not label-free. Evaluation ground truth uses the same thresholding recipe, so the reference prior is in-distribution with the evaluation - a self-favouring protocol. Like Hellwege it is 2-from-2 (determined, no sum-to-one row) and so says nothing about our 3-from-2 null space. Usable only as an unpaired-reference-prior baseline on a 2-material subproblem, and must be labelled as such. Unverified: published lambda (supplement not obtained; 1e4/1e2 from arXiv v1 only), exact GAN loss form, hyperparameter protocol; no code released. — evidence: Q-006
- F-018 [supported, medium] EXPERIMENT A (deterministic, no training): a pure measurement-consistency loss CANNOT identify 3 material maps from 2 spectral measurements on the AAPM model, and this is not a linearisation artifact. Using the REAL polychromatic forward model and the official CC-BY spectra/mu tables, I solved for alternative material path-length vectors that reproduce BOTH kVp transmissions to machine precision: holding calcification at 0.05/0.06/0.08/0.10 cm and re-solving adipose+fibroglandular by Newton gives exact-measurement decoys with relative transmission error <= 1.8e-16 and ||dx|| up to 6.53 cm (e.g. truth [8.0,4.0,0.05] vs decoy [12.72,-0.51,0.10]). The decoy family is one-dimensional, as predicted by the 2x3 rank. Consequence: ANY label-free objective whose only signal is measurement consistency has a continuum of exact global minima; a network trained on it converges to whichever member its implicit priors (architecture, initialisation, amortisation) prefer, NOT to the truth. Reporting a low consistency loss as evidence of material accuracy is therefore invalid for this problem. — evidence: Q-006
- F-020 [supported, medium] HELLWEGE 2026 IS NOT A PRECEDENT FOR OUR PROBLEM (full arXiv PDF text read). Their task is 2 basis materials (soft tissue, calcification) from 2 spectra: SQUARE and determined by counting, so pixelwise identifiability is automatic and never discussed in the paper. Our AAPM task is 3 materials from 2 spectra: underdetermined, and in their Eq. 6 objective L = (1/|D|)(1/M) sum_i [||p_LE - phat_LE||^2 + ||p_HE - phat_HE||^2] (unweighted MSE on log line integrals, no I0, no Poisson term, NO regulariser, no nonnegativity, no sum-to-one, no TV, no entropy, no softmax) our F-014 null direction would be EXACTLY FLAT. The paper reports no ablations at all; its gain over MBIR (VMI RMSE 5.83 vs 13.07 HU at 50 keV) comes from the network and training, not from the loss. The authors state the inverse crime explicitly (Sec 4.2: same polychromatic model generates and decodes) and run no mismatch experiment. Early stopping uses projection-domain validation loss with no ground truth, which is a genuinely label-free protocol worth copying. Unverified: energy-bin count, final activation, how 3+ AAPM tissue classes were merged into 2, whether data were noisy, and Table 1 mean-vs-median. — evidence: Q-006
- … 23 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

_none_

## Current experiments

- EXP-010 [needs_review] XCIST vs CTsynth differentiable forward model: staged mismatch on known-material phantoms — runs 4
- EXP-014 [proposed] E2 physics-consistent self-supervised material inference on XCIST measurements — runs 0
- EXP-015 [proposed] E3 supervised pretraining plus physics-consistent adaptation under changed acquisition — runs 0

## Active plans

- PLAN-003 P03 label-free physics-consistent decomposition (B0-B4 ladder) — 2/2 tasks done · objective: Implement and fairly compare B0 classical constrained, B1 regularized classical, B2 supervised U-Net, B3 label-free physics-consistent U-Net, B4 physics+priors; ablate priors individually; test identifiability, information, noise, mismatch, generalization
- PLAN-002 P02 AAPM benchmark: adapter, classical and supervised baselines — 7/25 tasks done · objective: Reproducible benchmark on AAPM DL-Spectral CT: adapter with explicit conventions, deterministic splits, streaming loader, constrained classical baseline, small U-Net, honest evaluation
  - next ready: T-017 P1.1 Acquire + validate the real AAPM 1000-case dataset (`research task context T-017`)
- PLAN-001 P01 foundation: spectral forward model and baselines — 4/9 tasks done · objective: Validated spectral forward model + classical baseline decomposition on laptop, then HPC-ready configs
  - next ready: T-005 Classical baseline: image-domain two-material decomposition on toy phantom (`research task context T-005`)
  - BLOCKED: T-004 Read AAPM starting_kit.tgz (103 MB) for spectra/geometry/metric (needs user OK to download)

## Needs attention (verification)

- [reviewer disagrees] F-012 AAPM 3-material conditioning, computed from the official starting-kit spectra and mu tables (md5-verified kit, 10 sample cases): the spectrum-weighted effective attenuation matrix is [[0.26697,0.31321,3.01182],[0.21930,0.24788,1.64238]] 1/cm for (adipose,fibroglandular,calcification) at 50/80 kVp. Adipose and fibroglandular are spectrally near-degenerate: column cosine 0.999834, i.e. 1.04 degrees apart. The unconstrained 2x3 null direction is [-0.723,+0.691,-0.008], confirming F-010's prediction concretely: adipose can be traded against fibroglandular with essentially no measurement change. Appending the sum-to-one row makes the 3x3 system full rank (singular values 3.697, 1.176, 0.00290) but POORLY CONDITIONED at cond=1273. Therefore the partition constraint restores identifiability but error in the adipose/fibroglandular direction is amplified ~1000x; exact recovery in the noiseless challenge data is possible, while any model mismatch or interpolation error is magnified by that factor. Image-domain linear inversion is expected to be far worse than the published model-based winners (6.8e-7).: reviewer3: needs work — RECOMPUTED from model_data npy: mu matrix, col angle 1.043deg, SVs 3.697/1.176/0.0029, cond 1272.9 all reproduce. BUT it is the zero-thi
- [reviewer disagrees] F-017 PROCESS (D-004 follow-up): when the provenance system itself writes into the repo during a run, a whole-tree git_dirty flag is ALWAYS true and therefore useless as a citability signal - it flagged RUN-0003/4/5 as non-citable purely because Research OS was logging the run. ctsynth.runmeta now records git_dirty_source (src/tests/configs only) alongside git_dirty; citability depends on the SOURCE being clean. Experiment outputs were also untracked (projects/*/results*/ gitignored) because writing results into a tracked directory dirties the tree as the run executes. RUN-0006 is the first run with git_dirty_source=False.: reviewer3: needs work — Overclaim: 'whole-tree git_dirty flag is ALWAYS true' - RUN-0006 record shows git dirty=False (RUN-0003 True). Process claim; could not 
- [awaiting review] F-037 CORRECTED: spectrum specification is the largest CTsynth-vs-XCIST projection mismatch (about 2.4-3.2%, binning-convention dependent): Agent finding awaiting an independent review
- [awaiting review] F-036 XCIST on Duke SLURM reproduces the laptop reference cell to float32 precision: Agent finding awaiting an independent review
- [awaiting review] F-034 Native Research OS HPC protocol completes a minimal CPU job on Duke SLURM (smoke test): Agent finding awaiting an independent review

## ⚠ Needs synthesis before new experiments

- EXP-010 XCIST vs CTsynth differentiable forward model: staged mismatch on known-material phantoms (0 unsynthesized runs)

## Active decisions

- D-011 (2026-10-12) XCIST is the first independent CT forward-data producer; DukeSim deferred
- D-010 (2026-10-12) Adopt Research OS 0.5.0 native HPC protocol; CTsynth keeps thin preflight/postcheck guards

## Blockers

_none_

## Project skills

- 9 more skills: `research skills list`, load one with `research skill show NAME`

## Agent policy

- max runs without synthesis: 5
- max failed runs without review: 3
- synthesis required before new experiment: True
- findings: agents create them preliminary; an independent review (`research finding review`) makes them supported
- tasks: `research task done` runs the task's checks; failures block completion
- Protocol: see AGENTS.md
