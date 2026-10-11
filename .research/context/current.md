# Research context — CTsynth

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2400 tokens.
> Generated 2026-10-11T19:54:00-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

Git: `master` @ `cf3b880689` (uncommitted changes)

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

- F-007 [preliminary] SUPERSEDES F-001. Identifiability separates into a structural part and an empirical part. STRUCTURAL (forced by matrix shape, not an experimental result): with n_meas measurements and k unknown materials, rank <= min(n_meas,k), so 1 energy-integrating measurement cannot separate 2+ materials and dual-kVp cannot separate 3. EMPIRICAL (the real content): among designs that ARE identifiable, conditioning differs substantially and is dose-independent - dual-kVp 2-material median condition number 195 (max 506), 4-bin photon counting 2-material 99, 4-bin 3-material 217. Rank verdicts are robust: smallest/largest singular-value ratios are 2e-3 to 1e-2, so any threshold from 1e-2 to 1e-10 gives identical ranks. Noise cost at equal total photons, SAME basis size (k=2, iodine CRLB): 4-bin 0.0019 vs dual-kVp 0.0047 g/cm2, i.e. photon counting is 2.5x BETTER. The previously quoted 2.70x 'worse' figure compared M4 k=3 against M2 k=2 and confounded design with basis size; it is the price of estimating a third material, not a penalty of photon counting. — evidence: EXP-002, RUN-0002
- F-008 [preliminary] AMENDS F-003 (scope). The dose-independent misspecification bias is an IODINE LINE-INTEGRAL statistic in g/cm2 (not g/cm3 density), measured on 27 rays of the 3-material phantom with a 2-material estimator. Both designs agree (bias ratio 1e6/1e5: M2 0.89, M4 1.02; 7.0x and 15.0x their noise floors). Dose-independence of misspecification bias is expected from theory, so this is a confirmation that the implementation behaves correctly, not a discovery. — evidence: EXP-002, RUN-0002
- F-006 [preliminary] SUPERSEDES F-002. Empirical Monte-Carlo std agrees with the Cramer-Rao bound to within 12.5% (median over rays; range 0.875-1.106) for the three identifiable cells at 1e6 photons, with a systematic pattern (P2 cells low 0.87-0.95, P3 k=3 cells high 1.05-1.11) rather than scatter about 1. The sampling spread of this check is ~3-4% (median of per-ray std estimates over 26-54 rays, 20 draws each), so the deviation is real, not check noise. This is a SELF-CONSISTENCY check of Jacobian + sampler + MLE under one shared model and table: a wrong attenuation table or spectrum would pass it equally. It is not validation of physical fidelity. — evidence: EXP-002, RUN-0002
- F-005 [preliminary] SUPERSEDES F-004. Spectral mismatch always produces genuine bias; the apparent 'no bias' cells were a detection-power and scope artifact of the pre-registered criterion. Noise-free calculation (MLE on EXACT expected counts from perturbed spectra, zero Poisson noise, 20cm water + 0.2 g/cm2 iodine): every condition gives nonzero bias. Relative to the 20-realisation detection floor (CRLB/sqrt(20)) at 1e6 photons: dual-kVp iodine tilt +-10% = 0.93-1.05x (genuinely at the floor: UNDER-POWERED, not unbiased), dual-kVp iodine energy-scale +-2% = 4.0-4.2x (CLEARLY BIASED: the original F-004 wording was wrong for this case), dual-kVp WATER tilt = 3.8-4.1x (the iodine-only criterion hid a large water bias), 4-bin photon counting = 9-21x on the 2-material phantom. Two defects in the pre-registered criterion: it scored iodine only, and it required 3x the floor at BOTH dose levels so the noisier 1e5 level dominated the verdict. — evidence: EXP-002, RUN-0002
- F-003 [preliminary] Fitting a 2-material basis to 3-material data produces bias that does NOT shrink with photon count (bias ratio 1e6/1e5 = 0.89 and 1.02 versus 0.32 expected for noise-driven error), at 7-15x the noise floor: basis misspecification is a systematic error that more dose cannot fix. — evidence: EXP-002, RUN-0002

## Failed directions (do not repeat without a new reason)

_none_

## Current experiments

_none_

## Active plans

- PLAN-001 P01 foundation: spectral forward model and baselines — 4/9 tasks done · objective: Validated spectral forward model + classical baseline decomposition on laptop, then HPC-ready configs
  - next ready: T-005 Classical baseline: image-domain two-material decomposition on toy phantom (`research task context T-005`)
  - BLOCKED: T-004 Read AAPM starting_kit.tgz (103 MB) for spectra/geometry/metric (needs user OK to download)

## Needs attention (verification)

- [awaiting review] F-008 AMENDS F-003 (scope). The dose-independent misspecification bias is an IODINE LINE-INTEGRAL statistic in g/cm2 (not g/cm3 density), measured on 27 rays of the 3-material phantom with a 2-material estimator. Both designs agree (bias ratio 1e6/1e5: M2 0.89, M4 1.02; 7.0x and 15.0x their noise floors). Dose-independence of misspecification bias is expected from theory, so this is a confirmation that the implementation behaves correctly, not a discovery.: Agent finding awaiting an independent review
- [awaiting review] F-007 SUPERSEDES F-001. Identifiability separates into a structural part and an empirical part. STRUCTURAL (forced by matrix shape, not an experimental result): with n_meas measurements and k unknown materials, rank <= min(n_meas,k), so 1 energy-integrating measurement cannot separate 2+ materials and dual-kVp cannot separate 3. EMPIRICAL (the real content): among designs that ARE identifiable, conditioning differs substantially and is dose-independent - dual-kVp 2-material median condition number 195 (max 506), 4-bin photon counting 2-material 99, 4-bin 3-material 217. Rank verdicts are robust: smallest/largest singular-value ratios are 2e-3 to 1e-2, so any threshold from 1e-2 to 1e-10 gives identical ranks. Noise cost at equal total photons, SAME basis size (k=2, iodine CRLB): 4-bin 0.0019 vs dual-kVp 0.0047 g/cm2, i.e. photon counting is 2.5x BETTER. The previously quoted 2.70x 'worse' figure compared M4 k=3 against M2 k=2 and confounded design with basis size; it is the price of estimating a third material, not a penalty of photon counting.: Agent finding awaiting an independent review
- [awaiting review] F-006 SUPERSEDES F-002. Empirical Monte-Carlo std agrees with the Cramer-Rao bound to within 12.5% (median over rays; range 0.875-1.106) for the three identifiable cells at 1e6 photons, with a systematic pattern (P2 cells low 0.87-0.95, P3 k=3 cells high 1.05-1.11) rather than scatter about 1. The sampling spread of this check is ~3-4% (median of per-ray std estimates over 26-54 rays, 20 draws each), so the deviation is real, not check noise. This is a SELF-CONSISTENCY check of Jacobian + sampler + MLE under one shared model and table: a wrong attenuation table or spectrum would pass it equally. It is not validation of physical fidelity.: Agent finding awaiting an independent review

## Active decisions

- D-003 (2026-10-11) Pre-registered success criteria must score all materials, state a detection floor from a power analysis computed before running, avoid requiring significance at every condition, separate structural from empirical results, and avoid confounding two factors. Superseding findings require their own independent review.
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
