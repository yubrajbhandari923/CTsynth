# Research context — CTsynth

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~3782 tokens.
> Generated 2026-10-11T20:22:56-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

Git: `master` @ `6d2fefc3d7` (uncommitted changes)

## Latest checkpoint

**CP-002** Checkpoint — 2026-10-11 (2026-10-11)
- **Understanding:** - SUPERSEDES F-005 (2nd independent review). Spectral mismatch ALWAYS biases a nominal-spectrum estimator, and on the real ray subset every condition is DETECTABLE with an appropriate test - no cell is genuinely 'unbiased'. Noise-free MLE on exact expected counts, 48 P2 rays, 1e6 photons, floor = rms(CRLB)/sqrt(20): dual-kVp iodine tilt +-10% gives rms/floor 0.82-0.92 BUT signed-mean z = +4.6/-5.2 (detectable); dual-kVp iodine energy-scale +-2% rms/floor 3.8-3.9, z = +-22; dual-kVp WATER tilt rms/floor 4.0-4.3, z = +-24 (a large bias the iodine-only criterion hid entirely); 4-bin photon counting on P2 is 21-38x floor. All fits converged. The pre-registered 3x-observed-rms criterion is DEFECTIVE: it compares a noise-inflated rms against an unsubtracted floor and so cannot distinguish 'no bias' from 'bias below the rms detection limit'. A signed-mean-over-rays test is the correct statistic. Supersedes the earlier 'under-powered, not unbiased' framing, which conceded too much to the defective criterion. (F-009, supported)
  - SUPERSEDES F-007 (2nd independent review). Identifiability: STRUCTURAL part (rank <= min(n_meas,k), so 1 measurement cannot separate 2+ materials and dual-kVp cannot separate 3) is arithmetic, not evidence, and holds at any threshold. EMPIRICAL part, on the real ray subset at 1e6 photons: rank verdicts are robust for rank thresholds 1e-3 to 1e-10 (100% full rank) but NOT at 1e-2, where the full-rank fraction collapses to 0% for dual-kVp k=2 and 4-bin k=3 and 58% for 4-bin k=2, because smallest/largest singular-value ratios are only 0.0020-0.0123. (The superseded F-007 wrongly claimed robustness up to 1e-2, contradicting the ratios quoted in its own sentence.) Median condition numbers: dual-kVp k=2 195 (max 506), 4-bin k=2 99, 4-bin k=3 217. At EQUAL basis size k=2 and equal total unattenuated photons, 4-bin photon counting has median iodine CRLB 0.00151 vs dual-kVp 0.00376 g/cm^2 = 2.5x better by median (2.4x by rms aggregation; the ratio is ray-dependent). This compares an idealised energy-weighted 2-exposure design against an idealised 4-window 1-exposure design with NO detector response, charge sharing or pile-up, so it is an upper bound on the photon-counting advantage. (F-010, supported)
  - SUPERSEDES F-002. Empirical Monte-Carlo std agrees with the Cramer-Rao bound to within 12.5% (median over rays; range 0.875-1.106) for the three identifiable cells at 1e6 photons, with a systematic pattern (P2 cells low 0.87-0.95, P3 k=3 cells high 1.05-1.11) rather than scatter about 1. The sampling spread of this check is ~3-4% (median of per-ray std estimates over 26-54 rays, 20 draws each), so the deviation is real, not check noise. This is a SELF-CONSISTENCY check of Jacobian + sampler + MLE under one shared model and table: a wrong attenuation table or spectrum would pass it equally. It is not validation of physical fidelity. (F-006, supported)
  - AMENDS F-003 (scope). The dose-independent misspecification bias is an IODINE LINE-INTEGRAL statistic in g/cm2 (not g/cm3 density), measured on 27 rays of the 3-material phantom with a 2-material estimator. Both designs agree (bias ratio 1e6/1e5: M2 0.89, M4 1.02; 7.0x and 15.0x their noise floors). Dose-independence of misspecification bias is expected from theory, so this is a confirmation that the implementation behaves correctly, not a discovery. (F-008, supported)
  - AMENDS F-009 (two numerical/scope corrections from the 2nd reviewer's verdict check, verified by Opus). (1) The 4-bin photon-counting bias on P2 is 27-43x the floor over all 8 (condition x material) cells, NOT the 21-38x stated in F-009. (2) The dual-kVp iodine TILT case is MARGINAL and must not be quoted as strongly detectable: per-ray bias is only 0.66-0.75 of the floor and the signed-mean z of about 5 arises only by pooling 48 rays. (3) All z values are noise-free hypothetical power computed against a noise-only null, i.e. the bias an infinite-data experiment would see relative to a 20-realisation floor - they are NOT observed test statistics from the Monte-Carlo run. F-009's central claim (mismatch always biases; the rms criterion is defective; dual-kVp water tilt bias is large and was hidden by iodine-only scoring) is unaffected. (F-011, preliminary)
  - Fitting a 2-material basis to 3-material data produces bias that does NOT shrink with photon count (bias ratio 1e6/1e5 = 0.89 and 1.02 versus 0.32 expected for noise-driven error), at 7-15x the noise floor: basis misspecification is a systematic error that more dose cannot fix. (F-003, preliminary)
- **Next step:** EXP-003 (revised by this review): spectral-mismatch bias study driven by DETERMINISTIC noise-free bias as the primary quantity, with Monte-Carlo only to establish detectability. Required design changes: score ALL materials, not iodine alone; pre-compute the detection floor and choose the realisation count from a power analysis BEFORE running; separate gain mismatch from shape mismatch; and compare against a first-order analytic bias prediction. Prerequisite for any inverse-crime-free comparison under H1.

## Current baseline

_none set_

## Active & open questions

- Q-003 [investigating] P01-H2: Does material-map error depend more on spectral separation (kVp pair / photon-counting bins) and noise than on the inference architecture?
- Q-001 [investigating] P01: Can material maps be inferred from dual-energy/spectral CT measurements well enough to seed editable virtual patients?
- Q-005 [open] P01-Q5: On the AAPM DL-Spectral CT benchmark, how do a constrained classical estimator and a small supervised U-Net compare for 3-material decomposition, and what makes the 3-from-2 problem well posed?
- … 2 more (`research question list`)

## Established findings (what we believe)

- F-009 [supported, medium] SUPERSEDES F-005 (2nd independent review). Spectral mismatch ALWAYS biases a nominal-spectrum estimator, and on the real ray subset every condition is DETECTABLE with an appropriate test - no cell is genuinely 'unbiased'. Noise-free MLE on exact expected counts, 48 P2 rays, 1e6 photons, floor = rms(CRLB)/sqrt(20): dual-kVp iodine tilt +-10% gives rms/floor 0.82-0.92 BUT signed-mean z = +4.6/-5.2 (detectable); dual-kVp iodine energy-scale +-2% rms/floor 3.8-3.9, z = +-22; dual-kVp WATER tilt rms/floor 4.0-4.3, z = +-24 (a large bias the iodine-only criterion hid entirely); 4-bin photon counting on P2 is 21-38x floor. All fits converged. The pre-registered 3x-observed-rms criterion is DEFECTIVE: it compares a noise-inflated rms against an unsubtracted floor and so cannot distinguish 'no bias' from 'bias below the rms detection limit'. A signed-mean-over-rays test is the correct statistic. Supersedes the earlier 'under-powered, not unbiased' framing, which conceded too much to the defective criterion. — evidence: EXP-002, RUN-0002
- F-010 [supported, medium] SUPERSEDES F-007 (2nd independent review). Identifiability: STRUCTURAL part (rank <= min(n_meas,k), so 1 measurement cannot separate 2+ materials and dual-kVp cannot separate 3) is arithmetic, not evidence, and holds at any threshold. EMPIRICAL part, on the real ray subset at 1e6 photons: rank verdicts are robust for rank thresholds 1e-3 to 1e-10 (100% full rank) but NOT at 1e-2, where the full-rank fraction collapses to 0% for dual-kVp k=2 and 4-bin k=3 and 58% for 4-bin k=2, because smallest/largest singular-value ratios are only 0.0020-0.0123. (The superseded F-007 wrongly claimed robustness up to 1e-2, contradicting the ratios quoted in its own sentence.) Median condition numbers: dual-kVp k=2 195 (max 506), 4-bin k=2 99, 4-bin k=3 217. At EQUAL basis size k=2 and equal total unattenuated photons, 4-bin photon counting has median iodine CRLB 0.00151 vs dual-kVp 0.00376 g/cm^2 = 2.5x better by median (2.4x by rms aggregation; the ratio is ray-dependent). This compares an idealised energy-weighted 2-exposure design against an idealised 4-window 1-exposure design with NO detector response, charge sharing or pile-up, so it is an upper bound on the photon-counting advantage. — evidence: EXP-002, RUN-0002
- F-006 [supported, medium] SUPERSEDES F-002. Empirical Monte-Carlo std agrees with the Cramer-Rao bound to within 12.5% (median over rays; range 0.875-1.106) for the three identifiable cells at 1e6 photons, with a systematic pattern (P2 cells low 0.87-0.95, P3 k=3 cells high 1.05-1.11) rather than scatter about 1. The sampling spread of this check is ~3-4% (median of per-ray std estimates over 26-54 rays, 20 draws each), so the deviation is real, not check noise. This is a SELF-CONSISTENCY check of Jacobian + sampler + MLE under one shared model and table: a wrong attenuation table or spectrum would pass it equally. It is not validation of physical fidelity. — evidence: EXP-002, RUN-0002
- … 5 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

_none_

## Current experiments

_none_

## Active plans

- PLAN-002 P02 AAPM benchmark: adapter, classical and supervised baselines — 2/5 tasks done · objective: Reproducible benchmark on AAPM DL-Spectral CT: adapter with explicit conventions, deterministic splits, streaming loader, constrained classical baseline, small U-Net, honest evaluation
  - next ready: T-012 Small supervised U-Net (2 CT channels -> 3 material maps) (`research task context T-012`)
  - running: T-011 Classical constrained decomposition baseline (NNLS + simplex constraint, TV variant) (claimed by opus)
- PLAN-001 P01 foundation: spectral forward model and baselines — 4/9 tasks done · objective: Validated spectral forward model + classical baseline decomposition on laptop, then HPC-ready configs
  - next ready: T-005 Classical baseline: image-domain two-material decomposition on toy phantom (`research task context T-005`)
  - BLOCKED: T-004 Read AAPM starting_kit.tgz (103 MB) for spectra/geometry/metric (needs user OK to download)

## Needs attention (verification)

- [awaiting review] F-013 AAPM phantom maps are a TWO-REGIME partition, not a plain simplex (verified directly on the 10 kit sample cases, case 0): the three maps sum to exactly 0 in 35.1% of pixels (air background, no fourth material class), to 1 within 4e-8 in 59.6%, and lie strictly between 0 and 1 in only 5.2% (boundary/partial-volume pixels). Maxima: adipose 1.0, fibroglandular 1.0, calcification 0.566 (10-case sample). A classical estimator must therefore handle air explicitly - imposing sum-to-one globally would be wrong for 35% of pixels - and the 5.2% transition pixels are where a hard two-class assumption will fail. Arrays verified: phantoms/images (N,512,512) float32, transmission (N,256,1024) float32 in (0,1].: Agent finding awaiting an independent review
- [awaiting review] F-012 AAPM 3-material conditioning, computed from the official starting-kit spectra and mu tables (md5-verified kit, 10 sample cases): the spectrum-weighted effective attenuation matrix is [[0.26697,0.31321,3.01182],[0.21930,0.24788,1.64238]] 1/cm for (adipose,fibroglandular,calcification) at 50/80 kVp. Adipose and fibroglandular are spectrally near-degenerate: column cosine 0.999834, i.e. 1.04 degrees apart. The unconstrained 2x3 null direction is [-0.723,+0.691,-0.008], confirming F-010's prediction concretely: adipose can be traded against fibroglandular with essentially no measurement change. Appending the sum-to-one row makes the 3x3 system full rank (singular values 3.697, 1.176, 0.00290) but POORLY CONDITIONED at cond=1273. Therefore the partition constraint restores identifiability but error in the adipose/fibroglandular direction is amplified ~1000x; exact recovery in the noiseless challenge data is possible, while any model mismatch or interpolation error is magnified by that factor. Image-domain linear inversion is expected to be far worse than the published model-based winners (6.8e-7).: Agent finding awaiting an independent review
- [awaiting review] F-011 AMENDS F-009 (two numerical/scope corrections from the 2nd reviewer's verdict check, verified by Opus). (1) The 4-bin photon-counting bias on P2 is 27-43x the floor over all 8 (condition x material) cells, NOT the 21-38x stated in F-009. (2) The dual-kVp iodine TILT case is MARGINAL and must not be quoted as strongly detectable: per-ray bias is only 0.66-0.75 of the floor and the signed-mean z of about 5 arises only by pooling 48 rays. (3) All z values are noise-free hypothetical power computed against a noise-only null, i.e. the bias an infinite-data experiment would see relative to a 20-realisation floor - they are NOT observed test statistics from the Monte-Carlo run. F-009's central claim (mismatch always biases; the rms criterion is defective; dual-kVp water tilt bias is large and was hidden by iodine-only scoring) is unaffected.: Agent finding awaiting an independent review

## Active decisions

- D-005 (2026-10-11) AAPM benchmark design is governed by two facts established in discovery: (1) the challenge data are NOISELESS and the forward model was shared with all participants, so this is an EXACT-RECOVERY problem (winning s1 RMSE 6.8e-7, i.e. ~1e-6 is reachable and is NOT a noise floor) - a denoising-style evaluation would be meaningless; (2) 3-material recovery from 2 spectral measurements is made well posed by a SUM-TO-ONE constraint on VOLUME FRACTIONS (2 spectral equations + 1 partition constraint = 3 equations, unique when the three (mu_low, mu_high) points are non-collinear), not by nonnegativity or priors. Therefore: the classical baseline uses volume fractions with an explicit simplex constraint; nonnegativity alone is insufficient; and our metric must match the published s1 = mean over cases of sqrt(||t-r||^2/(3n)) on dimensionless volume fractions.
- D-004 (2026-10-11) The correct test for a systematic bias measured over many rays is the SIGNED MEAN over rays compared to floor/sqrt(n_rays), not an rms compared to an unsubtracted noise floor. An rms statistic is inflated by noise and cannot distinguish 'no bias' from 'bias below the rms detection limit'; it declared a bias with z=5 to be 'at the noise floor'.

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
