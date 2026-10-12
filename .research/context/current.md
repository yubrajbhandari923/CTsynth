# Research context — CTsynth

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~4240 tokens.
> Generated 2026-10-11T20:35:04-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

Git: `master` @ `c4b043bdc0`

## Latest checkpoint

**CP-003** Checkpoint — 2026-10-11 (2026-10-11)
- **Understanding:** - SUPERSEDES F-005 (2nd independent review). Spectral mismatch ALWAYS biases a nominal-spectrum estimator, and on the real ray subset every condition is DETECTABLE with an appropriate test - no cell is genuinely 'unbiased'. Noise-free MLE on exact expected counts, 48 P2 rays, 1e6 photons, floor = rms(CRLB)/sqrt(20): dual-kVp iodine tilt +-10% gives rms/floor 0.82-0.92 BUT signed-mean z = +4.6/-5.2 (detectable); dual-kVp iodine energy-scale +-2% rms/floor 3.8-3.9, z = +-22; dual-kVp WATER tilt rms/floor 4.0-4.3, z = +-24 (a large bias the iodine-only criterion hid entirely); 4-bin photon counting on P2 is 21-38x floor. All fits converged. The pre-registered 3x-observed-rms criterion is DEFECTIVE: it compares a noise-inflated rms against an unsubtracted floor and so cannot distinguish 'no bias' from 'bias below the rms detection limit'. A signed-mean-over-rays test is the correct statistic. Supersedes the earlier 'under-powered, not unbiased' framing, which conceded too much to the defective criterion. (F-009, supported)
  - SUPERSEDES F-007 (2nd independent review). Identifiability: STRUCTURAL part (rank <= min(n_meas,k), so 1 measurement cannot separate 2+ materials and dual-kVp cannot separate 3) is arithmetic, not evidence, and holds at any threshold. EMPIRICAL part, on the real ray subset at 1e6 photons: rank verdicts are robust for rank thresholds 1e-3 to 1e-10 (100% full rank) but NOT at 1e-2, where the full-rank fraction collapses to 0% for dual-kVp k=2 and 4-bin k=3 and 58% for 4-bin k=2, because smallest/largest singular-value ratios are only 0.0020-0.0123. (The superseded F-007 wrongly claimed robustness up to 1e-2, contradicting the ratios quoted in its own sentence.) Median condition numbers: dual-kVp k=2 195 (max 506), 4-bin k=2 99, 4-bin k=3 217. At EQUAL basis size k=2 and equal total unattenuated photons, 4-bin photon counting has median iodine CRLB 0.00151 vs dual-kVp 0.00376 g/cm^2 = 2.5x better by median (2.4x by rms aggregation; the ratio is ray-dependent). This compares an idealised energy-weighted 2-exposure design against an idealised 4-window 1-exposure design with NO detector response, charge sharing or pile-up, so it is an upper bound on the photon-counting advantage. (F-010, supported)
  - SUPERSEDES F-002. Empirical Monte-Carlo std agrees with the Cramer-Rao bound to within 12.5% (median over rays; range 0.875-1.106) for the three identifiable cells at 1e6 photons, with a systematic pattern (P2 cells low 0.87-0.95, P3 k=3 cells high 1.05-1.11) rather than scatter about 1. The sampling spread of this check is ~3-4% (median of per-ray std estimates over 26-54 rays, 20 draws each), so the deviation is real, not check noise. This is a SELF-CONSISTENCY check of Jacobian + sampler + MLE under one shared model and table: a wrong attenuation table or spectrum would pass it equally. It is not validation of physical fidelity. (F-006, supported)
  - AMENDS F-003 (scope). The dose-independent misspecification bias is an IODINE LINE-INTEGRAL statistic in g/cm2 (not g/cm3 density), measured on 27 rays of the 3-material phantom with a 2-material estimator. Both designs agree (bias ratio 1e6/1e5: M2 0.89, M4 1.02; 7.0x and 15.0x their noise floors). Dose-independence of misspecification bias is expected from theory, so this is a confirmation that the implementation behaves correctly, not a discovery. (F-008, supported)
  - PROCESS (D-004 follow-up): when the provenance system itself writes into the repo during a run, a whole-tree git_dirty flag is ALWAYS true and therefore useless as a citability signal - it flagged RUN-0003/4/5 as non-citable purely because Research OS was logging the run. ctsynth.runmeta now records git_dirty_source (src/tests/configs only) alongside git_dirty; citability depends on the SOURCE being clean. Experiment outputs were also untracked (projects/*/results*/ gitignored) because writing results into a tracked directory dirties the tree as the run executes. RUN-0006 is the first run with git_dirty_source=False. (F-017, preliminary)
  - On SYNTHETIC AAPM-schema data (EXP-003/RUN-0003, 17 val cases, 32x32, noiseless), an exact-on-tissue estimator is WORSE THAN PREDICTING NOTHING by the official aggregate metric. classical_sum_to_one recovers tissue pixels to RMSE 1.29e-6 (near machine precision, as theory predicts for noiseless data) yet scores s1 = 1.179, far worse than the no-training zero_prediction reference (s1 = 0.430), because the 5-9% boundary band carries RMSE 3.84 with individual values up to 11.5 volume fraction (grossly unphysical). Simplex projection clips those outliers and wins (s1 = 0.0508) WITHOUT improving the model: its tissue RMSE is the same 1.2e-6. CONSEQUENCE: a pooled RMSE over a two-regime phantom is dominated by the smallest, most mis-modelled population, so reporting s1 alone would rank a clipped estimator above an exact one for the wrong reason. Regime-separated reporting is mandatory for this benchmark. Caveat: synthetic data only; boundary-band fraction and geometry differ from real AAPM data. (F-016, preliminary)
  - REFINES F-013 (boundary pixels are the hard case, not merely a partial-volume nuisance). In the two-regime AAPM partition, pixels whose three fractions sum strictly between 0 and 1 (~5% of the image) are mis-modelled by a sum-to-one estimator: the unit-sum row CONTRADICTS the measurement there, and that contradiction is amplified by the ~1273 conditioning, so the resulting error is far LARGER than the sum deficit (1 - sum) itself. Simplex projection only clips the result and does not fix the model error. Consequence for the benchmark: boundary-band error must be reported separately (it is, in the evaluation record), and an air/tissue decision rule is a MODELLING CHOICE, not a fact recoverable from the data - the implementation uses AIR_CUTOFF_PER_CM = 0.1 (half the smallest tissue mu from F-012), which is untested on real AAPM data. (F-015, preliminary)
  - F-012's null space is now demonstrated OPERATIONALLY, not just algebraically (synthetic data, src/ctsynth/aapm/classical.py + tests/aapm/test_classical.py). Unconstrained per-pixel inversion of the 2x3 AAPM system genuinely fails (max per-pixel error > 0.05 in volume fraction) and its error vector is parallel to the SVD null direction to <1e-6 deg; that independently-computed null vector lies 0.024 deg from the published F-012 vector [-0.723,+0.691,-0.008]. Crucially the error is INVISIBLE to the data: |e @ mu^T| < 1e-12, i.e. the wrong answer reproduces the measurements exactly. Adding the sum-to-one row restores a unique solution (exact recovery to ~1e-10 in float64 on noiseless synthetic tissue pixels) but amplifies measurement perturbation: the derived worst-direction amplification sigma_max(B^-1[:,:2])*|d|/|x| = 181 for a typical 50/50 adipose/fibroglandular pixel, matched to rel 1e-4, which is 0.14x cond(B)=1273 (below cond because the appended sum row is exact and unperturbed). (F-014, preliminary)
- **Next step:** T-012 small U-Net on the same synthetic harness, evaluated against zero_prediction and classical_simplex on the SAME val split, with the test split untouched. Then the fan-beam projector needed for transmission-domain methods (the regime the published winners used).

## Current baseline

_none set_

## Active & open questions

- Q-005 [investigating] P01-Q5: On the AAPM DL-Spectral CT benchmark, how do a constrained classical estimator and a small supervised U-Net compare for 3-material decomposition, and what makes the 3-from-2 problem well posed?
- Q-003 [investigating] P01-H2: Does material-map error depend more on spectral separation (kVp pair / photon-counting bins) and noise than on the inference architecture?
- Q-001 [investigating] P01: Can material maps be inferred from dual-energy/spectral CT measurements well enough to seed editable virtual patients?
- … 2 more (`research question list`)

## Established findings (what we believe)

- F-009 [supported, medium] SUPERSEDES F-005 (2nd independent review). Spectral mismatch ALWAYS biases a nominal-spectrum estimator, and on the real ray subset every condition is DETECTABLE with an appropriate test - no cell is genuinely 'unbiased'. Noise-free MLE on exact expected counts, 48 P2 rays, 1e6 photons, floor = rms(CRLB)/sqrt(20): dual-kVp iodine tilt +-10% gives rms/floor 0.82-0.92 BUT signed-mean z = +4.6/-5.2 (detectable); dual-kVp iodine energy-scale +-2% rms/floor 3.8-3.9, z = +-22; dual-kVp WATER tilt rms/floor 4.0-4.3, z = +-24 (a large bias the iodine-only criterion hid entirely); 4-bin photon counting on P2 is 21-38x floor. All fits converged. The pre-registered 3x-observed-rms criterion is DEFECTIVE: it compares a noise-inflated rms against an unsubtracted floor and so cannot distinguish 'no bias' from 'bias below the rms detection limit'. A signed-mean-over-rays test is the correct statistic. Supersedes the earlier 'under-powered, not unbiased' framing, which conceded too much to the defective criterion. — evidence: EXP-002, RUN-0002
- F-010 [supported, medium] SUPERSEDES F-007 (2nd independent review). Identifiability: STRUCTURAL part (rank <= min(n_meas,k), so 1 measurement cannot separate 2+ materials and dual-kVp cannot separate 3) is arithmetic, not evidence, and holds at any threshold. EMPIRICAL part, on the real ray subset at 1e6 photons: rank verdicts are robust for rank thresholds 1e-3 to 1e-10 (100% full rank) but NOT at 1e-2, where the full-rank fraction collapses to 0% for dual-kVp k=2 and 4-bin k=3 and 58% for 4-bin k=2, because smallest/largest singular-value ratios are only 0.0020-0.0123. (The superseded F-007 wrongly claimed robustness up to 1e-2, contradicting the ratios quoted in its own sentence.) Median condition numbers: dual-kVp k=2 195 (max 506), 4-bin k=2 99, 4-bin k=3 217. At EQUAL basis size k=2 and equal total unattenuated photons, 4-bin photon counting has median iodine CRLB 0.00151 vs dual-kVp 0.00376 g/cm^2 = 2.5x better by median (2.4x by rms aggregation; the ratio is ray-dependent). This compares an idealised energy-weighted 2-exposure design against an idealised 4-window 1-exposure design with NO detector response, charge sharing or pile-up, so it is an upper bound on the photon-counting advantage. — evidence: EXP-002, RUN-0002
- F-006 [supported, medium] SUPERSEDES F-002. Empirical Monte-Carlo std agrees with the Cramer-Rao bound to within 12.5% (median over rays; range 0.875-1.106) for the three identifiable cells at 1e6 photons, with a systematic pattern (P2 cells low 0.87-0.95, P3 k=3 cells high 1.05-1.11) rather than scatter about 1. The sampling spread of this check is ~3-4% (median of per-ray std estimates over 26-54 rays, 20 draws each), so the deviation is real, not check noise. This is a SELF-CONSISTENCY check of Jacobian + sampler + MLE under one shared model and table: a wrong attenuation table or spectrum would pass it equally. It is not validation of physical fidelity. — evidence: EXP-002, RUN-0002
- … 9 more (`research finding list`)

## Failed directions (do not repeat without a new reason)

_none_

## Current experiments

_none_

## Active plans

- PLAN-002 P02 AAPM benchmark: adapter, classical and supervised baselines — 4/5 tasks done · objective: Reproducible benchmark on AAPM DL-Spectral CT: adapter with explicit conventions, deterministic splits, streaming loader, constrained classical baseline, small U-Net, honest evaluation
  - next ready: T-012 Small supervised U-Net (2 CT channels -> 3 material maps) (`research task context T-012`)
- PLAN-001 P01 foundation: spectral forward model and baselines — 4/9 tasks done · objective: Validated spectral forward model + classical baseline decomposition on laptop, then HPC-ready configs
  - next ready: T-005 Classical baseline: image-domain two-material decomposition on toy phantom (`research task context T-005`)
  - BLOCKED: T-004 Read AAPM starting_kit.tgz (103 MB) for spectra/geometry/metric (needs user OK to download)

## Needs attention (verification)

- [awaiting review] F-017 PROCESS (D-004 follow-up): when the provenance system itself writes into the repo during a run, a whole-tree git_dirty flag is ALWAYS true and therefore useless as a citability signal - it flagged RUN-0003/4/5 as non-citable purely because Research OS was logging the run. ctsynth.runmeta now records git_dirty_source (src/tests/configs only) alongside git_dirty; citability depends on the SOURCE being clean. Experiment outputs were also untracked (projects/*/results*/ gitignored) because writing results into a tracked directory dirties the tree as the run executes. RUN-0006 is the first run with git_dirty_source=False.: Agent finding awaiting an independent review
- [awaiting review] F-016 On SYNTHETIC AAPM-schema data (EXP-003/RUN-0003, 17 val cases, 32x32, noiseless), an exact-on-tissue estimator is WORSE THAN PREDICTING NOTHING by the official aggregate metric. classical_sum_to_one recovers tissue pixels to RMSE 1.29e-6 (near machine precision, as theory predicts for noiseless data) yet scores s1 = 1.179, far worse than the no-training zero_prediction reference (s1 = 0.430), because the 5-9% boundary band carries RMSE 3.84 with individual values up to 11.5 volume fraction (grossly unphysical). Simplex projection clips those outliers and wins (s1 = 0.0508) WITHOUT improving the model: its tissue RMSE is the same 1.2e-6. CONSEQUENCE: a pooled RMSE over a two-regime phantom is dominated by the smallest, most mis-modelled population, so reporting s1 alone would rank a clipped estimator above an exact one for the wrong reason. Regime-separated reporting is mandatory for this benchmark. Caveat: synthetic data only; boundary-band fraction and geometry differ from real AAPM data.: Agent finding awaiting an independent review
- [awaiting review] F-015 REFINES F-013 (boundary pixels are the hard case, not merely a partial-volume nuisance). In the two-regime AAPM partition, pixels whose three fractions sum strictly between 0 and 1 (~5% of the image) are mis-modelled by a sum-to-one estimator: the unit-sum row CONTRADICTS the measurement there, and that contradiction is amplified by the ~1273 conditioning, so the resulting error is far LARGER than the sum deficit (1 - sum) itself. Simplex projection only clips the result and does not fix the model error. Consequence for the benchmark: boundary-band error must be reported separately (it is, in the evaluation record), and an air/tissue decision rule is a MODELLING CHOICE, not a fact recoverable from the data - the implementation uses AIR_CUTOFF_PER_CM = 0.1 (half the smallest tissue mu from F-012), which is untested on real AAPM data.: Agent finding awaiting an independent review

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
