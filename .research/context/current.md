# Research context — CTsynth

> Generated from .research/ by `research current`. Do not edit; it is overwritten. ~2504 tokens.
> Generated 2026-10-11T19:58:58-04:00. Details: `research show <ID>` · `research search "…"` · `research task context T-…`.

## Goal

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

Git: `master` @ `5b411ffebc` (uncommitted changes)

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

- F-009 [supported, medium] SUPERSEDES F-005 (2nd independent review). Spectral mismatch ALWAYS biases a nominal-spectrum estimator, and on the real ray subset every condition is DETECTABLE with an appropriate test - no cell is genuinely 'unbiased'. Noise-free MLE on exact expected counts, 48 P2 rays, 1e6 photons, floor = rms(CRLB)/sqrt(20): dual-kVp iodine tilt +-10% gives rms/floor 0.82-0.92 BUT signed-mean z = +4.6/-5.2 (detectable); dual-kVp iodine energy-scale +-2% rms/floor 3.8-3.9, z = +-22; dual-kVp WATER tilt rms/floor 4.0-4.3, z = +-24 (a large bias the iodine-only criterion hid entirely); 4-bin photon counting on P2 is 21-38x floor. All fits converged. The pre-registered 3x-observed-rms criterion is DEFECTIVE: it compares a noise-inflated rms against an unsubtracted floor and so cannot distinguish 'no bias' from 'bias below the rms detection limit'. A signed-mean-over-rays test is the correct statistic. Supersedes the earlier 'under-powered, not unbiased' framing, which conceded too much to the defective criterion. — evidence: EXP-002, RUN-0002
- F-010 [supported, medium] SUPERSEDES F-007 (2nd independent review). Identifiability: STRUCTURAL part (rank <= min(n_meas,k), so 1 measurement cannot separate 2+ materials and dual-kVp cannot separate 3) is arithmetic, not evidence, and holds at any threshold. EMPIRICAL part, on the real ray subset at 1e6 photons: rank verdicts are robust for rank thresholds 1e-3 to 1e-10 (100% full rank) but NOT at 1e-2, where the full-rank fraction collapses to 0% for dual-kVp k=2 and 4-bin k=3 and 58% for 4-bin k=2, because smallest/largest singular-value ratios are only 0.0020-0.0123. (The superseded F-007 wrongly claimed robustness up to 1e-2, contradicting the ratios quoted in its own sentence.) Median condition numbers: dual-kVp k=2 195 (max 506), 4-bin k=2 99, 4-bin k=3 217. At EQUAL basis size k=2 and equal total unattenuated photons, 4-bin photon counting has median iodine CRLB 0.00151 vs dual-kVp 0.00376 g/cm^2 = 2.5x better by median (2.4x by rms aggregation; the ratio is ray-dependent). This compares an idealised energy-weighted 2-exposure design against an idealised 4-window 1-exposure design with NO detector response, charge sharing or pile-up, so it is an upper bound on the photon-counting advantage. — evidence: EXP-002, RUN-0002
- F-006 [supported, medium] SUPERSEDES F-002. Empirical Monte-Carlo std agrees with the Cramer-Rao bound to within 12.5% (median over rays; range 0.875-1.106) for the three identifiable cells at 1e6 photons, with a systematic pattern (P2 cells low 0.87-0.95, P3 k=3 cells high 1.05-1.11) rather than scatter about 1. The sampling spread of this check is ~3-4% (median of per-ray std estimates over 26-54 rays, 20 draws each), so the deviation is real, not check noise. This is a SELF-CONSISTENCY check of Jacobian + sampler + MLE under one shared model and table: a wrong attenuation table or spectrum would pass it equally. It is not validation of physical fidelity. — evidence: EXP-002, RUN-0002
- F-008 [supported, medium] AMENDS F-003 (scope). The dose-independent misspecification bias is an IODINE LINE-INTEGRAL statistic in g/cm2 (not g/cm3 density), measured on 27 rays of the 3-material phantom with a 2-material estimator. Both designs agree (bias ratio 1e6/1e5: M2 0.89, M4 1.02; 7.0x and 15.0x their noise floors). Dose-independence of misspecification bias is expected from theory, so this is a confirmation that the implementation behaves correctly, not a discovery. — evidence: EXP-002, RUN-0002
- F-011 [preliminary] AMENDS F-009 (two numerical/scope corrections from the 2nd reviewer's verdict check, verified by Opus). (1) The 4-bin photon-counting bias on P2 is 27-43x the floor over all 8 (condition x material) cells, NOT the 21-38x stated in F-009. (2) The dual-kVp iodine TILT case is MARGINAL and must not be quoted as strongly detectable: per-ray bias is only 0.66-0.75 of the floor and the signed-mean z of about 5 arises only by pooling 48 rays. (3) All z values are noise-free hypothetical power computed against a noise-only null, i.e. the bias an infinite-data experiment would see relative to a 20-realisation floor - they are NOT observed test statistics from the Monte-Carlo run. F-009's central claim (mismatch always biases; the rms criterion is defective; dual-kVp water tilt bias is large and was hidden by iodine-only scoring) is unaffected. — evidence: EXP-002, RUN-0002
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

- [awaiting review] F-011 AMENDS F-009 (two numerical/scope corrections from the 2nd reviewer's verdict check, verified by Opus). (1) The 4-bin photon-counting bias on P2 is 27-43x the floor over all 8 (condition x material) cells, NOT the 21-38x stated in F-009. (2) The dual-kVp iodine TILT case is MARGINAL and must not be quoted as strongly detectable: per-ray bias is only 0.66-0.75 of the floor and the signed-mean z of about 5 arises only by pooling 48 rays. (3) All z values are noise-free hypothetical power computed against a noise-only null, i.e. the bias an infinite-data experiment would see relative to a 20-realisation floor - they are NOT observed test statistics from the Monte-Carlo run. F-009's central claim (mismatch always biases; the rms criterion is defective; dual-kVp water tilt bias is large and was hidden by iodine-only scoring) is unaffected.: Agent finding awaiting an independent review
- [awaiting review] F-003 Fitting a 2-material basis to 3-material data produces bias that does NOT shrink with photon count (bias ratio 1e6/1e5 = 0.89 and 1.02 versus 0.32 expected for noise-driven error), at 7-15x the noise floor: basis misspecification is a systematic error that more dose cannot fix.: Agent finding awaiting an independent review

## Active decisions

- D-004 (2026-10-11) The correct test for a systematic bias measured over many rays is the SIGNED MEAN over rays compared to floor/sqrt(n_rays), not an rms compared to an unsubtracted noise floor. An rms statistic is inflated by noise and cannot distinguish 'no bias' from 'bias below the rms detection limit'; it declared a bias with z=5 to be 'at the noise floor'.
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
