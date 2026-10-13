# CTsynth — Scientist Guide

<!-- research:guide:begin — narrative maintained by the scientist-guide skill -->

Build the minimum reliable foundation for a long-term research platform that creates editable, physics-grounded virtual CT patient populations.

The questions currently driving the work:

- Q-001: P01: Can material maps be inferred from dual-energy/spectral CT measurements well enough to seed editable virtual patients?
- Q-002: P01-H1: Does a physics-consistency (differentiable polychromatic forward model) loss beat image-domain baselines when the test-time forward model is mismatched from the training one (no inverse crime)?
- Q-003: P01-H2: Does material-map error depend more on spectral separation (kVp pair / photon-counting bins) and noise than on the inference architecture?
- Q-004: P01-H3: Do models trained on procedural phantoms generalise to held-out material compositions (e.g., unseen iodine concentrations)?
- Q-005: P01-Q5: On the AAPM DL-Spectral CT benchmark, how do a constrained classical estimator and a small supervised U-Net compare for 3-material decomposition, and what makes the 3-from-2 problem well posed?
- Q-006: P02-Q6: Can a network recover physically meaningful material maps from spectral measurement consistency ALONE (no paired labels)? Prior analysis (F-014) says NO for the AAPM 2-measurement/3-material case: the null direction is invisible to the data, so a pure consistency loss is minimised by a one-parameter family containing the truth and infinitely many wrong maps. The scientific content is therefore WHICH added prior selects the truth, and whether it does so for physical reasons or by accident.
- Q-007: P04-Q7: How much do realistic photon-counting detector effects (energy response, charge sharing, pile-up, calibration) change material identifiability and the producer/inverse-model mismatch, relative to idealised energy bins?
- Q-008: P04-Q8: Do material fields learned from simulated pairs (XCIST now; DukeSim, XCAT later) transfer to real or clinical spectral CT and to heterogeneous virtual patients, and how would that be measured without clinical truth?

## Where we stand (consolidated 2026-10-12, checkpoint CP-008; all data synthetic or simulated unless stated)

**The core difficulty is arithmetic, not learning.** With two spectral measurements per ray and three materials the
problem has a one-dimensional null space (F-010 for the structural/empirical split; F-018: exact-measurement decoys,
verified deterministic runs RUN-0007). A decoy that is entirely nonnegative exists about 20 cm of path length from the
truth (19.73 cm, F-024), so nonnegativity does not bound the ambiguity. A low measurement-consistency loss is therefore never
evidence of material accuracy. The label-free network ladder (EXP-004/RUN-0008, F-022) agrees: about 92% of the
learned error is the identifiability gap, not undertraining, and an unpreregistered-weight soft sum-to-one penalty was a
badly designed test, not a result about priors (F-023, my own correction). Published "label-free" comparators are not
precedents: Hellwege has a square 2x2 problem and no ablations (F-020), Peng uses an unpaired learned reference
prior (F-021).

**Real data punished the clean story.** On the real AAPM kit images the image-domain linear effective-mu model is
wrong by 7.8% (50 kVp) and 5.9% (80 kVp); amplified by 343 this makes the exact-on-paper sum-to-one estimator worse
than predicting zero (RMSE 1.555 vs 0.447, F-025), and simplex "wins" by clipping, not by physics (F-030). The
unconstrained estimator was best on those 3 cases. The RMSE comparison used 3 kit sample cases, not the 1000-case set, which
we never downloaded (T-017). Several of these claims are still preliminary pending independent review (T-037).

**Infrastructure, honestly bounded.** Research OS 0.5.0's native HPC protocol replaced our own bundle/import code
(D-010). A CPU smoke job ran on Duke SLURM (RUN-0012, F-034); that says nothing about GPU or scale. The XCIST
environment on the cluster needed a patched sdist build step (EXP-011) and then reproduced the laptop reference cell to
float32-level agreement (RUN-0022, F-036).

**An independent simulator.** XCIST 1.6.9 is now our first independent forward-data producer (D-011). Truth is always
our input phantom; XCIST is a model too. Adapter conventions were checked against analytic references (tests, not
registered findings). On three known water/calcium/iodine disk phantoms at 80 and 120 kVp (dataset v0, RUN-0017) the
staged ladder (EXP-010, RUN-0023, F-037 - preliminary, needs independent review T-033) shows: with XCIST's own
ingredients our forward model matches to 0.26-0.47% (voxel chords; an adapter check, not independent physics); using
our own spectra raises it to 2.36-3.21% and is the largest term, but that figure is sensitive to how a 1-keV spectrum
is mapped onto a 4-6 keV quadrature, so read it as "a few percent", not a precise value. My first version of this
result (F-035) double-counted photons at 80 kVp; an independent reviewer caught it and F-037 replaces it. Attenuation
tables and detector response were small (about 0.1-0.2% and under 0.25%). This quantifies sensitivity to spectrum
specification; it is not an error in either spectrum, and it says nothing about scatter, photon counting or
reconstruction-domain effects, which we have not compared.

**What we do not know.** Whether any prior selects the truth for a physical reason rather than by accident
(E2/E4, EXP-014/EXP-016, registered, not run); whether supervised pretraining survives a change of acquisition (E3,
EXP-015); how photon-counting detector effects change any of this (Q-007; XCIST's PCCT model is unvalidated and unused);
whether anything learned on simulated pairs transfers to real clinical spectral CT or to heterogeneous XCAT virtual
patients (Q-008); DukeSim data and formats (nothing verified beyond the 2019 paper, T-025).

**Next.** Generate dataset v1 with declared phantom-level splits and a power analysis before training (T-032), then the
cheapest informative experiment: E4's deterministic decoy analysis on XCIST data (T-039), then E1 (T-038). Do not
interpret same-simulator fit as clinical validation.


<!-- research:guide:end -->
