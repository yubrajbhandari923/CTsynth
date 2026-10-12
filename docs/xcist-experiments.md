# Registered learning experiments on XCIST pairs (E1–E4) — design, NOT yet run

Registered in Research OS before any run (see experiment ids in `research experiment list`). Common rules
(CLAUDE.md pre-registration): score **every** material (water, calcium, iodine) with its own metric and detection
floor; floor from a power analysis *before* running (task: compute from dataset-v1 between-phantom variance); prefer
deterministic quantities; structural results (rank ≤ min(n_meas, k)) are never reported as evidence; one factor at a time.
Findings of F-014/F-018: 3 materials from 2 spectra is rank-deficient; a low consistency loss is not material accuracy.

## Data and splits (declared now; the file with ids+hash is committed before any training)
- Unit of splitting = **phantom** (a phantom and all its acquisitions share one split). Dataset v1 (to be generated from
  `projects/p04_xcist/generate.py`, parametrised random disk phantoms via `ctsynth.seeding.rng(seed, phantom_idx)`):
  train 24 / validation 6 / test 12 phantoms. Test touched once per experiment.
- Held-out acquisition cells (declared): E1/E3 test on (a) held-out phantoms at training kVp, (b) held-out **kVp pair**
  (e.g. train 80/120, test 100/140 — needs added spectra), (c) held-out dose (lowest mA), (d) corner = (b)+(c).
- Mismatch ladder rungs (see F-035 and the validation design): L1 noise seed only; L2 spectrum specification
  (our spectra_v1 vs XCIST); L3 + detector weights; L4 + 2×-sampled XCIST; **no tuning on a rung or any higher rung**.
- Normalisation statistics from train only. Always report zero-prediction and train-mean baselines.

## E1 supervised material estimation
Arms: classical constrained decomposition, per-pixel MLP, small U-Net (matched parameters where possible). Primary: per-material
RMSE (volume fraction) on held-out phantoms/cells + rung at which each learned method stops beating classical. Secondary:
boundary vs interior, composition violation (|Σf−1| inside body), forward consistency (self-consistency check only).
Gate: only claims about held-out phantoms/protocols actually tested; no clinical claim.

## E2 physics-consistent self-supervision
Differentiable CTsynth forward model fitted to XCIST *measurements*; truth used only for evaluation. Arms: physics-only
(negative control), +nonnegativity, +oracle occupancy (labelled upper bound), +estimated occupancy (no truth leakage),
+spatial prior (TV), +anatomical prior. Primary: null-direction vs orthogonal error component per arm; decoy test
(does output move toward truth or toward exact-measurement decoys?); 5 seeds. A prior is credited only if it helps on
held-out phantoms. Projection-domain early stopping allowed (label-free).

## E3 hybrid adaptation
Pretrain supervised on XCIST pairs (source rung L1); adapt with physics loss on unlabeled data at a changed acquisition
(held-out kVp / different spectrum specification / 2×-sampled). Arms: synthetic-only, target-only (N labels ∈ {0,10,50,200}
phantoms is the plan for later; first run N=0), pretrain+adapt, classical. Primary: per-material RMSE on held-out phantoms,
and drift along the null direction before/after adaptation (does the prior survive?).

## E4 identifiability and priors
Deterministic first (no training): construct exact-measurement decoys on XCIST data and measure their
distance from truth per material; then compare physics-only / nonnegativity / oracle occupancy / estimated occupancy /
justified spatial prior on the same data. Do not interpret low reconstruction error as material accuracy.

## Not covered / future
DukeSim paired import (docs/dukesim-request.md, T-025), PCCT detector effects (T-024), real AAPM (T-017/T-022),
clinical spectral CT adaptation, XCAT heterogeneity (T-027), 3D virtual populations (T-026), scatter.
