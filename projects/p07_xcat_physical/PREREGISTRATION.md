# Pre-registration: physical-phantom inversion of simulated COPD CT (EXP-040, PLAN-009)

Written 2026-10-29 BEFORE any network was trained and before any test-patient baseline or network score was computed. Numeric detection floors are appended in section 7 after the baselines have been run on the VALIDATION patients only, and before training starts. Changing sections 1-7 after training starts voids this registration; deviations are reported as deviations.

## 1. Question and scope
Can a 3D U-Net recover XCAT-derived physical composition concentrations from simulated monoenergetic COPD CT of unseen patients, does the predicted phantom reproduce the input CT through our LEAP simulator, and does it transfer to a photon energy the network never saw? Scope: simulated CT (borrowed Duke1 helical geometry, monoenergetic, no scatter/electronic noise) of the same 67 XCAT COPD phantoms that supply the targets; frozen split `configs/split_v1_frozen.json` (47/10/10, sha256 3417bfcb...; 22728 and 21535 in train); 0.8 mm grid. Not real data, not DukeSim. Follows EXP-039; does not reinterpret F-049.

## 2. Target definition (validated before training; `reports/material_audit.md`)
- Organ IDs are not materials. All 67 patient tables give the SAME elemental composition for each organ ID (checked); only densities differ per patient (smeared, about +-3 %, emphysema 0.02-0.08 g/cm3). The 53 organ IDs / 9 `mc_id`s reduce to **8 distinct normalised compositions** (sternum and rib/scapula/sternum-trabecular bone are identical). Channels: 0 air, 1 body, 2 flat bone (first xlsx name `sternum`), 3 muscle, 4 lung tissue, 5 spine bone, 6 skin, 7 blood (`aorta`).
- Target `c_k(r)` = mass concentration (g/cm3) of composition k at voxel r = box average over the voxel of (native one-hot channel x patient density). Then `mu(r,E) = sum_k c_k kappa_k(E)` (kappa from `data/tables/xcat_element_attenuation_v1.json`, xraylib total incl. coherent) reproduces the simulator's box-averaged mu EXACTLY (linear). Total density = sum_k c_k. Partial volumes are occupancy-weighted mixtures, never relabelled. Stored as uint16 with step 1/8192 g/cm3 (max error 0.25 HU in mu, mean 0.02 HU).
- This 8-channel basis is lossless with respect to the simulator. It is NOT identifiable from CT: over 30-140 keV the 8 attenuation shapes span about 2 dimensions (singular values 2.79, 0.47, then <6e-4), and at {70, 80} keV the second singular value is 0.049 (relative 1.7 %). Air, body, muscle, lung tissue, skin and blood are within 0.02-4.5 deg of each other in attenuation shape (0.3 deg or less except `body` 3.4-4.5 deg); they differ mainly in density. Consequently a network that separates them is using anatomical prior knowledge, not physics. No smaller basis is imposed on the network; the audit reports what is and is not separable.
- Validation of the targets (all 67 patients, `conc_v1.json`): closure to the simulator mu at 70 and 80 keV (relative < 1e-6), density closure, alignment against the cached clean CT by integer voxel shifts (minimum at 0 on all axes), agreement of the dominant channel with the organ-group label (about 99 %, remaining = partial-volume voxels).

## 3. Data and arms (identical budget to EXP-039)
Cached CT from EXP-039 (`cache_v1`, 291 volumes) is reused unchanged. Arms: **A** clean only; **B** noisy only (n0_1e5); **C** varied (clean, n0_1e5, var1); **D** = C + FiLM conditioning on the 7-number acquisition vector. 2 seeds each. 40 000 steps, batch 4 patches of 96^3, AdamW lr 2e-3 cosine, bf16, class-balanced foreground-centred patch sampling (70 %), UNet3D base 16, depth 3. **No hyper-parameter search.** Held-out acquisition settings (80 keV, pitch 1.4) are never trained on. Checkpoint selection: lowest validation sum_k MAE_k (g/cm3) on the 10 validation patients, each arm's own training variants. Test patients are scored once.

## 4. Loss (fixed a priori, nothing tuned)
`L = mean_k Huber_0.02(c_hat_k - c_k)/0.02 + mean|sum_k(c_hat_k - c_k)| + mean|mu70(c_hat) - mu70(c)|/mu_water(70)`, output through softplus (c_hat >= 0). Dice is not optimised. The 80 keV attenuation is NOT in the loss, so the 70 -> 80 keV test is not tuned for.

## 5. Baselines (use only the 70 keV image + stated assumptions)
**P0** one reference composition (muscle) with continuous density mu70/kappa70: reproduces mu70 exactly by construction (STRUCTURAL, not evidence) and cannot predict any energy dependence other than that of muscle. **P1** two compositions (flat bone vs muscle) split by an HU threshold fitted on TRAIN patients only (minimising body mu80 MAE). **O1** ORACLE true dominant composition per voxel with density from mu70: an upper-bound diagnostic that uses ground truth, not an inference baseline; at pure-material voxels it is exact by arithmetic.

## 6. Outcomes (every output scored; no single favoured material)
Per test patient and variant: per-channel concentration MAE (g/cm3) over the valid region and over each channel's support (c > 0.05 g/cm3), total-density MAE (valid region and body), mu70 and mu80 error from the PREDICTED concentrations in HU units (valid region and body). Closed loop (LEAP, same geometry and FBP): predicted vs true phantom CT at 70 keV (reproduction) and at 80 keV (cross-energy; network input is the 70 keV image only), CT HU MAE/RMSE/bias/SSIM, projection-domain line-integral residual; deterministic (noise-free) results are primary; coupled-noise and independent-noise-floor results are reported separately and never mixed. Also reported: the "70 keV image reused as the 80 keV image" trivial reference.
**Primary quantities** (patient-level paired bootstrap, 4000 draws, over the 10 test patients, arms averaged over 2 seeds): body mu80 MAE (HU) and body density MAE (g/cm3) for C - P1 and C - A on `clean` and on the pooled in-distribution panel (clean, n0_1e6, n0_3e5, n0_1e5, n0_3e4, lowpass_w2, lowpass_w4, dec2, e60, var1); secondary: B - A, C - B, D - C, held-out e80 and pitch1p4 inputs, per-channel results. Differences are interpreted only if the paired CI excludes 0 AND |difference| exceeds both the section-7 floor and the seed range of the arms compared; "not detected" does not mean "no effect". No condition is required to be significant (not a multiple-condition conjunction).

## 7. Detection floors (frozen after the VALIDATION baseline run, before training)
Frozen from baselines on the 10 VALIDATION patients (clean CT); P1 threshold 125 HU fitted on 47 train patients.

| quantity | P0 mean (SD) | P1 mean (SD) | O1 mean (SD) | floor = 2.8*SD(P1)/sqrt(n_test) |
|---|---|---|---|---|
| body density MAE (g/cm3) | 0.0334 (0.00326) | 0.02268 (0.00101) | 0.01605 (0.00129) | 0.000892 |
| body mu70 MAE (HU) | 15.29 (1.23) | 15.29 (1.23) | 15.29 (1.23) | 1.09 |
| body mu80 MAE (HU) | 20.64 (1.56) | 17.59 (0.978) | 15.5 (1.24) | 0.866 |

Interpretation rule: a network-vs-baseline or arm-vs-arm difference smaller than its floor, or smaller than the seed range of the arm means, is reported as 'not detected', not as 'no effect'.

## 8. Not claimed
Segmentation-free recovery of real anatomy; anything about real scanner data; uniqueness of the recovered composition (non-identifiable, section 2); ground-truth fidelity beyond the simulator (the targets come from the same phantoms that generated the CT, so the CT-to-target map has no label noise); seed-level variance beyond 2 seeds; the generalisation of a 10-patient test set to a population.
