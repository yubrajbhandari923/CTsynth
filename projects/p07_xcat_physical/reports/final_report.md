# Physical-phantom inversion of simulated COPD CT with a 3D U-Net (EXP-040)

Simulated data only (XCAT COPD phantoms through our LEAP helical simulator, borrowed Duke1 geometry, monoenergetic, no scatter). 10 held-out test patients scored once, 2 seeds per arm. Preliminary until an independent review (CLAUDE.md promotion gate). Pre-registration with frozen floors: `PREREGISTRATION.md`.

## What was built
- **Target** (not the 15 semantic classes): per-voxel mass concentration c_k (g/cm³) of the 8 distinct elemental compositions of the XCAT tables. μ(E) = Σ c_k (μ/ρ)_k(E) reproduces the simulator's μ at 70 and 80 keV to ≤5.7e-7 relative on all 67 patients (`material_audit.md`).
- **Network** `PhysUNet` (UNet3D base 16, depth 3, softplus head), 70 keV CT in, 8 concentration maps out. Loss: Huber on concentrations + total density + μ70 consistency; no Dice; 80 keV not in the loss.
- **Arms** (same budget as EXP-039, 40 000 steps, batch 4×96³): A clean only; B n0=1e5 noisy only; C clean + n0_1e5 + var1; D = C + FiLM on the acquisition vector. Seeds 0, 1.
- **Baselines** (70 keV image only): P0 one composition (muscle); P1 bone/soft split at 125 HU (fitted on train); O1 oracle true composition identity per voxel (diagnostic, uses ground truth).
- **Closed loop**: predicted concentrations → μ(70) and μ(80) → LEAP forward projection + helical FBP, compared with the same pipeline on the true phantom.

## Results (10 test patients, body voxels; mean over patients; [seed range])
Errors are against the TRUE phantom's quantities: density in g/cm³, μ in HU-equivalent.

Pooled in-distribution panel (clean, n0 1e6/3e5/1e5/3e4, lowpass w2/w4, dec2, e60, var1), pre-registered primary:

| model | body density MAE | body μ80 MAE (HU) |
|---|---|---|
| P0 | 0.0499 | 37.4 |
| P1 | 0.0373 | 32.5 |
| O1 (oracle) | 0.0333 | 32.9 |
| A | 0.0235 [0.0233, 0.0236] | 23.5 [23.4, 23.6] |
| B | 0.0136 [0.0130, 0.0142] | 13.7 [13.1, 14.2] |
| C | 0.0132 [0.0131, 0.0134] | 13.2 [13.0, 13.3] |
| D | 0.0135 [0.0129, 0.0140] | 13.2 [13.0, 13.3] |

Paired patient-bootstrap differences (4000 draws; patient sampling only, not training randomness), pooled panel, μ80 HU / density g/cm³:
- C − P1: −19.3 HU [−21.9, −17.0], −0.0240 [−0.0268, −0.0216]; 10/10 patients better. Far above the frozen floors (0.866 HU, 0.00089).
- C − A: −10.3 HU [−12.3, −8.5], −0.0102; 10/10 better. A was trained on clean CT only and degrades on noisy inputs (clean test input: A is the best arm, 8.3 HU vs C 10.2 HU, density 0.0085 vs 0.0103; C − A = +1.9 HU on clean, 10/10 patients worse).
- C − B: −0.50 HU [−1.06, +0.12], 8/10 better. **Not detected**: CI includes 0 and |diff| < the 0.866 HU floor.
- D − C: −0.02 HU [−0.30, +0.30]. **Not detected.**
- C − O1: −19.8 HU [−22.7, −17.1]. The network beats the oracle-composition baseline because the baselines inherit the CT-vs-phantom reconstruction discrepancy; see below.

Held-out acquisitions (never trained on; network input is the 70 keV-trained pipeline):
- e80 (input is an 80 keV CT), body μ80 MAE: P1 27.9, A 21.1, B 14.1, C 13.6, D 15.0 HU. C − P1 = −14.2 HU [−16.2, −12.3].
- pitch1p4: P1 30.0, A 22.5, B 11.0, C 12.4 HU. **D fails: 142 HU (density 0.143 g/cm³)**, D − C = +129.9 HU [+126.9, +133.0], 0/10 better. The FiLM conditioning vector for pitch 1.4 is outside the training range, and arm D extrapolates badly. This is a clear negative result for conditioned inversion on unseen acquisition settings. Arm D also did worse than C on e80 (+1.4 HU).
- Noise: for n0_3e4, μ80 MAE P1 57.1, A 47.7, B 15.4, C 17.2, D 17.3 HU. The noise-trained arms denoise strongly; the noisy-only arm B is slightly better than C on its own noise level (as expected) and C is better on var1 and e60 (13.8 vs 16.5 and 12.9 vs 18.4 HU).

Per-channel concentration MAE over each channel's support, clean input (g/cm³): body (adipose-like) C 0.044 (P1 0.850), muscle C 0.034 (P1 0.058), lung C 0.020 (P1 0.258), flat bone C 0.071 (P1 0.078), spine cortical C 0.102 (P1 0.943). **Skin (0.657) and blood/aorta (0.799) are not recovered by any arm** (error ≈ the channels' own mean concentration, i.e. predicted ≈ 0): their attenuation shape is within ≈1° of muscle (`material_audit.md`), so the CT carries no way to separate them; they are absorbed by neighbouring channels. Air is below the 0.05 g/cm³ support threshold and is not scored per-channel.

## Closed loop (LEAP resimulation, 10 test patients, mean over patients, body voxels, MAE in HU)
Truth pipeline check: truth70 vs the cached clean CT: 0.066 HU, SSIM 0.999999 (the resimulation reproduces the cache).
- Reproduction at 70 keV (deterministic, predicted phantom → CT vs true-phantom CT; mean of 2 seeds, body MAE HU): A 6.3, B 7.2, C 7.7, D 7.5; SSIM ≥ 0.994; projection line-integral RMSE ≈0.6–0.75 % of the truth RMS (3.03).
- Cross-energy 80 keV (network saw only 70 keV; 2-seed mean, body MAE HU): A 6.2, B 7.1, C 7.5, D 7.2. Reference "reuse the 70 keV image as the 80 keV image": 9.1 HU (bias +3.4 HU). The predicted phantom beats that trivial reference by 3.0 (A), 2.0 (B), 1.6 (C), 1.9 (D) HU; 80 keV error ≈ 70 keV error for every arm, i.e. no detectable degradation from the energy change. The margin over the trivial reference is small in absolute terms because 70 → 80 keV changes soft-tissue HU little; the transfer signal is carried mainly by bone. Not tested: energies further from 70 keV.
- Noise-aware reading: coupled noise (same RNG on predicted and true projections, n0 = 1e5), 2-seed mean: A 8.2, B 9.1, C 9.6, D 9.4 HU. Independent noise floor (two realisations of the TRUE phantom at n0 = 1e5): 44.2 HU body MAE. The resimulation differences are far below the independent noise floor; the coupled-noise number is not a noise comparison between independent scans and is reported only as such.
- Per-run values: `test/resim_summary.json`.

## Invalidated assumptions and deviations (documented, not hidden)
1. **"μ70 error of any voxelwise method has a floor of ≈15.3 HU" (PREREGISTRATION §7 note) was wrong for the network.** P0/P1/O1 inherit the 15.3 HU CT-vs-phantom discrepancy because they read μ70 from the CT. The network does not: it outputs a phantom, and its error against the phantom is 6–8 HU on clean input, i.e. it is closer to the phantom than the CT is. Interpretation (not tested here): the CT-vs-phantom discrepancy is a systematic function of anatomy and reconstruction blur that the network can learn. The floors (§7) were frozen from the baselines and remain valid for the comparisons made; this note only corrects the interpretation of the baseline μ70 value.
2. Resimulation had CUDA out-of-memory failures on the 24 GB GPUs for large patients. Fixes (truth/cached CTs on CPU, free concentration volumes before projection, abort on silent FBP failure) are committed. **Patient 11989 first produced a silent invalid FBP (860 HU error in all 8 arms; LEAP printed a non-fatal cudaMalloc error); the result was discarded (kept as `resim_p11989.FAILED_fbp_oom.json.bak`) and the patient rerun alone.** The other nine patients ran before the abort-guard existed; their values (5.4–6.7 HU for arm A) are consistent, so I treat them as valid, but I did not rerun them with the guard.
3. The long single-patient overfit check (Stage A) was stopped at step 1600 of 4000 (loss 0.078, still falling), so "overfit to near zero" was not demonstrated. Gradient finiteness, checkpoint round trip, resume and export round trip passed.
4. Cross-energy transfer was tested only for 70 → 80 keV.
5. Seed ranges use 2 seeds. B and D have seed-to-seed variation in validation selection score (e.g. B: 0.067 vs 0.096), so arm differences smaller than the seed range are reported as not detected.

## Limitations
- Simulated data; borrowed Duke1 geometry; 10 test patients drawn from 67 phantoms of one phantom family; native voxel 0.25 mm assumed; orientation provisional (`axcodes` convention, check in Slicer).
- The 8 compositions are not identifiable from CT intensities (two dominant singular values); the network separates air/lung/body/muscle/bone by anatomical context, and cannot separate skin or blood from soft tissue (measured above).
- Targets come from the same phantoms that generated the CT, so there is no label noise; this overstates what real data would permit.
- No independent review yet (F-049-style); finding stays preliminary.
- `origin/feature/xcat-simulator-clean` still contains six mistaken EXP-036 commits; do not use it.

## Files
Aggregates: `reports/test/{summary.csv, paired_differences.csv, resim_summary.json, baselines_test.json}`. Cluster outputs and Slicer files: see the handoff message.
