# Pre-registration draft: inverse U-Net arms A-D (COPD XCAT, simulated CT)

Status: **not run**. It needs the 65 missing phantoms (T-084). Written before any held-out result exists; to be registered as a Research OS experiment when the data are staged.

**Question.** Does training on controlled acquisition/noise/kernel variation (arm C) or additionally conditioning on the acquisition metadata (arm D) improve organ-group segmentation of simulated COPD CT relative to fixed-acquisition training (arms A, B), on held-out patients, at in-distribution and held-out acquisition settings?

**Fixed design.** Split: `configs/split_seed0.json` (47/10/10 patients, one seed, identical for all arms, patient level). Grid 0.8 mm, 70 keV baseline, borrowed Duke1 helical geometry. Targets: 15 organ-group labels (air + 14), cross-entropy + soft Dice. Network: 3D U-Net, base 16, depth 3 (1.4 M parameters), patches 96^3, batch 4, bf16, AdamW, cosine schedule, same step budget and seed for every arm. Training variants: A `clean`; B `n0_1e5`; C and D `clean, n0_1e6, n0_3e5, n0_1e5, n0_3e4, lowpass_w2, lowpass_w4, apod_0p75, e60, dec2`. **Held out from every training set:** `e80` (energy) and `pitch1p4` (pitch). Model selection: validation patients only (in-distribution variants), best macro foreground Dice. The test patients are evaluated once at the end.

**Primary outcome.** Macro Dice over the 14 foreground classes, pooled voxels per variant, bootstrap CI over test patients (10 patients: only large differences are detectable; stated before running: with 10 test patients a paired difference below about 0.02 macro Dice should be treated as undetected, not as no effect).

**Comparisons (each reported with the per-class table and the confusion matrix).** (1) A vs B on `clean` and on `n0_1e5` (does noise-aware training help under noise); (2) B vs C across all in-distribution variants; (3) C vs D (value of conditioning beyond augmentation); (4) all arms on the held-out `e80` and `pitch1p4` (acquisition generalisation, distinct from patient generalisation); (5) the intensity-only baseline (per-class Gaussian on HU fitted on the training clean CTs) as a floor.

**Not claimed.** Segmentation accuracy of simulated CT says nothing about recovery of physical material composition or about real scanner data; the organ-group labels follow the xlsx names only; orientation letters stay provisional; noise levels are photon-count scales, not calibrated dose.
