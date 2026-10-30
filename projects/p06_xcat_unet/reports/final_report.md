# Inverse U-Net on simulated COPD CT: final report (EXP-039, PLAN-008)

Scope of every number below: **simulated** CT (our own LEAP/Poisson simulator, borrowed Duke1 helical geometry, monoenergetic, no scatter or electronic noise) of 67 XCAT COPD phantoms; targets are the 15 organ-group labels derived from the phantoms; **10 unseen test patients**, 2 seeds per arm; nothing here is about real scanner data or about recovery of physical material composition. Pre-registration: `EXPERIMENT_PLAN.md` (frozen before training; split `configs/split_v1_frozen.json`, sha256 3417bfcb...).

## 1. The answers

**How accurately does the U-Net recover XCAT labels on unseen patients?** On clean CT, macro Dice over the 14 foreground classes is **0.911-0.917** for every arm (95% bootstrap CI over the 10 test patients about +-0.003), against **0.27** for an intensity-only (HU) classifier. Twelve of 14 classes have Dice >= 0.89 (body, sternum, muscle, scapulas, spine, lung parenchyma 0.99; ribs 0.98; emphysema 0.985; aorta 0.96; skin 0.94; vessel 0.94; esophagus 0.90). Two classes are hard in every arm: `fissure` (0.75; thin, 0.022% of voxels) and `lung` (0.37-0.40, see section 4: it is intrinsically ambiguous, not a model failure).

**How does performance change with noise and acquisition?** Strongly, and it depends on what the arm was trained on (macro Dice, mean over 2 seeds; full table with CIs and seed ranges in `final/macro_dice_by_arm_variant.csv`):

| test condition | A clean-only | B noisy-only | C varied | D varied + conditioned | intensity baseline |
|---|---|---|---|---|---|
| clean | **0.917** | 0.911 | 0.911 | 0.911 | 0.271 |
| noise N0=1e6 / 3e5 | 0.900 / 0.829 | 0.912 / 0.912 | 0.910 / 0.909 | 0.910 / 0.908 | 0.268 / 0.259 |
| noise N0=1e5 (about 29 HU) | 0.691 | **0.908** | 0.905 | 0.904 | 0.241 |
| noise N0=3e4 (about 54 HU) | 0.496 | 0.865 | **0.886** | 0.884 | 0.203 |
| LEAP low-pass W=2 / W=4 (smoother kernel) | 0.865 / 0.670 | 0.896 / 0.716 | 0.907 / 0.892 | 0.906 / 0.893 | 0.273 / 0.273 |
| every 2nd view (dec2) / 60 keV | 0.737 / 0.793 | 0.909 / 0.907 | 0.906 / 0.908 | 0.905 / 0.908 | 0.245 / 0.218 |
| per-patient varied draw (var1) | 0.732 | 0.859 | **0.903** | 0.901 | 0.237 |
| **held out: 80 keV** | 0.736 | 0.800 | 0.840 | **0.848** | 0.219 |
| **held out: pitch 1.4** | 0.802 | 0.911 | 0.908 | 0.907 | 0.256 |

- A model trained only on clean CT is fragile: it loses 0.23 Dice at N0=1e5 and 0.42 at N0=3e4. Training on noise (B) removes that at the noise level it saw (+0.217 [0.156, 0.276] over A at N0=1e5, all 10 test patients better) and costs 0.006 on clean CT.
- Training on varied settings (C) beats noisy-only training (B) on every setting B did not see: smoother kernel W=4 +0.177 [0.170, 0.184], var1 +0.044, 80 keV +0.040 [0.032, 0.047], N0=3e4 +0.021; it loses at most 0.004 where B was trained (N0=1e5 -0.004, pitch 1.4 -0.003; clean no difference). Averaged over the in-distribution panel C-B = +0.024 [0.020, 0.030]. (Part of this gain is by design: B never saw kernels, 60 keV or decimation.)
- **Held-out acquisition settings.** Pitch 1.4 is not a hard shift for the noise-trained models (0.91): its main effect in this simulator is more noise (N0 is per view), which B, C and D have seen. **80 keV is a genuine contrast shift** and degrades every arm (A -0.18, B -0.11, C -0.07, D -0.06 relative to clean CT). The damage is in the bone classes: ribs 0.66 (B) vs 0.87 (C, D); scapulas 0.56 vs 0.82 / 0.87; spine 0.68 vs 0.77 / 0.80; sternum about 0.55-0.59 for all arms (nobody learns it at 80 keV). Training on two energies (60 and 70 keV, in C and D) helps most.

**Does acquisition conditioning help?** **No detectable benefit beyond varied training.** D - C is -0.0007 [-0.0015, 0.0000] in distribution (no difference) and +0.0037 [0.0022, 0.0055] on the two held-out settings, driven by 80 keV (+0.0084 [0.0051, 0.0121], 10/10 patients). That is below the pre-registered detection floor (0.02-0.03) and inside the seed spread at 80 keV (C 0.829-0.850, D 0.842-0.854), so by the pre-registered rule it is **not interpreted** as an effect. The bootstrap interval reflects patient sampling only, not training randomness (2 seeds). Conditioning on a 7-number acquisition vector gives the network nothing it cannot already read from the noise and kernel in the image.

## 2. What the comparison can and cannot support

- 10 test patients and 2 seeds: differences below about 0.02 are not resolvable; seed ranges are given next to every mean (arm A under noise varies by up to 0.09 between seeds and arm B at 80 keV by 0.043; the other arm/variant pairs mostly by under 0.02).
- The test set was evaluated **twice**: my first evaluation script did not save per-patient confusion matrices, so I fixed the script and re-ran the identical evaluation (same 8 frozen checkpoints, data and stride). All 9 pooled results reproduced exactly (log diff identical). No choice was made between the two runs. No hyper-parameter was tuned on any test data; the only selection was best-validation checkpoint (validation patients, each arm's own training variants).
- The development patients copd_22728 and copd_21535 were forced into train, so no patient used to build the simulator is in the test set.
- Labels come from the phantoms that generated the CT: this measures recoverability of the simulator's labels from its images, which can overstate real-world performance (no segmentation noise, no scanner differences).
- Arms differ in the number of training volumes (A, B: 47; C, D: 141) but not in optimiser steps (40 000 steps x 4 patches of 96^3, identical for all), so compute per arm is equal.

## 3. Compute and storage (measured)

| stage | wall | GPU-h | notes |
|---|---|---|---|
| transfer of 65 phantoms (181.27 GB), CPU job | 29.6 min | 0 | 102 MB/s; verified: all 260 files same size, 65/65 phantoms same first/last 256 MiB, 67/67 size = product of dims |
| simulation of 291 volumes, 8 GPUs, 0 errors | 13.2 min | 1.76 | 54 GB cache (int16 `.npy`), persisted per patient; estimate was 8.5 min/worker, cold NFS reads of the phantoms made it 13.2 |
| training, 8 runs x 40 000 steps | 4.76 h | 38.1 | 0.41 s/step on one A5000 each, 272-282 min per run |
| test evaluation (x2, see above) | 18.7 min each | 2.5 each | 12 variants x 10 patients per model, stride 64 |
| **all jobs of this task** | about 9 h elapsed of 48 h | **45.7** | 447 CPU-core-hours; one node (node004) at all times |

Storage: new durable data about 245 GB on `/home` (172 GiB phantoms, 54 GB cache, checkpoints < 1 GB, 1.7 GB NIfTI exports); node-local NVMe 54 GB working copy (staged in 219 s). Memory: simulator peak 12.6 GB GPU / 3.9 GB host RSS per process; trainer 4.5 GB GPU per run; no out-of-memory event in any job. Conclusion for repeat experiments: simulation is cheap (about 1.8 GPU-h for 291 volumes); training dominates, and caching removes LEAP from the trainer entirely.

## 4. Class difficulty

`lung` (organ IDs 6, 106-110; 0.033% of voxels) has Dice 0.37-0.40 in all arms: 50% of its voxels are called `lung_parenchyma`, 17.6% `body` (arm C seed 0, clean CT). Its ideal attenuation (density 0.29-0.30 g/cm3, about -710 HU at 70 keV) is the same material as several `lung_parenchyma` density bins (IDs 29-30, about -730 and -670 HU from their densities): the label is distinguishable only by anatomical position, not by CT intensity. This is the label-semantics caveat the task asked about: a fine-grained material-label experiment on these groups would be ill-posed. `fissure` (density 1.035, +28 HU, thin) and `esophagus` (0.90, next to muscle) are limited by partial volume and contrast. Per-class tables for clean, noisy and 80 keV: `final/per_class_dice.csv`; heat maps `final/per_class_dice_armC.png`, `final/per_class_dice_armD.png`.

## 5. Visualisation (3D Slicer)

Representative **held-out** test cases (patients 11989 and 13799, never in training), 4 conditions each (clean, N0=1e5, held-out 80 keV, held-out pitch 1.4): CT, ground truth and the predictions of arms A-D (seed 0), 1.7 GB, on DRCC in `/home/yb107/datasets/truect_pilot/outputs/xcat_unet/slicer_export/` (`p<patient>_<variant>_{ct,truth,pred_<arm>_s0}.nii.gz`, `class_names.json`, `class_colors.ctbl`). Overview figure: `final/test_patient_11989_overlays.png`. All volumes share one grid and affine (array axes S, L, P: **orientation letters are PROVISIONAL**). In Slicer: Add Data for the CT; Add Data for `*_truth.nii.gz` and `*_pred_*.nii.gz` with **LabelMap** ticked; load `class_colors.ctbl` as a Color table and assign it to each label map (Volumes module, Display); then switch label opacity (0.4) between truth and the four predictions. Lung window W 1400, L -300.

## 6. Reproduce

Code: `src/ctsynth/{xcat,seg}`, scripts in `projects/p06_xcat_unet/scripts`, all jobs submitted through `sub.sh` (node004, one-node cap). Order: `make_split.py` -> `make_dataset_plan.py` -> `dataset_job.sh` -> `validate_cache_job.sh` -> `train_job.sh 40000` -> `eval_job.sh` (calls `evaluate_unet.py`, `analyze_arms.py`) -> `export_job.sh`. Seeds: split 0, simulation 0 (stream per patient and variant), training 0 and 1.

## 7. Limitations and open items

Simulated data only; geometry borrowed from Duke1 (not the phantoms' original acquisition); N0 is a photon-count scale per ray per view, not dose; native voxel size 0.25 mm assumed for COPD; orientation letters provisional; held-out acquisition shifts limited to two settings (80 keV, pitch 1.4); 2 seeds; no study yet of more variants per patient (the cost model says 5 variants would cost about 1.4x the generation time and 1.7x the storage, but whether it helps was not measured; the C-vs-B gain suggests breadth of settings matters more than count).
