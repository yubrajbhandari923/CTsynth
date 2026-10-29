# Noise, kernel and acquisition variation study (copd_22728, 0.8 mm, 70 keV baseline)

22 variants simulated from the SAME phantom (job 203527, node004), metrics against the ideal noise-free box-averaged image at the same energy. Figures: `variation/variants_same_slice_lung_window.png`, `variation/variants_same_slice_soft_tissue_window_zoom.png`, `variation/variation_summary.png`; full table `variation/variation_metrics.csv` (also per-organ-group MAE). copd_22728 is a TEST patient in the split; it was used only descriptively here.

**Definitions.** N0 = incident photons per ray per view (Poisson in the count domain, zero counts clamped to 1, no electronic noise): a photon-count scale, NOT a calibrated mAs, and not dose-normalised (halving the views or raising the pitch lowers the total photon count). Body MAE/RMSE: HU error vs ideal inside the eroded body, valid FOV and helical z range (unsmoothed). Noise SD and texture: (variant - same-kernel clean reconstruction) inside eroded muscle (4.3 M voxels). Edge: 10-90% width of the angle-averaged edge-spread function at the body/air boundary (ideal image: 0.89 mm).

| variant | what changes | body MAE [HU] | muscle noise SD [HU] | noise ACF lag-1 x / y / z | edge [mm] | FBP [s] |
|---|---|---|---|---|---|---|
| clean | nothing (noise-free) | 12.6 | - | - | 1.27 | 4.7 |
| n0_1e6 / 3e5 / 1e5 / 3e4 | photon statistics only | 14.5 / 17.3 / 22.9 / 35.3 | 9.3 / 16.9 / 29.3 / 53.8 | -0.14 / +0.12 / +0.16 | 1.26-1.27 | 4.8-5.0 |
| lowpass_w2, lowpass_w4 (N0 3e5) | LEAP `set_FBPlowpass` window | 17.6 / 22.6 | 8.2 / 3.7 | +0.31 / +0.51 / +0.16 ; +0.68 / +0.79 / +0.15 | 1.61 / 2.39 | 4.8 |
| apod_0p75, apod_1p5 (N0 3e5) | Gaussian apodisation of projections along channels (own implementation) | 17.7 / 22.6 | 8.3 / 3.7 | +0.27 / +0.47 / +0.16 ; +0.67 / +0.78 / +0.15 | 1.63 / 2.39 | 4.8-4.9 |
| e60 / e80 (N0 3e5) | monoenergetic 60 / 80 keV (new projections) | 18.7 / 16.6 | 19.0 / 15.7 | -0.15 / +0.14 / +0.16 | 1.25 / 1.27 | 4.9-5.1 |
| pitch1p4 (N0 3e5) | table feed 1.4x (8000 views) | 18.9 | 20.1 | -0.15 / +0.12 / +0.16 | 1.26 | 3.6 |
| dec2 (N0 3e5) | every 2nd view (500 / rotation) | 21.9 | 24.3 | -0.15 / +0.12 / +0.16 | 1.28 | 2.6 |

**Findings (this one patient, this one slice-set; measured).**
1. Noise follows N0 as expected: SD scales as 1/sqrt(N0) (9.3 to 53.8 HU over a 33x range of N0). The unfiltered helical FBP noise is slightly anti-correlated along x (-0.14) and y-correlated (+0.12), and positively correlated along z (+0.16, helical interpolation); it does not change edge width.
2. `set_rampFilter` has no effect on the helical FBP (output bit-identical to the default, source: it acts only on the axial ramp path). `set_FBPlowpass` does: W=2 gives 2.0x lower noise and a 1.6 mm edge, W=4 gives 4.5x lower noise and a 2.4 mm edge. My projection-domain apodisation with sigma 1.5 pixels reproduces W=4 (3.7 HU, 2.39 mm), so it is a genuine smoother-kernel family, not post-reconstruction smoothing; the LEAP window is the one used in the dataset.
3. A smoother kernel raises the body MAE against the (sharp) ideal image because it blurs edges although it lowers noise: 22.6 HU at W=4 vs 17.3 HU at the unfiltered N0=3e5 (the MAE metric rewards neither noise nor sharpness alone).
4. Fewer views (dec2) and higher pitch increase noise by about sqrt(1000/500) and sqrt(1.4) respectively because N0 is per view (24.3 and 20.1 HU vs 16.9 HU); dec2 also adds streaks visible at the lung and rib boundaries.
5. Energy changes contrast, not only noise: bone is brighter at 60 keV (HU scale relative to water at that energy) and noise at fixed N0 is higher at 60 keV (19.0 HU) than at 80 keV (15.7 HU) because the body attenuates more.
6. Reconstruction cost: FBP 4.7-5.1 s at 11000 views, 3.6 s at 8000, 2.6 s at 5500 (about 0.43 ms/view); peak GPU 7-12 GB.
7. The clean reconstruction's body MAE of 12.6 HU is mostly edge and partial-volume mismatch against the box-averaged ideal image (consistent with the pipeline validation, T-077); per-organ-group MAE for every variant is in `variation/variation_metrics.csv` (columns `mae_<group>`), not interpreted here.
