# AAPM DL-spectral CT challenge: starting-kit schema audit (2026-10-11)
Source: Zenodo 14262737 API (https://zenodo.org/api/records/14262737) + starting_kit.tgz only. No .npy.gz downloaded.
Kit extracted (never executed, never imported) at scratchpad/aapm_kit/starting_kit/ ; arrays inspected via `uv run --no-project --with numpy python -I`.
Line refs "F:n" = file:line in the kit. VERIFIED = read in kit/API; UNVERIFIED = not determinable from kit.

## 1. Zenodo record (VERIFIED, API JSON)
- Title: "Data repository for the 2022 AAPM deep-learning spectral CT grand challenge"
- Creator: Sidky, Emil (only one listed). DOI 10.5281/zenodo.14262737. License cc-by-4.0. Published 2024-12-02.
- Version: null (no version field). Related: isSupplementTo 10.1002/mp.16363 (Sidky & Pan, Med Phys 51:772-785, 2024). Grant NIH 1R01EB026282.
- Description says: 1000 training cases; float32 .npy (gzipped); Phantom_* 1000x512x512; *Transmission 1000x256x1024; *Images 1000x512x512 (neg-log + FBP).
- Files (name, bytes, md5 per API):
  - highkVpTransmission.npy.gz 842546719 57011ed1b0346d1062627a6ba86f6b7e
  - highkVpImages.npy.gz 716205150 f600ba6546c0f8fab3b227e3d470c5ed
  - lowkVpImages.npy.gz 717374937 8afab98fef99400d4f293f57281fa6a3
  - lowkVpTransmission.npy.gz 850758740 8c48d814d3db4c34f1cb96348cd1c9d9
  - starting_kit.tgz 102706887 7aaf6b55708a6576ebfa91e43ec80082
  - Phantom_Adipose.npy.gz 102183683 1d0c7239fa747a285c7ef1bcf684e3da
  - Phantom_Fibroglandular.npy.gz 113764766 746641e2910c231a39441d242bdfd458
  - Phantom_Calcification.npy.gz 6937257 2e04dc154dc36087a2404007c1b36baa
- md5 check starting_kit.tgz: expected 7aaf6b55...6ec80082, computed (`md5`) 7aaf6b55708a6576ebfa91e43ec80082 -> MATCH; size 102706887 matches.
- Archive listing (`tar tzvf`): 0 entries with absolute path or `..`; all under starting_kit/; no symlinks.

## 2. Kit file tree (VERIFIED)
starting_kit/: README (111 lines, overview + future-folder plan), metrics.py (scoring demo), startingkit_additionalInformation.pdf (3-page slide: objective, model equation, metrics), 10-case arrays Phantom_{Adipose,Fibroglandular,Calcification}.npy, TestPrediction_*.npy (= Phantom + Gaussian noise, demo only), {low,high}kVpTransmission.npy, {low,high}kVpImages.npy, .DS_Store.
model/: projection.py (line-intersection fan-beam projector, compares to stored sinogram), dualEnergyTransmission.py (spectral model), fanbeam_fbp_de.py (FBP that made *kVpImages), fanbeam_fbp_sino.py (FBP of tissue sinograms, ~1% discretisation error, F:9-10), model_data_50kVp.npy, model_data_80kVp.npy, Sinogram_{Adipose,Fibroglandular,Calcification}.npy (10x512x1024 float32, 512 views, noiseless).
README says validation = 10 cases and testing = 100 cases, inputs only (README:103-111).

## 3. Forward model (VERIFIED)
- Equation, dualEnergyTransmission.py:45 (PDF page 3 agrees: I_w = sum over E of s_w(E) exp(-sum_m mu_m(E) P x_m)):
  `T_kVp[view,bin] = sum_E dE * s_kVp(E) * exp(-mu_a(E)*Pa - mu_f(E)*Pf - mu_c(E)*Pc)`, dE=0.5 keV (F:35), P*=Sinogram_* line integrals of the tissue maps.
- POLYCHROMATIC, two spectra, no separate detector response term: spectrum is used as a plain weight => energy-integrating-style, no E factor, no detector efficiency (UNVERIFIED whether s already absorbs energy weighting; kit does not say).
- Spectra from model_data_{50,80}kVp.npy: float32 shape (5, N): rows = [energies, spectrum, mu_adipose, mu_fibroglandular, mu_calcification] (F:29-33).
  - 50 kVp: N=98, E=1.25..49.75 keV, step 0.5 (bin centres); nonzero from 6.75 keV; peak 35.75 keV.
  - 80 kVp: N=158, E=1.25..79.75 keV, step 0.5; nonzero from 6.75 keV; peak 59.25 keV.
  - Normalised so sum(0.5*s)=1.0 (computed 0.99999994; F:37-39 prints it). Hence incident I0=1, transmission in (0,1] (low 0.0084..1.0, high 0.0239..1.0 on 10 cases). Spectrum units/filtration/generator: UNVERIFIED (not stated).
- mu units: not stated in kit text. mu(E) values (row 2..4): E=1.25 keV: 895.2/1491.4/31862.9; E=20: 0.555/0.724/12.58; E=30: 0.295/0.352/3.976 (adi/fib/cal). 50 kVp last bin 49.75: 0.2022/0.2237/1.098; 80 kVp 79.75: 0.1712/0.1846/0.4813. mu rows identical at shared energies for both files (checked 20,30 keV).
- Noise: NONE in code. Transmission = exact exp, float32, max 0.99999994; no Poisson step anywhere in dualEnergyTransmission.py or elsewhere. Data are noiseless. Challenge-report mention of noise: UNVERIFIED.
- Images: kVpImages = FBP of -ln(T) (fanbeam_fbp_de.py:51-53), other-kVp views zero-filled, result x2 (F:184). Ranges (10 cases): low -0.054..1.70, high -0.032..0.909. Cupping + streaks are by design (F:49-50, 55).

## 4. Geometry (VERIFIED, projection.py:27-39, same constants repeated in fbp_de.py:29-38)
- Circular FAN-beam, flat linear detector. Source radius 50.0 cm; source-to-detector 100.0 cm (so iso-to-detector 50 cm, magnification 2).
- Image 512x512, side 18.0 cm x 18.0 cm -> pixel 18/512 = 0.03515625 cm; origin x0=y0=-9 cm, centred; pixel centre at x0+dx*(ix+0.5).
- Detector: 1024 bins; length = 2*tan(asin(9/50))*100 = 36.598 cm (computed), pitch 0.03574 cm at detector (0.01787 cm at iso). Bin centre u=u0+(i+0.5)du, u0=-L/2.
- Views: 512 full-circle views, ds=2*pi/512, s=sindex*ds from 0 to 2*pi (projection.py:35-36,67). Source (R cos s, R sin s); detector centre (R-SDD)(cos s, sin s); detector axis e_u=(-sin s, cos s) => counter-clockwise rotation.
- Interlacing (dualEnergyTransmission.py:13-21): 80 kVp (high) = even views [::2], 50 kVp (low) = odd views [1::2]; 256 views each; each kVp spaced 2*pi/256, low offset by one ds = 2*pi/512 from high (fbp_de.py:47). Sinogram arrays are [case, view, bin].
- Array index convention: image[ix, iy] (axis0 = x, axis1 = y), NOT row/col image convention (projection.py:110,118; fbp_de.py:135). Transmission [case, view, bin].
- FBP fov_radius 9 cm (circle inscribed in the square): images zero outside.

## 5. Materials (partly VERIFIED)
- Three: Adipose, Fibroglandular, Calcification. mu curves are tabulated per energy in model_data_*.npy rows 2-4 (see 3). Source of mu table (NIST? density?): UNVERIFIED, kit does not say. Reference densities: UNVERIFIED (none in kit).
- VOLUME FRACTION vs MASS DENSITY: kit never uses either word (grep fraction|density|mass|units over README, metrics.py, model/*.py: no hits). Best-supported answer: VOLUME-FRACTION-like dimensionless maps in [0,1], mu in linear attenuation (cm^-1). Evidence:
  1. Phantom_Adipose and Fibroglandular max = 1.0 exactly; Calcification max 0.738 (10 cases); all min 0.0, float32.
  2. Sum of the 3 maps: max 1.0000001, never >1.001; sum==1 (|s-1|<1e-3) in 60.4% of pixels, sum==0 in 35.5% (air background), 0<sum<0.999 in 4.1% (edge/partial-volume pixels, presumably antialiased boundary).
  3. Sinogram = sum of map*pixel-length in cm (projection.py:100,110: raysum += travPixlen*image); max total Pa+Pf+Pc = 16.02 cm < 18*sqrt2 chord => the maps behave as dimensionless per-pixel tissue fractions; mu must then be per-cm to make exp(-mu*P) dimensionless; mu(25 keV)~0.37 /cm for adipose is plausible as linear coefficient.
  Still INFERENCE, not stated text. If maps were g/cm3, adipose could not equal exactly 1.0 and sum==1 would not occur.
- Partition constraint: YES approximately: sum=1 inside tissue (60% of pixels, i.e. the interior of the breast), sum<=1 everywhere. Background/air class: implicit, not a 4th map; air = all three zero (35% pixels), mu(air)=0 implied by absence from the exponent.
- Calcification is sparse (mean 3.2e-4, only 717 unique values in image 0) and can coexist with the others in a pixel (calcification fraction included in the sum-to-1).

## 6. Array schema, full dataset (from README + Zenodo description + kit shapes; NOT downloaded)
| file | shape | dtype | meaning |
|---|---|---|---|
| Phantom_{Adipose,Fibroglandular,Calcification}.npy.gz | 1000x512x512 | float32 | [case,ix,iy] tissue maps, ~[0,1] (kit: 10x512x512) |
| {low,high}kVpTransmission.npy.gz | 1000x256x1024 | float32 | [case,view,bin] normalised transmission, I0=1, noiseless; low=50 kVp odd views, high=80 kVp even views |
| {low,high}kVpImages.npy.gz | 1000x512x512 | float32 | [case,ix,iy] neg-log FBP, linear-attenuation-like (cm^-1), unitless stated: UNVERIFIED units |
Sizes in the table are doc-stated; full-array value ranges are UNVERIFIED (only 10-case ranges above). Index order of the .gz = same as kit (same writer, README:13-33).

## 7. Metric (VERIFIED, metrics.py)
- Mean RMSE (L19): per image, sqrt( sum over 3 materials and all 512x512 pixels of (truth-pred)^2 / (3*512*512) ), then mean over images. No normalisation by dynamic range; absolute units of the maps. Pooled over materials (no per-material weighting).
- Worst-case ROI RMSE (L25-46): sliding 25x25 ROI, stride 1, over x,y in [0, nx-25); ROI RMSE = sqrt(sum_{3 mats,25x25}(diff^2)/(3*625)); ROIs skipped unless truth ROI max > 0.01 (all materials, ROI.max()). Max over ROIs and images. (Loop bound range(nx-roisize) omits the last ROI position: minor quirk.)
- Official challenge ranking combination of these two: UNVERIFIED (report not read).
- Ground truth: training Phantom_* public (1000 cases); validation (10) and testing (100) truth withheld per README:103-111. Whether truth was later released: UNVERIFIED (Zenodo has only the 8 files above, i.e. training only).

## 8. Discrepancies / notes
- Kit is consistent with 50/80 kVp (README:79-80, model_data names, spectra max E 49.75/79.75). No discrepancy found, but I did not read the Med Phys report (paper values UNVERIFIED).
- README says "128-view fbp images" in metrics.py header comment (L2) - stale copy from the 2021 sparse-view challenge; actual data are 256 views per kVp.
- README says arrays shown are 10 cases; Zenodo has 1000. Kit TestPrediction_* files are not in the Zenodo record.
- kit "low=50, high=80" naming; first interlaced view (idx 0) is HIGH kVp, easy to flip.
- Zenodo description typo "Phyiscs"; version field null.

## Top UNVERIFIED
1. Units/meaning of maps and mu (volume fraction + cm^-1 inferred, never stated); mu source and densities.
2. Spectrum physical definition (photon counts vs energy fluence, filtration) and any noise in challenge-time data (kit: none).
3. Full-dataset value ranges/dtype of the 7 gz arrays; official ranking rule from the report.
