# TrueCT pilot: NIfTI, previews, projections, LEAP reconstruction

Data (read-only): `/home/yb107/datasets/truect_pilot/`. Outputs: `/home/yb107/datasets/truect_pilot/outputs/` (this file is copied there).
Scripts: `scripts/truect/` in the repo (synced to `~/ctsynth_truect/` on DRCC). **No phantom-to-projection pairing is assumed anywhere**;
`dcm_000`, `dcm_001`, `copd_22728`, `copd_21535` are kept separate.

## What status means
- **METADATA**: stated in DICOM tags or the challenge PDF.
- **MEASURED**: derived here from the data, with evidence in `reports/`.
- **INFERRED**: judged from anatomical appearance; needs your visual confirmation.
- Every NIfTI carries `provisional` in its name or a `PROVISIONAL` description because the orientation is INFERRED, not METADATA.

## Files (outputs/nifti)
Frame for all `provisional` files: array axes are listed as the direction each index *increases toward*; the affine is origin-centred (no physical origin is known).

| file | shape | voxel mm | array axes | units | notes |
|---|---|---|---|---|---|
| `reference_ct_provisional.nii.gz` | 192x768x768 | 0.6 iso | (I,P,L) | HU **assumed** (int16, no rescale tags) | supplied recon, 192 Secondary-Capture DICOMs with no spatial tags; 0.6 mm is MEASURED, not the 0.5476 mm/0.976 mm in the PDF |
| `reference_leap_recon_mu_per_mm.nii.gz` | 192x768x768 | 0.6 iso | (I,P,L) | mu, 1/mm | our LEAP recon of the 3000 reference projections |
| `reference_leap_recon_HUest_refcal.nii.gz` | same | same | same | HU **estimate** = 1000(mu/0.019273-1) | mu_water fitted on the reference case, see below |
| `reference_material_labels_provisional.nii.gz` | 320x860x860 uint8 | 0.25 iso | (S,L,P) | material ID | full resolution |
| `reference_density_provisional.nii.gz` | 320x860x860 | 0.25 iso | (S,L,P) | g/cm3 | LUT from `reference_phantom_mat.xlsx` |
| `copd_{22728,21535}_material_labels_provisional.nii.gz` | native (z,y,x) | 0.25 (ASSUMED) | (S,L,P) | material ID | full resolution, uint8 |
| `copd_{...}_material_labels_s3_provisional.nii.gz` | stride 3 | 0.75 | (S,L,P) | material ID | nearest-index subsample, light copy for viewing |
| `copd_{...}_density_provisional.nii.gz` | stride 3 | 0.75 | (S,L,P) | g/cm3 | LUT from the case's own xlsx; cross-checked against its `.in` table (0 mismatches, 0 unmapped labels) |
| `lung_18460_*`, `liver_279_*` | as COPD | 0.25 (ASSUMED) | (S,L,P) assumed | | lower priority; orientation convention copied from COPD, looks consistent but unchecked |
| `copd_dcm_000_recon_mu_per_mm.nii.gz` | 526x768x768 | 0.6 iso | (I,P,L) | mu, 1/mm | **dcm_000 only**, ~0.9 GB |
| `copd_dcm_001_recon_mu_per_mm.nii.gz` | 585x768x768 | 0.6 iso | (I,P,L) | mu, 1/mm | **dcm_001 only**, ~1 GB |
| `*_HUest_refcal.nii.gz` | same | same | same | HU estimate | float32 versions of the above |
| `*_HUest_refcal_int16_cropped.nii.gz` | cropped to body | 0.6 iso | (I,P,L) | HU estimate, int16 | ~300 MB each: **use these in Slicer on a Mac** |
| `*_legend.json` | | | | | label ID -> name, density, MC id; unmapped labels |
Per-NIfTI JSON next to each file holds shape, voxel size, affine and `aff2axcodes`. `projections/` holds native projections (below).

The requested names `copd_dcm_00X_recon.nii.gz` were deliberately replaced by `..._recon_mu_per_mm` (physical) and `..._HUest_refcal` (estimate) so the units are in the name.

## Findings
**Phantom storage (MEASURED, `reports/survey_phantoms.json`).** All phantoms are C-order `[z, y, x]` uint8; filename dims read (nx, ny, nz)
except the liver file, which reads (ny, nx, nz). Neighbour-agreement score 0.99-1.0 for the chosen layout vs <=0.69 for every other reshape.
The reference phantom is **not anatomy**: a 200 mm diameter, 80 mm long water cylinder with 4 inserts (1.19, 0.91, 1.52, 1.0 g/cm3).

**Orientation.**
- Reference phantom vs supplied recon (MEASURED, `reports/reference_insert_test.json`): of 16 axis mappings only one reproduces the insert
  densities in the recon (+120 / -108 / +857 HU for 1.19 / 0.91 / 1.52 g/cm3; all others ~0 HU). Under the standard DICOM display assumption
  (row down = posterior, column right = patient left) the phantom array is (z,y,x) = (S,L,P) if recon slice index runs inferiorly (slice direction is
  NOT in the DICOM and is therefore INFERRED; the in-plane directions and the slice sense are only relative to each other).
- COPD phantoms (INFERRED, `reports/orientation_analysis*.json`, `previews/copd_*_labels.png`): sternum at low x and spine at high x (x increases posterior),
  descending aorta at higher y than the spine (y increases to patient left), scapulas/manubrium/aortic arch at high z (z increases superior).
  The resulting mapping is a proper rotation (not a mirror). Heart apex, larger right lung and liver are visible on the expected sides.
  Lung-size asymmetry was clear for 22728 and weak for 21535 (about 4%, wrong sign), so L/R rests mainly on the aorta marker.
- LEAP recon of `dcm_000` and `dcm_001` (INFERRED, `previews/copd_dcm_00*_HUest_refcal_*.png`): displayed through the frame derived from the reference
  case, both give non-mirrored anatomy (heart apex left, larger right lung, liver inferior right, spine posterior, apex up). That is an independent
  consistency check of the chain, but it is still appearance-based.
- **Needs your confirmation:** left/right and superior/inferior for every file. Use the Slicer steps below and the contact sheets.

**Voxel size.** Reference 0.25 mm is from its filename (the PDF does not state it). COPD/lung/liver 0.25 mm is ASSUMED (not documented; body sizes
at 0.25 mm are plausible: 22728 chest 349 x 254 mm, 21535 390 x 210 mm). Treat COPD physical sizes as unverified.

**Projections (METADATA, `outputs/projections/*/summary.json`).** DICOM-CT-PD, uint16, stored as `(views, 900 channels, 64 rows)`.
Value = `slope*raw + intercept` per view = beam-hardening-corrected (4th-order water polynomial per the PDF) post-log line integral, unitless; raw is kept
untouched in `raw_u16.npy`. Cylindrical detector, 1 mm pixels at SDD 1050 mm, SOD 575 mm, 1000 views/rotation, angle step 2*pi/1000, table advance 0.03505 mm/view
(35.05 mm/rotation, pitch 1.0), no flying focal spot. Series: reference 3000 views (105.1 mm), dcm_000 9000 views (315.4 mm, mAs vector), dcm_001 10000 views (350.5 mm, mAs vector).
Saturated samples (raw 0 or 65535): about 2e-5 of the data per series. `*_projection_tensor_NONANATOMICAL.nii.gz` are views/5 x channels/3 x rows tensors, NOT anatomy.

**LEAP (built from source: commit 0c8846f on DRCC, CUDA 12.2, sm_86; the prebuilt .so needs GLIBC 2.34, the nodes have 2.31).** Curved-detector helical cone-beam FBP via `tomographicModels.FBP` (pitch != 0 selects LEAP's helical path:
parallel-ray derivative + Hilbert filter + weighted backprojection, not the axial ramp/FDK filter; default settings, no kernel tuning), 768x768 in-plane at 0.6 mm, 0.6 mm slices; ~20 s per case on one A5000.
Reference validation against the supplied recon (`reports/mu_water_calibration.json`, `previews/reference_leap_vs_supplied.png`):
- Sign conventions chosen by search (`reports/recon_search.json`): rotation sign -1, channel order as stored, detector rows flipped; slice order reversed to the DICOM order.
- In-plane pixel 0.600 mm: body diameter ratio LEAP/supplied = 1.0003. The 80 mm phantom body occupies slices 30-161 in both volumes (pitch, z scale and centring agree).
- mu_water = 0.019273 mm^-1 (0.1927 cm^-1); water-region mean difference -1.5 HU.
- Insert means, supplied vs LEAP-HU-estimate: 1.19 g/cm3: 116 vs 124 HU; 0.91: -100 vs -97; 1.52: 856 vs 849.
- Smoothed (sigma 2 voxels) in-body RMSE 5.5 HU. Unsmoothed RMSE 62 HU is noise: water SD 29 HU (supplied) vs 45 HU (LEAP, helical FBP default settings).
- **Empirical alignment, not a measured geometry:** the detector centre is shifted -1 channel from the middle and the volume offset is (-0.1, -0.3) mm, fitted to the supplied recon
  (RMSE 8.8 -> 5.8 HU; `reports/center_test*.json`). No DICOM tag provides these. A pure translation did not give the same improvement (9.96 HU), so the detector centre really differs.
- This is **not a pixel-equivalence claim**: different algorithm, filter and noise texture.

**Calibration caveat.** LEAP output is mu in 1/mm and is the primary quantity. `HUest` uses mu_water from the reference case only and assumes the same scanner water calibration
applies to the COPD series (same scanner and BHC recipe per the PDF; not verified per case). Slices near the first and last table positions may have incomplete helical coverage (not quantified here).

## Unresolved
1. Absolute L/R and S/I of every phantom and recon (appearance-based only).
2. COPD/lung/liver voxel size (assumed 0.25 mm).
3. Supplied reference recon: HU interpretation, slice direction, and its true z spacing (0.6 mm measured, PDF says 0.5476 mm).
4. Detector centre and volume offset (empirical).
5. HU calibration for dcm_000/dcm_001 (transferred from the reference case).
6. Which phantom produced which projection series: out of scope by instruction.

## Download to your Mac
Needs Duke VPN or network access to the login node. Light set (about 1 GB):
```bash
mkdir -p ~/TrueCT && cd ~/TrueCT
R=yb107@plp-drcc-login2.dhe.duke.edu:/home/yb107/datasets/truect_pilot/outputs
SSH='ssh -p 8022 -i ~/.ssh/id_ed25519_duke_hpc'
rsync -avP -e "$SSH" $R/previews $R/reports $R/README.md .
rsync -avP -e "$SSH" --include='*/' \
  --include='reference_ct_provisional.nii.gz' --include='reference_*provisional.nii.gz' \
  --include='*_s3_provisional.nii.gz' --include='copd_*_density_provisional.nii.gz' \
  --include='*_HUest_refcal_int16_cropped.nii.gz' --include='*_legend.json' --include='*.json' \
  --exclude='*' $R/nifti/ nifti/
# optional, large: full-resolution labels (~140-160 MB each) and float32 mu volumes (~0.35-1 GB each)
rsync -avP -e "$SSH" --include='*_material_labels_provisional.nii.gz' --include='*_mu_per_mm.nii.gz' --exclude='*' $R/nifti/ nifti/
```
(An `~/.ssh/config` entry `Host plp-drcc-login2` with Port 8022 and your key makes the `-e` option unnecessary.) Open `previews/*.png` or `reports/index.html`.

## Viewing in 3D Slicer
1. File > Add Data > Choose File(s), select the `.nii.gz` files. For `*material_labels*` tick **LabelMap** under Show Options (or use Segment Editor import). Slicer reads the NIfTI sform as RAS, so it shows the orientation the file claims.
2. CT recon and supplied CT: Volumes module > Display > preset CT-Lung (about -1400..200) or CT-Abdomen/Soft-tissue (about -160..240).
3. Align the same case: reference CT and reference phantom share the frame (different grids); COPD recons and COPD phantoms share the frame convention but **must not be paired**.
4. Check anatomy in the three slice views (R/L/A/P/S/I letters are drawn from the affine). If you find a flip, tell me which axis; the fix is a one-argument rerun of the converters (`AXCODES`).
5. Projection tensors: load as a volume but remember the axes are (view/5, channel/3, row), not anatomy.

## Rerun
```bash
rsync -a scripts/truect/ plp-drcc-login2:ctsynth_truect/       # login shell is needed for sbatch: scripts use '#!/bin/bash -l'
ssh plp-drcc-login2 'cd ~/ctsynth_truect
./run_cpu.sh survey 4 24G 00:30:00 survey_phantoms.py
./run_cpu.sh refct 4 8G 00:15:00 convert_reference_ct.py 0.6 IPL provisional 0.6      # px, axcodes, tag, dz
./run_cpu.sh ph 8 40G 01:00:00 convert_phantom.py copd_22728 SLP 0.25 3 provisional   # case axcodes voxel_mm stride tag
for s in reference dcm_000 dcm_001; do ./run_cpu.sh proj_$s 12 24G 00:40:00 convert_projections.py $s 12
                                      ./run_cpu.sh pviz_$s 4 16G 00:30:00 visualize_projections.py $s; done
./run_gpu.sh rec 8 64G 02:00:00 recon_final.py dcm_000 copd_dcm_000_recon 768 0.6 0.6   # series name nxy vox dz [nz [col_shift]]
./run_cpu.sh light 4 48G 00:30:00 make_light_recon.py copd_dcm_000_recon
./run_cpu.sh report 4 48G 00:40:00 make_report.py'
```
LEAP build (once): `sbatch --gres=gpu:1 build_leap.sh` after `git clone --depth 1 https://github.com/LLNL/LEAP ~/ctsynth_truect/leap/LEAP`.
Order for the reference calibration: `recon_final.py reference ... 192` then `compare_reference.py` (writes `mu_water_calibration.json`), then rerun the others to get `HUest`.
Evidence scripts: `orientation_analysis.py`, `orientation_analysis2.py`, `reference_insert_test.py`, `recon_search.py`, `center_test.py`, `measure_shift.py`.
