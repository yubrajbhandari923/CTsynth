# DukeSim collaboration — information to request (priority order)

Nothing about DukeSim formats, API, licence or datasets is verified (only the 2019 TMI paper — DOI
10.1109/TMI.2018.2886530 — was read in full; Salinas 2025 and Valand 2026 abstract-level). Treat every item as a question.
XCIST (docs/xcist.md) is the working independent producer meanwhile.

## A. Scientifically essential (cannot start without)
1. **Paired truth**: are per-voxel material maps (not just labels) available for the simulated volumes? Representation:
   volume fractions, partial mass densities, or labels + a lookup table? Units; do fractions sum to 1 (is air a material)?
2. **Material definitions**: composition and reference density per material, and the attenuation source (xraylib /
   NIST XCOM / Geant4) — needed to avoid mixing mass/linear attenuation or fraction/density.
3. **Acquisition spec per volume**: kVp, filtration/bowtie, spectrum file, mA/dose, geometry (SID/SDD, detector size,
   views, helical pitch), detector response (EI vs PCCT thresholds), focal spot.
4. **Outputs available**: noiseless and noisy projections? reconstructed images (kernel, slice thickness, units HU/mu)?
   air scans for log normalisation? scatter on/off per dataset.
5. **Coordinates**: voxel spacing, orientation, origin/affine of material maps vs reconstructions; confirmation they
   are co-registered (or the transform).
6. **Access and sharing rules**: who may hold the data, whether it can be used for training/publication, storage
   restrictions (HPC vs laptop), licence for DukeSim/XCAT.

## B. Needed for rigorous validation
7. Source phantom diversity (number of phantoms/patients, anatomy, pathology, sex/BMI), identifiers for patient-level splits.
8. Scanner/protocol variation actually simulated (≥2 kVp pairs, dose levels, kernels) for held-out-protocol tests.
9. Scatter model details (MC histories, ASG), detector noise model, any approximations (fidelity flags).
10. Reproducibility: software version/commit, config files, random seeds, runtime per case.
11. Whether Valand 2026 / Salinas 2025 simulation configs or datasets are obtainable ("available on reasonable request").
12. Contrast handling: static iodine concentration vs phase-dependent perfusion (do not conflate).

## C. Minimum viable paired dataset (start supervised work)
≥ 20 distinct phantoms, 2 slices–volumes each is not enough; need: **per-voxel material truth with declared semantics,
noisy and noiseless CT (or projections) at ≥2 kVp for the same phantoms, spectra and geometry files, voxel geometry,
patient-level identifiers.** One scanner model, one kernel is acceptable for a first run.

## D. Ideal dataset
≥ 100 patients/phantoms; ≥ 3 protocols incl. held-out kVp pair and dose levels; raw spectral/PCCT projections with
detector model; scatter-on/off pairs; recon with two kernels; per-voxel truth densities; lesion annotations; documented
fidelity limits; permission to share derived models.
