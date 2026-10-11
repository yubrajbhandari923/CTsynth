# Vision

**Goal:** a platform that generates editable, physics-grounded virtual CT patient populations.

## Patient representation (target, not yet built)
1. **Anatomy** — organ/tissue label volumes (geometry, topology).
2. **Material fields** — per-voxel elemental/material composition and density (the physical truth CT measures).
3. **Pathology** — lesions as edits to anatomy + material fields.
4. **Physiology (optional)** — contrast kinetics, motion.

Acquisition + reconstruction (spectral forward model, noise, geometry) map this to synthetic sinograms/images.
"Editable" means a change to one layer (e.g. add a calcification) propagates through the physics to measurements.

## Why P01 first
Material inference (measurements → material fields) is the inverse of the forward model the platform needs. Building and validating it
first fixes the material-field and spectral conventions everything else depends on.

## Non-goals (now)
Full anatomy generation, pathology models, physiology, clinical claims, scanner-vendor-specific calibration, GPU/HPC training code.
