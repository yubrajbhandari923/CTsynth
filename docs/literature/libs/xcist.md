# XCIST / CatSim (github.com/xcist/main, package gecatsim) - evaluated 2026-10-11

Verdict: **TEST ORACLE ONLY** (optional, off-repo; run in a separate venv to cross-check spectra/noise/geometry, never a mandatory dependency).

## Licensing (first criterion)
- VERIFIED (gh api .../contents/LICENSE): root `LICENSE` = "BSD 3-Clause License, Copyright 2024, GE Precision HealthCare. All rights reserved." Standard redistribution terms (retain notice; binaries reproduce notice; no endorsement with contributor names). No non-commercial restriction. `licenses/` dir also holds `LICENSE` and `LICENSE_GEANT4`. setup.py `license='BSD 3-Clause License'`; a header still says "Copyright 2020, General Electric Company" (inconsistent, minor).
- Geant4 cross-section data used (README: "includes software developed by Members of the Geant4 Collaboration"); Geant4 license in licenses/LICENSE_GEANT4 (text not reviewed, UNVERIFIED terms). Phantoms such as XCAT are separate and typically restricted (UNVERIFIED).
- BSD permits reuse, but copying code is still not needed; use as external tool.

## Platform / dependencies
- VERIFIED: core is prebuilt C++ `gecatsim/lib/libcatsim_macos.so` plus `clib_build/MakeMacOS`; setup.py auto-builds only on linux. install_requires numpy, scipy, matplotlib, tqdm. No CUDA, no torch. macOS arm64 binary architecture UNVERIFIED (not downloaded/inspected). CPU-only, single heavyweight config-driven pipeline.

## Maintenance
- VERIFIED: created 2020-09-03; last commit 2026-03-27 (push 2026-05-05); releases v0.1.0 (2020) and v1.0.0 (2023-12-19) only; setup.py version 1.6.8 (not released as tag); 28 open issues; 81 stars; GE-backed, NCI U01CA231860 funded.

## Tests
- VERIFIED: `gecatsim/tests/test_catsim/` (test_C_DD3Proj, CrossSectionDB_Initialize, CrossSectionMAC_ByProc_Get, FORBILD phantom); `gecatsim/dose/pyfiles/unittest/*.mat` in/out pairs; CI .github/workflows/build.yml (ubuntu, py3.10, flake8, pytest; functional tests gated by repo variable). Coverage appears thin relative to size.

## Spectral features (VERIFIED from paths/README; depth UNVERIFIED)
- Polychromatic tube spectra, bowtie/flat filters, material cross-section tables (Geant4-based MAC), detector models (flat panel, third-gen curved, EID/PCD options, crosstalk, DQE), scatter, Poisson+electronic noise, focal spot/detector blur, dose module, 3-D and analytic phantoms. kVp switching and PCD thresholds: UNVERIFIED, check wiki (github.com/xcist/documentation/wiki).

## Projector interface
- C library via config files (`.cfg` / python dict); no torch, no autograd, no adjoint exposed. Sinogram-in/out batch simulator.

## Fit
- Not differentiable, 3-D oriented, heavy; wrong fit for the model. Best use: independent oracle for attenuation/noise/beam-hardening numbers on a simple water+bone/iodine cylinder, run outside the repo env.
