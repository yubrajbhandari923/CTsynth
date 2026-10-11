# dex-ct-sim (gjadick/dex-ct-sim) - evaluated 2026-10-11

Verdict: **TEST ORACLE ONLY** (its own MIT code could be run off-repo to cross-check Siddon/Gauss-Newton results; do not vendor; do not depend). Submodule/data licensing is unclear.

## Licensing (first criterion)
- VERIFIED (gh api .../contents/LICENSE): file `LICENSE`, "MIT License, Copyright (c) 2023 Giavanna Jadick", standard permissive text; GitHub metadata spdx MIT. No stated non-commercial restriction.
- CAVEAT, no license found for the submodule: `.gitmodules` points to github.com/gjadick/x-tomo-sim (xtomosim/), which holds the forward projector, FBP and spectrum code. VERIFIED its root listing (.gitignore README.md back_project.py bhc.py forward_project.py input main.py requirements.txt system.py xcompy.py) has no LICENSE and API license metadata is null: all rights reserved by default, cannot reuse code; could only be used as a conceptual reference or contacted for permission. The MIT file in the parent arguably does not cover it (UNVERIFIED legal reading).
- `xcompy.py` header credits "daverigie" (2012), NIST XCOM data wrapper; its license is UNVERIFIED.
- input/ contains binary spectra (80/120/140 kV, MV) and detector response (.bin); provenance/license UNVERIFIED.

## Platform / dependencies
- VERIFIED: x-tomo-sim requirements.txt pins `cupy_cuda11x==12.2.0`, numpy 1.26.2, scipy 1.11.4, matplotlib 3.8.0. `matdecomp.py` does `import cupy as cp` at module top (plus a numpy `optimize_sino_cpu` fallback), so import fails on macOS arm64 without editing. CUDA 11 pin = effectively Linux/NVIDIA.

## Maintenance
- VERIFIED: created 2023-05-22; last push 2023-12-18 (xtomosim 2024-10-08); no releases; 2 open issues; 8 stars. Effectively dormant.

## Tests
- VERIFIED tree: no tests directory; only main.py driver and plots.py.

## Spectral features (VERIFIED README / code grep)
- Polychromatic 80/120/140 kV and MV spectra at fixed dose; energy-integrating and PCD (Si 30 mm) detector response files; dual-energy with dose split (e.g. detunedMV + 80kV); Gauss-Newton basis decomposition (Rigie & La Riviere 2015); NIST attenuation via xcompy. Noise: dose-matched Poisson (UNVERIFIED exact model). kV switching: separate acquisitions only (UNVERIFIED).

## Projector interface
- Siddon ray tracing + fan-beam FBP in numpy/cupy. No torch, no autograd, no exposed adjoint (VERIFIED: imports in main.py are numpy/matplotlib only).

## Fit
- Poor for a differentiable torch model. Value is as a small, readable dual-energy Siddon + GN reference for sanity checks (run in a scratch env, compare numbers, copy nothing).
