# ct_laboratory (tivnanmatt/ct_laboratory) - evaluated 2026-10-11

Verdict: **CONCEPTUAL REFERENCE ONLY** (no license). Do not copy code, do not depend on it.

## Licensing (first criterion)
- No license found: all rights reserved by default, cannot reuse code; could only be used as a conceptual reference or contacted for permission (owner: Matt Tivnan). 
- VERIFIED (gh api repos/tivnanmatt/ct_laboratory): `license: null`. Root listing: .dockerignore .gitignore Makefile README.md benchmark ct_laboratory env_notes.txt print_code.py python_examples setup.py src. No LICENSE/COPYING/NOTICE anywhere in the 395-path recursive tree. README has no license text. setup.py has no `license=` field.
- Note: the repo vendors/merges code from the gmi package (Generative-Medical-Imaging-Lab/gmi); that upstream's license was NOT checked (UNVERIFIED), so the merged part is doubly unclear.

## Platform / dependencies
- VERIFIED setup.py: unconditionally imports `torch.utils.cpp_extension.CUDAExtension` and builds `ct_laboratory._C` from `src/*.cu` (nvcc). `pip install` on macOS arm64 will fail without CUDA toolkit. env_notes.txt targets conda + CUDA 12.4 + torch cu124 + wandb, diffusers, etc.
- VERIFIED README: pure-PyTorch reference projectors exist (`ct_projector_*_torch.py`, "any device, slow"); spectral physics "no CUDA extension required". UNVERIFIED: whether the package imports cleanly without `_C` on macOS (not run, nothing installed).
- spekpy used for tube spectra (VERIFIED README); its license/portability UNVERIFIED.

## Maintenance
- VERIFIED: created 2025-04-23; last push 2026-10-10 (very active, but a research monorepo that is churning: merges gmi, diffusion, etc.); 0 releases; 0 open issues; 5 stars; single-author style. API stability: low.

## Tests
- VERIFIED README: ad hoc scripts in python_examples/ (line-integral, module, fan/cone-beam, StaticCT); pytest suite `python_examples/gmi_merge_tests` (README claims 345 passing, CPU) covers the merged linalg/sde/diffusion stack, not the spectral physics. Spectral model test coverage: UNVERIFIED.

## Spectral features (VERIFIED README "Spectral X-ray physics" + tree paths)
- Polychromatic: spekpy tube sources, mono source, Al/material filters; `DualExposureXraySource` for kV-switching studies.
- Material tables: `MaterialDatabase`, basis bases (PECS, water-iodine, water-calcium, soft tissue).
- Detector: energy-integrating and dual-exposure detectors, GOS interaction, blur, scatter (zero/constant). Poisson noise in `XraySystem`.
- Photon counting: no PCD module seen in the tree (only EnergyIntegrating/DualExposure). Newton optimizer for decomposition.

## Projector interface
- VERIFIED README: Siddon-type 2D/3D, exact adjoint, `torch.autograd.Function` wrappers, `nn.Module` layers, fan/cone/StaticCT; `CTSystem` composes projector + XraySystem differentiably; basis tensors `[n_rays, n_materials]`.

## Fit for small differentiable 2D forward model
- Design is exactly the shape we want (basis line integrals -> spectrum -> detector -> Poisson), so it is a useful design reference for module boundaries and tensor shapes. Reimplement independently from papers/NIST, not from this code.
