# Library evaluation (read-only; 2026-10-11)

Evidence: subagent reads of repos/PyPI (`gh api`, web); nothing was built except two xraylib/SpekPy probes. Verdicts are Opus's reading of the per-library notes.

| Library | License | Verdict | Why |
|---|---|---|---|
| LEAP | MIT | test oracle only (opt-in, skip if absent) | not on PyPI; source build makes torch mandatory; 3D/global-state API; no spectral physics |
| XrayPhysics | MIT | offline table generator / cross-check | not on PyPI, ctypes source build, no tests, not differentiable; has EPDL97 + spectra |
| xraylib 4.3.0 | PyPI says BSD (text unread) | **optional**: offline generator of frozen μ/ρ tables | arm64 wheels cp310–cp314; matches NIST water at 30/60/100 keV to 4 s.f. (3 points only); scalar-energy API |
| SpekPy 2.5.4 | MIT | **optional**: spectra generator for experiments | pure-python wheel, installs on 3.14; pulls scipy+matplotlib; model details UNVERIFIED |
| ct_laboratory | **none found** | conceptual reference only — never copy code | all rights reserved by default; unconditional CUDAExtension build |
| dex-ct-sim | MIT root; core submodule x-tomo-sim unlicensed | hand-run sanity check only | imports cupy; dormant since 2023 |
| XCIST | BSD-3 (Geant4 data terms unreviewed) | optional cross-check in a separate venv | CPU, maintained, no autograd/adjoint |

## Decision (D-E)
1. **No third-party physics or projector is a runtime dependency.** `ctsynth.physics` is a small torch reference implementation (autograd native, CPU-first).
2. **Real attenuation data enter as frozen, versioned tables** (`.npz` + provenance JSON: library, version, database, NIST spot-check), generated offline with xraylib (T-003). Runtime never imports xraylib.
3. Spectra for experiments: SpekPy output frozen the same way; a parametric toy spectrum is the fallback.
4. Oracles (LEAP, XCIST) are opt-in tests guarded by `importorskip`; absence never fails CI.

Unverified: library correctness beyond the stated spot-checks; macOS arm64 builds of LEAP/XrayPhysics/XCIST; SpekPy tube model and normalisation; legal reading of licences (not legal advice).
