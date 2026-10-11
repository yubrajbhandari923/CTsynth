# xraylib evaluation (2026-10-11)

Verdict: **optional dependency + test oracle** (never mandatory; pure lookup of cross sections, no heavy deps).

Tags: V = VERIFIED (PyPI JSON / probe / NIST page), U = UNVERIFIED (from memory, not checked this session).

## Packaging
- V: PyPI xraylib 4.3.0 released 2026-08-19; prior 4.2.1 (2025-12-21), 4.2.0 (2025-08-27), 4.1.5 (2024-08). Active.
- V: 4.3.0 has `cp310..cp315` `macosx_11_0_arm64` wheels, incl. cp312 and cp314 (+cp314t). requires_python >=3.10.
- V: `uv run --no-project --python 3.12|3.14 --with xraylib` installs and runs; version string 4.3.0.
- V: PyPI classifier "BSD License". U: exact text (BSD-3 by Schoonjans; check LICENSE in repo before redistributing data).
- U: bundles compiled C library via SWIG; no runtime deps. Numpy not required.

## Data provenance (all U; cite from xraylib docs / Schoonjans et al. 2011, Spectrochim. Acta B 66:776)
- U: CS_Total = photo + Rayleigh + Compton, built from Livermore EPDL97 cross sections (not XCOM). CS_Total_Kissel uses Kissel photo-cross sections.
- U: Edges (EdgeEnergy) from Elam/Ritchie/Rowe tables; Z=1..120 supported.
- V (probe): K edge of I = 33.1694 keV (literature ~33.17 keV).
- V: CS_Total_CP(formula, E) is mass-weighted sum for a compound formula string; compounds via NIST names (`GetCompoundDataNISTByName`) U.

## Interface fit
- V: units cm^2/g, energy keV. Scalar-only: `CS_Total(8, np.array([30.,60.]))` raises `TypeError: argument 2 of type 'double'`. Need `np.vectorize` or a list comprehension; fine for a precomputed (E x material) table.
- V: CS_Total(Z, E) is the elemental value; compound by formula needs CS_Total_CP or mass-fraction mixing by hand (mixture rule, no molecular-binding effects).

## Probe (xraylib 4.3.0, py3.12, cm^2/g, total incl. coherent)
| E keV | Z=8 | Z=53 (I) | H2O (CS_Total_CP) | NIST XCOM water |
|---|---|---|---|---|
| 30 | 0.37793 | 8.5617 | 0.37559 | 0.3756 |
| 60 | 0.19073 | 7.5770 | 0.20590 | 0.2059 |
| 100 | 0.15514 | 1.9422 | 0.17075 | 0.1707 |
- V: NIST water (physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html, "mu/rho" col) = 0.3756 / 0.2059 / 0.1707 at 30/60/100 keV; xraylib matches to 4 s.f. (the 0.2059 hint was correct).
- Note: Z=8 alone is not water; do not compare element O to water.

## Validation / risk
- U: papers cite xraylib vs XCOM agreement of a few % in the diagnostic range. Only 3 points checked here.
- V: gaps unprobed: energies near edges (K-edge of I 33.17 keV step), < 1 keV, Z-dependence beyond I.
- Use as: independent oracle for a NIST/XCOM-table-based attenuation module (tests with ~0.1-1% tolerance, the tolerance not yet justified), or optional backend behind an import guard.
