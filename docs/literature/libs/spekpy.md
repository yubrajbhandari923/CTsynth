# SpekPy evaluation (2026-10-11)

Verdict: **optional dependency** (x-ray tube spectrum generator; pure Python but pulls scipy+matplotlib). Not mandatory; keep a built-in/tabulated spectrum path.

Tags: V = VERIFIED (PyPI JSON / probe), U = UNVERIFIED.

## Packaging
- V: PyPI spekpy 2.5.4, released 2026-03-31 (2.5.3 on 2025-10-14, 2.0.13 on 2024-08). Active.
- V: license MIT (Copyright 2019 Bujila & Poludniowski); no restrictions beyond notice.
- V: `py3-none-any` wheel, requires_python >=3.6, so arm64/macOS and 3.12/3.14 work. V: `uv run --python 3.12` and `--python 3.14` both install and import (13 packages: numpy, scipy, matplotlib, pillow, fonttools...).
- V: heavy-ish deps (scipy, matplotlib) are the only cost; no CUDA. Install is slow on first run (~2 min with cold cache).

## Model / provenance (all U unless noted)
- U: SpekPy v2 is a Python port/extension of Poludniowski's PENELOPE-based model ("SPEKTR/SPEKCALC" lineage): Poludniowski 2007 Med Phys 34:2175; Poludniowski & Evans 2007 Med Phys 34:2164; Omar et al. 2020 (SpekPy v2, Phys Med Biol / Med Phys; verify citation with reference-lookup before use); Poludniowski et al. 2021 SpekPy v2.0 (Med Phys).
- U: tungsten default anode; Mo, Rh also; kVp ~20-300 kV; anode angle `th` degrees (probe passed th=12 explicitly).
- U: filtration via `filter('Al', mm)`; materials from built-in list with NIST attenuation data; custom material possible.
- U: authors request citing the SpekPy papers; check the wiki for exact wording.

## Probe (spekpy 2.5.4, py3.12): `Spek(kvp=100, th=12); filter('Al',2.5)`
- V: `get_spectrum(edges=False)` -> (k, f) numpy arrays; k = 1.25..99.75 keV midbins, 198 bins, 0.5 keV step. So kVp bound = 100 keV max, no photons above.
- V: sum(f) = 5.02e8, sum(f)*dk = 2.512e8 = `get_flu()` (2.512e8). So default f is **differential fluence per keV** (photons cm^-2 keV^-1), and fluence = integral. Pass `flu=...`/`diff=False` to change (docstring V). Reference distance 100 cm and per-mAs normalisation are U (default `z=100`, `mas=1`; check wiki).
- V: `get_kerma()` 118.9 (uGy per mAs at 1 m is U), mean E 49.14 keV, HVL1 Al 3.458 mm.
- U: no independent check of spectrum shape vs measured/TASMICS/IPEM 78 here.

## Interface fit
- V: numpy float arrays, keV, fluence units as above. Easy to normalise to unit total weight: `w = f*dk / flu`.
- Spectrum must be rebinned/interpolated to the project energy grid; bin integral preserved if done on `f*dk`.

## Open items
- Verify citation/model details against the bitbucket wiki; compare 100 kVp spectrum to an independent source (IPEM 78 or TASMICS) before calling it supported.
