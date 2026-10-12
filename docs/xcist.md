# XCIST as an independent forward-data producer

Status: installed and verified (EXP-009, EXP-010, EXP-011, EXP-012). XCIST (`gecatsim` 1.6.9, BSD-3, Wu et al.
PMB 2022, DOI 10.1088/1361-6560/ac9174 — abstract-level read only) produces *measurement data*; material truth is
always the input phantom. It does not replace the differentiable PyTorch model. Code: `src/ctsynth/paired/xcist.py`.

## Install (optional, isolated; never in the core env)
- Laptop: `uv venv --python 3.11 .venv-xcist && uv pip install --python .venv-xcist/bin/python "numpy<2" gecatsim==1.6.9 -e . pytest`
  (numpy<2 is a precaution: gecatsim imports `numpy.matlib`). macOS arm64: the prebuilt FDK library links
  `/opt/homebrew/.../libgomp.1.dylib`; without it reconstruction fails (RUN-0014). Set `XCIST_LIBGOMP_DIR` to a
  directory holding an arm64 `libgomp.1.dylib`; `reexec_with_libgomp()` then re-execs with
  `DYLD_FALLBACK_LIBRARY_PATH` (SIP strips `DYLD_*` across `/bin/sh`, so the variable has another name).
- Cluster: `scripts/hpc_setup_xcist_env.sh` (EXP-011). The 1.6.9 sdist's `clib_build/BuildLinux64` has CRLF line
  endings (upstream defect); the script verifies the sdist sha256 against PyPI, fixes line endings in the build files
  only, rebuilds with `module gcc/11.2.0`. Jobs must `module load gcc/11.2.0` (`scripts/hpc_xcist_replicate.sh`).

## Verified conventions (tests/paired/test_xcist.py, independent analytic reference)
Raw phantom maps: float32 little-endian, x fastest = C-order `(iy, ix)`; **row 0 is +y** (y flipped), x grows with
column; source at (0, +SID) at view 0, angle grows counter-clockwise; detector arc centred on the source. Affine in
`xcist_affine`. 1-row (2D) scans only. HU = mu·1000/0.02 mm⁻¹ − 1000 (fixed nominal mu, not calibrated to the spectrum).

## Capabilities (from source, not all exercised)
- Energy-integrating detector (Lumex 3 mm) with prefilter; internal energy quadrature is 20 equal 6 keV bins treated
  monochromatically — **not** independent energy channels.
- A photon-counting model exists (`Detection_PC.py`, CZT response matrix of undocumented origin, no pile-up, no
  configurable charge sharing). **Not used or validated here.** Realistic PCCT detection is a deferred task (T-024).
- Scatter: empirical convolution only (`physics.scatterScaleFactor`); off in all our runs.
- Attenuation: Geant4 EDLP97 via `GetMu`; vs our xraylib tables 0.0–0.3% for water/Ca/I over 20–140 keV (30 keV iodine
  K-edge: 1.4%), and ~0.1–0.2% in projections (F-035).
- Seeding: noise-free path is bit-exact; with quantum noise same-seed runs differed in 1 of 21 600 values by ~0.2 sd
  (tests). C RNG exits the process on seed 0 — the adapter offsets seeds.

## Data
`ctsynth.paired.schema` (paired-ct/0.1) records: `record.json` + `.npy` arrays (outside Git), sha256-verified on load.
Dataset v0 (generated, not committed): `projects/p04_xcist/generate.py` → 3 phantoms × {80,120 kVp} × {ref, 2×-sampled,
noisy}; manifest hashes in `results/dataset_v0/MANIFEST.json` (RUN-0017).
