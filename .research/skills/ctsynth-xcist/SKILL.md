---
name: ctsynth-xcist
description: Use when a task touches XCIST/gecatsim: simulating phantoms, reconstructing, the .venv-xcist env, the Duke XCIST env, or paired-ct records from XCIST.
---

# ctsynth-xcist

XCIST is an optional forward-data producer, never a runtime dependency (D-001, D-011). Details and verified
conventions: `docs/xcist.md`. Code: `ctsynth.paired.xcist`, `ctsynth.paired.phantoms`; drivers: `projects/p04_xcist/`.

**Inputs/assumptions**
- Laptop: `.venv-xcist/bin/python` (gecatsim 1.6.9, numpy<2). The core `uv` env has no gecatsim; anything importing it lazily
  must not break `import ctsynth`.
- macOS FDK recon needs `XCIST_LIBGOMP_DIR`; without it recon is unavailable (simulation still works).
- Cluster: env built by `scripts/hpc_setup_xcist_env.sh`; location from `CTSYNTH_XCIST_ENV` (default `$HOME/research-os/projects/CTsynth/env`).
- Truth is always the input phantom (volume fractions), never XCIST output.

**Steps**
1. `.venv-xcist/bin/python -m ctsynth.hpc.cli xcist-check` (seconds): `ok_simulate` / `ok_recon` and the reason if not.
2. Geometry/convention changes: run `.venv-xcist/bin/python -m pytest tests/paired/test_xcist.py` (≈1 min; the recon test skips without libgomp, say so).
3. Dataset records: `ctsynth paired-check <record dirs>` verifies every sha256 (use before and after transferring).
4. Cluster jobs follow the `ctsynth-experiment` skill; confirm the reservation with the human.

**Verification**: orientation and calibration stay pinned by the analytic reference in `tests/paired/test_xcist.py`
(row 0 = +y, source at +y at view 0, counter-clockwise). Do not change a convention to make a comparison agree.

**Record**: runs through `research run exec` / `hpc submit`; register `mismatch.json`-style outputs as artifacts; XCIST version and
library hashes come from `run_xcist` provenance. Spectrum specification is the largest known CTsynth-vs-XCIST gap (F-037, preliminary).
