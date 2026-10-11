# EXP-002 — Spectral identifiability toy

Pre-registered before running (`research show EXP-002`); run as **RUN-0002** on clean commit cf3b880, 71 s CPU.

## Reproduce
```bash
.research/bin/research run exec EXP-002 -- uv run python projects/p01_material_inference/exp002/run_exp002.py
uv run --no-project --with matplotlib --with numpy python projects/p01_material_inference/exp002/figures.py
```
Byte-identical `metrics.json` on re-run. Tests: `uv run pytest tests/exp002`.

## What this experiment does
2D 48×48 phantom, parallel beam (90 views × 72 detectors), materials = **partial mass densities** (g/cm³).
Designs at **equal total unattenuated photons per ray**: M1 (one 120 kVp energy-integrating bin),
M2 (80+120 kVp), M4 (four photon-counting windows on one 120 kVp exposure); basis size k ∈ {2,3};
N_tot ∈ {1e5, 1e6}; matched + four spectral-mismatch conditions. Estimator: per-ray Poisson MLE (LM),
always using the **nominal** spectra.

Statistics (Fisher/CRLB/rank/empirical std/bias) are on a **seeded ray subset** (48 rays P2, 60 P3;
20 Poisson realisations each) — *not* the full image. 16 full-image reconstructions use 1 realisation each.

## Results after independent review
The original findings F-001…F-004 were challenged by an independent reviewer; three were superseded.
**Current: F-005, F-006, F-007, F-008** (`research show F-007` etc.). Read those, not the originals.

- **F-007** Rank ≤ min(n_meas, k) is *arithmetic, not evidence*. The empirical content is conditioning among
  identifiable designs (dual-kVp k=2: median cond 195, max 506; 4-bin k=2: 99; 4-bin k=3: 217) and noise at
  equal basis size, where **photon counting is 2.5× better** than dual-kVp (iodine CRLB 0.0019 vs 0.0047 g/cm²).
- **F-005** Spectral mismatch **always** biases a nominal-spectrum estimator. A noise-free calculation gives
  nonzero bias in every condition; visibility depends on material, dose and statistic.
- **F-006** Empirical std matches CRLB to 12.5% — a *self-consistency* check of Jacobian+sampler+MLE,
  **not** validation of physical fidelity (a wrong table would pass equally).
- **F-008** Basis misspecification bias is systematic and dose-independent (iodine *line integral*, g/cm²).

## Limits (do not cite beyond these)
Shared forward model for data and estimator (**no inverse-crime avoidance**: best case, overstates accuracy).
D-002: iodine K-edge under-resolved → **no absolute iodine-concentration claim**. Equal-photon scaling excludes
pure gain mismatch by construction. Idealised energy-integrating Poisson model. No scatter, detector response,
charge sharing or pile-up — so the photon-counting advantage is an upper bound. 2D, laptop CPU/MPS only;
nothing here validates GPU or HPC. Headline recovery numbers (iodine RMSE 0.005 g/cm³ against a 0.005 g/cm³
insert; calcium 0.085 against 0.1–0.3) are **not** evidence of good recovery.
