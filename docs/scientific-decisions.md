# Scientific decisions, prior art, hypotheses, open questions

Owner: Opus. Decisions are superseded, never rewritten. Evidence levels: see `literature/README.md`.

> **Status 2026-10-13 (annotation, original text kept below).** This file is the 2026-10-11 scientific framing. The authoritative,
> current records are in Research OS: `research decision list` (D-001..D-011), `research question list`, `research finding list`.
> Where this file disagrees with those records, the records win. Stale items are marked inline with *[status: …]*; nothing was deleted.

## Prior art (what we can say)
- **Hellwege 2026** (full text): unsupervised UNet++ decoding DECT into materials/VMIs through a *differentiable polychromatic forward model* (MSE on measurements); simulated AAPM data; VMI RMSE 12.6/5.8/5.9 HU at 30/50/70 keV vs 13–49 HU for EDEC/AEDEC/MBIR. Authors admit an **inverse crime** (same forward model generates and decodes). Table-1 "mean" vs text "median" unresolved.
- **Peng 2024** (manuscript, equations unread): image-domain GAN, data-fidelity + adversarial noise prior, *linear two-basis model, no polychromatic physics*; noise SD −91–97% in simulation; iodine quantification close to reference.
- **Valand 2026** (abstract firsthand; body secondary): physics-informed GAN on DukeSim/XCAT photon-counting CT for density and Zeff; reported NRMSE 1.29%/0.68%. Numbers unverified against the PDF (summary showed internal inconsistencies).
- **Salinas 2025** (abstract only): 3D texture GAN on XCAT/DukeSim; relevant to realistic texture but whether textures carry material composition is **unknown**.
- **AAPM DL-Spectral CT** (metadata verified): 1000 training cases, 2D breast phantoms, fan-beam kV-switching 50/80 kVp, 3 tissue maps, 256×1024 sinograms. Spectra, geometry, noise model and metric code are **unverified** (in the unopened starting kit).

## Assumptions we are making (NOT verified)
- A1: a numpy/NIST-XCOM polychromatic model is accurate enough for method comparison (no scanner-calibrated claims).
- A2: AAPM-style 2D data is a sufficient first benchmark for P01 conclusions about 3D patients.
- A3: DukeSim/XCAT are not available to us; we build a simple procedural phantom generator instead. *[status: superseded by D-011: XCIST is the first independent forward-data producer; DukeSim deferred (`docs/dukesim-request.md`); phantoms are analytic disks, `paired/phantoms.py`]*

## Hypotheses (registered as Research OS questions)
| ID | Hypothesis | Falsified if |
|---|---|---|
| Q-002 H1 | Physics-consistency training beats image-domain baselines **when test forward model ≠ training forward model** | gain vanishes under mismatch (⇒ prior gains were inverse crime) |
| Q-003 H2 | Error is dominated by spectral separation and noise, not architecture | architecture ranking changes more than spectral-pair ranking |
| Q-004 H3 | Procedural-phantom-trained models generalise to unseen compositions | error on held-out compositions ≫ in-distribution |

## Decisions
- **D-A (2026-10-11)** Research OS is the experiment record; no custom tracker. See `architecture.md`. *[historical lettering; Research OS D-001..D-011 are the numbered decisions]*
- **D-B** Build the forward model *before* any learning; every learned method is compared to a classical decomposition baseline.
- **D-C** Evaluation must include a **forward-model-mismatch** test set (different spectra/NIST version/noise) — mitigates inverse crime.
- **D-D** No GPU/HPC claims from laptop runs (`hpc.md`).

## Non-goals
Clinical performance claims; reproducing any cited paper's numbers (not planned until a `reproduce-paper` task is approved); vendor-specific calibration; 3D anatomy generation.

## Open questions (for Yubraj / GPT Astra) *[status 2026-10-13: 1 answered (kit downloaded, T-004 done); 4 answered (Duke SLURM, `docs/hpc.md`); 5 answered (XCIST available, D-011); 2-3 remain open]*
1. Permission to download the AAPM starting kit (103 MB) for spectra/geometry? (T-004)
2. Is photon-counting (Valand-style) or kV-switching DECT the primary target for P01?
3. Material basis: 2 basis materials, 3 (AAPM), or elemental ρ/Zeff? Affects the `MaterialMap` contract.
4. HPC details (scheduler, GPU, env management) for `backends.slurm.setup`.
5. Any access to XCAT/DukeSim, or procedural phantoms only?
