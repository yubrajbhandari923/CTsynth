---
id: PLAN-002
title: 'P02 AAPM benchmark: adapter, classical and supervised baselines'
status: active
question: Q-005
skills: []
author_type: agent
author_name: opus
author_model: null
created_at: '2026-10-11T20:01:57-04:00'
updated_at: '2026-10-11T20:01:57-04:00'
completed_at: null
---

# PLAN-002 · P02 AAPM benchmark: adapter, classical and supervised baselines

## Objective

Reproducible benchmark on AAPM DL-Spectral CT: adapter with explicit conventions, deterministic splits, streaming loader, constrained classical baseline, small U-Net, honest evaluation

## Success Criteria

Synthetic smoke benchmark runs end-to-end locally; adapter verified against real kit metadata; HPC configs prepared; no dataset-dependent claim without processed files

## Context

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (7/20 done)

- ✓ **T-010** AAPM adapter + synthetic schema-mirror generator + deterministic splits — done
- ✓ **T-011** Classical constrained decomposition baseline (NNLS + simplex constraint, TV variant) — done (deps: T-010)
- ✓ **T-012** Small supervised U-Net (2 CT channels -> 3 material maps) — done (deps: T-010)
- ✓ **T-013** Evaluation protocol + metrics/figures/manifests — done (deps: T-010)
- ✓ **T-014** Full-data HPC acquisition and experiment plan — done
- ○ **T-017** P1.1 Acquire + validate the real AAPM 1000-case dataset — todo
- ○ **T-018** P1.5 Occupancy-aware constraint: oracle vs estimated — todo
- ○ **T-019** P1.6 Prior ablations with justified weights and a power analysis — todo
- ○ **T-020** P1.7 Independent forward-model validation / inverse-crime avoidance — todo
- ○ **T-021** P1.2 Fan-beam interlaced projector + projection-domain forward model — todo (deps: T-017)
- ○ **T-022** P1.3 Classical vs supervised vs self-supervised on REAL AAPM cases — todo (deps: T-017)
- ○ **T-023** P1.4 Full-tomography identifiability beyond single-ray decoys — todo (deps: T-021)
- ○ **T-024** P2.1 PCCT detector realism: energy-bin response, charge sharing, pile-up, calibration — todo
- ○ **T-025** P2.2 DukeSim paired material-to-CT dataset contract and import — todo
- ○ **T-026** P2.3 3D multi-organ material inference — todo
- ○ **T-027** P2.4 Anatomy-to-material transfer (XCAT / AbdomenGen) — todo
- ○ **T-028** P2.5 Efficient simulation and surrogate residual physics — todo
- ○ **T-029** P2.6 Disease interventions and downstream virtual-trial validation — todo
- ✓ **T-030** P1.8 HPC bundle validator + environment pinning (Prompt 4 prerequisite) — done
- ✓ **T-031** P1.9 Laptop<->HPC bundle workflow CLI — done (deps: T-030)

_Author: agent (opus)_
