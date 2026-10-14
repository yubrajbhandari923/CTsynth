---
id: PLAN-005
title: P05 AAPM end-to-end physics pipeline and benchmark suite
status: active
question: Q-006
skills: []
author_type: agent
author_name: claude-code
author_model: null
created_at: '2026-10-13T20:37:53-04:00'
updated_at: '2026-10-13T21:00:35-04:00'
completed_at: null
---

# PLAN-005 · P05 AAPM end-to-end physics pipeline and benchmark suite

## Objective

Milestone A: verified differentiable fan-beam AAPM pipeline and label-free demo. Milestone B: reproducible AAPM benchmark (classical, supervised, physics-only, priors, hybrid) on the full 1000-case dataset

## Success Criteria

Exp1 oracle forward agrees with kit transmissions within pre-derived tolerance; gradients verified; frozen split and held-out evaluation; findings independently reviewed

## Context

Supersedes execution order of PLAN-002 open tasks T-017/T-020/T-021/T-022 (re-parenting unsupported). XCIST/PCCT/XCAT out of scope. Projector: torch port of the official kit projector (src/ctsynth/aapm/fanbeam.py); reproduces kit sinograms to float32 rounding. CORRECTION (2026-10-13): an earlier version of this context and adapter-spec s4 said the kit projector is NOT exact-chord. That was wrong: its layerwise split is proportional to ray extent, i.e. exact chord length; local reduced-geometry check shows agreement with the dense exact-chord reference to 4e-14 cm. EXP-017 reports this comparison formally.

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (1/10 done)

- ✓ **T-061** Fan-beam projector (kit port) + tests — done
- ○ **T-062** Polychromatic AAPM forward: fractions -> kVp-switched transmissions — todo (deps: T-061)
- ○ **T-063** Register tolerance derivation for EXP-017 (numerical analysis) — todo
- ○ **T-064** Run EXP-017 oracle forward on HPC — todo (deps: T-062, T-063, T-017)
- ○ **T-065** Run EXP-018 gradient verification — todo (deps: T-062)
- ○ **T-066** U-Net physics-loss training driver (supervised / physics-only / feasibility arms) — todo (deps: T-062)
- ○ **T-067** Run EXP-019 smoke test, EXP-020 demo; Milestone A checkpoint — todo (deps: T-064, T-065, T-066)
- ⛔ **T-068** T-017 acquire + validate full dataset on HPC (existing task) — blocked
- ○ **T-069** Freeze 800/100/100 split with leakage checks — todo (deps: T-017)
- ○ **T-070** Experiments 5-12 (Milestone B) in dependency order — todo (deps: T-067, T-069)

⚠ 1 task(s) blocked

_Author: agent (claude-code)_
