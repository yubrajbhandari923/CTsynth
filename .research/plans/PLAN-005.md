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
updated_at: '2026-10-13T20:37:53-04:00'
completed_at: null
---

# PLAN-005 · P05 AAPM end-to-end physics pipeline and benchmark suite

## Objective

Milestone A: verified differentiable fan-beam AAPM pipeline and label-free demo. Milestone B: reproducible AAPM benchmark (classical, supervised, physics-only, priors, hybrid) on the full 1000-case dataset

## Success Criteria

Exp1 oracle forward agrees with kit transmissions within pre-derived tolerance; gradients verified; frozen split and held-out evaluation; findings independently reviewed

## Context

Supersedes execution order of PLAN-002 open tasks T-017/T-020/T-021/T-022 (re-parenting unsupported). XCIST/PCCT/XCAT out of scope. Projector: torch port of the official kit layered projector (src/ctsynth/aapm/fanbeam.py), reproduces kit sinograms to float32 rounding (dev check, not a registered result). NOTE adapter-spec s4 says exact chord; the kit actually uses layered linear splitting.

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (0/0 done)

_(no tasks yet)_

_Author: agent (claude-code)_
