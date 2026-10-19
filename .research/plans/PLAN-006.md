---
id: PLAN-006
title: 'P06 Null-direction information: what makes AAPM adipose/fibroglandular decomposition identifiable'
status: active
question: Q-006
skills: []
author_type: agent
author_name: claude-code
author_model: sonnet
created_at: '2026-10-15T13:41:37-04:00'
updated_at: '2026-10-15T13:41:37-04:00'
completed_at: null
---

# PLAN-006 · P06 Null-direction information: what makes AAPM adipose/fibroglandular decomposition identifiable

## Objective

Determine what additional information (calcification/sum knowledge, bimodality, spatial or distributional priors) makes the 3-from-2 decomposition identifiable or reliably recoverable label-free, separating information gain from optimisation effects.

## Success Criteria

A prior (or structural constraint) is supported only if, versus the EXP-023 baseline and a wrong-prior control, it lowers 3-seed mean s1 by more than max(2x pooled range, 0.05) with >=2/3 seeds below 0.15, AND the converged per-case solve (information bound) shows the same ordering; every claim carries synthetic/validation scope and independent review.

## Context

Motivation (reviewed numbers, preliminary): F-041/RUN-0091: the F-012 null direction is the A<->F swap compensated by ~0.8% calcification per unit; the swap with calcification fixed is 330x more visible (1.3e-2 vs 4e-5 RMSE per unit). F-043/EXP-034 (corrected): with calcification absent and sum known, even SECT identifies the adipose fraction at ~0.03 per ray; spectral bins add nothing; PCCT gains ~2x only if calcification stays unknown. EXP-025 priors (entropy, TV, range hinge, calc L1) were screened with weights set to 10-100% of the projection loss at the collapsed solution and were inert; F-039: sum pinning never actually tested (EXP-024 penalty left |S-1| unchanged). Stage gating: do not start Stage 2 unless Stage 0 and Stage 1 pass. EXP-026 (domain comparison) remains deferred by the owner. No 10-seed replication of occ_softmax.

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (0/0 done)

_(no tasks yet)_

_Author: agent (claude-code)_
