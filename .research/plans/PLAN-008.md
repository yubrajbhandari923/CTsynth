---
id: PLAN-008
title: P08 COPD XCAT noisy-CT simulator and inverse U-Net (staged simulate-then-train, node004 only)
status: active
question: null
skills: []
author_type: agent
author_name: claude-code
author_model: null
created_at: '2026-10-29T12:53:11-04:00'
updated_at: '2026-10-29T12:53:11-04:00'
completed_at: null
---

# PLAN-008 · P08 COPD XCAT noisy-CT simulator and inverse U-Net (staged simulate-then-train, node004 only)

## Objective

Reproducible, resource-efficient pipeline: XCAT -> LEAP projections -> Poisson noise / controlled acquisition and kernel variants -> compact CT cache -> 3D U-Net recovering organ-group labels; compare A clean, B noisy, C varied, D conditioned on identical patient splits.

## Success Criteria

Simulator integrated and tested; one-case feasibility measured; Stage-1 smoke passes (overfit + resume); full patient-level experiment only after the 65 missing COPD phantoms are staged (needs the user's go-ahead); every claim carries scope (2 patients vs 67).

## Context

User task 2026-10-29: one compute node at a time (node004: 128 CPU, 8 A5000), 48 h project budget, no commit of data. Branch feature/xcat-simulator (push blocked: no GitHub credentials on the laptop).

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (0/0 done)

_(no tasks yet)_

_Author: agent (claude-code)_
