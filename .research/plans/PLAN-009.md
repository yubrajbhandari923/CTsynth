---
id: PLAN-009
title: 'P09 Physical-phantom inversion: composition-concentration targets, baselines, U-Net, closed-loop
  LEAP and 70->80 keV test'
status: active
question: Q-009
skills: []
author_type: agent
author_name: claude-code
author_model: sonnet-5.5
created_at: '2026-10-29T22:01:54-04:00'
updated_at: '2026-10-29T22:01:54-04:00'
completed_at: null
---

# PLAN-009 · P09 Physical-phantom inversion: composition-concentration targets, baselines, U-Net, closed-loop LEAP and 70->80 keV test

## Objective

Predict XCAT-derived per-voxel composition mass concentrations (g/cm3) from cached simulated COPD CT and validate by passing the predicted phantom through LEAP at the input and a different energy.

## Success Criteria

Targets validated against xlsx tables and simulator mu; baselines frozen; networks trained once on existing cache; test evaluated once; closed-loop LEAP and 80 keV transfer reported with identifiability limits.

## Context

User task 2026-10-29. Reuse EXP-039 cache (cache_v1), frozen split, node004 only (one node cap), no CT regeneration except necessary held-out evaluation sims. Do not overwrite EXP-039/F-049. Branch feature/xcat-simulator.

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (0/0 done)

_(no tasks yet)_

_Author: agent (claude-code)_
