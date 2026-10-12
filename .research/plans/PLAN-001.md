---
id: PLAN-001
title: 'P01 foundation: spectral forward model and baselines'
status: active
question: Q-001
skills: []
author_type: agent
author_name: opus
author_model: null
created_at: '2026-10-11T19:17:41-04:00'
updated_at: '2026-10-11T19:17:41-04:00'
completed_at: null
---

# PLAN-001 · P01 foundation: spectral forward model and baselines

## Objective

Validated spectral forward model + classical baseline decomposition on laptop, then HPC-ready configs

## Success Criteria

Forward model reproduces analytic Beer-Lambert cases; baseline recovers noise-free materials

## Context

<!-- research:generated — everything below is regenerated; edits here are ignored -->

## Tasks (4/9 done)

- ✓ **T-001** Spec: spectral forward model interface + physics tests — done
- ✓ **T-002** Implement spectral forward model (numpy, CPU) — done (deps: T-001)
- ✓ **T-003** Vendor NIST XCOM mass-attenuation tables with provenance — done
- ⛔ **T-004** Read AAPM starting_kit.tgz (103 MB) for spectra/geometry/metric (needs user OK to download) — blocked
- ○ **T-005** Classical baseline: image-domain two-material decomposition on toy phantom — todo (deps: T-002, T-003)
- ○ **T-006** Attenuation table v2 with K-edge-resolved energy grid — todo
- ✓ **T-007** EXP-002: identifiability toy experiment (designs x basis size x mismatch) — done
- ○ **T-008** EXP-003: spectral-mismatch power study (resolve F-004) — todo
- ○ **T-009** Inverse-crime-free evaluation harness (H1/Q-002 prerequisite) — todo

⚠ 1 task(s) blocked

_Author: agent (opus)_
