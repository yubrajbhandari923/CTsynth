---
title: Phase 2 architecture and workflow audit (2026-10-13)
pinned: true
links: [PLAN-004]
created: '2026-10-13T00:26:28-04:00'
---

# Phase 2 architecture and workflow audit (2026-10-13)

Full report: docs/architecture-workflow-audit.md. Plan: PLAN-004 (T-043..T-060). Verified headline findings: (A1) three different 'source clean' definitions - runmeta checks src/tests/configs only, preflight also scripts/projects/pyproject/uv.lock, so a dirty projects/ tree is recorded git_dirty_source=false; (A2) orchestration.enabled is false so sonnet_first routes but nothing is contracted/verified, no hooks, no usage recorded; (A3) experiment drivers in projects/p04 etc. untested; (A4) hard-coded /home/yb107 paths in two HPC scripts; (A5) 20 exp002 result files tracked in git against the documented rule (keep, make explicit exception); (A6) AAPM image[ix,iy] vs paired (iz,iy,ix) with no converter; PLAN-002 holds all XCIST E1-E4 tasks; D-009 still active though its CLI was removed; docs/scientific-decisions.md stale. NOT duplication (keep both): physics.SpectralForwardModel vs aapm.AapmTorchForward; exp002 estimator vs aapm.classical. Preserve unchanged: physics/*, frozen tables, aapm APIs, paired/schema, native HPC protocol, hash-frozen hpc_smoke/env_probe. Decisions needed from human: enable orchestration, approve hooks, seed_everything, doc consolidation of Opus-owned file, plan re-parenting, T-042. No engineering finding here is a scientific finding.
