# P01 — Material inference from dual-energy / spectral CT

Question Q-001 and hypotheses H1–H3 (Q-002..Q-004) live in Research OS; rationale in `docs/scientific-decisions.md`.
Plan: PLAN-001 (`research plan show PLAN-001`).

| Task | Role | Depends | Status |
|---|---|---|---|
| T-001 Forward-model spec + frozen physics tests | Sonnet | — | todo |
| T-002 Implement forward model | Haiku | T-001 | todo |
| T-003 Vendor NIST XCOM tables w/ provenance | Haiku | — | todo |
| T-004 Read AAPM starting kit (needs user OK) | Opus | — | blocked on approval |
| T-005 Classical 2-material baseline, first experiment | Haiku | T-002, T-003 | todo |

Live status is authoritative in Research OS, not this table.

## Next task proposal: spectral forward model
See T-001/T-002. Scope: `src/ctsynth/physics/{spectrum,attenuation,forward}.py`, numpy, float64.
Frozen tests (Sonnet): (1) monochromatic spectrum ⇒ `p = Σ ρ_m (μ/ρ)_m(E) t_m` exactly; (2) zero thickness ⇒ p=0;
(3) p linear in thickness for monochromatic, strictly sub-linear for polychromatic (beam hardening); (4) normalisation: scaling S leaves p unchanged;
(5) water cylinder analytic line integral; (6) Poisson noise mean/variance at stated counts. Tolerances stated with rationale; no loosening to pass.
Inputs: material map + spectrum + parallel-beam geometry first (no projector library yet). Output: line integrals `(n_spectrum, view, col)`.
