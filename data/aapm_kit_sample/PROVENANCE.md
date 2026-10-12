# AAPM DL-Spectral CT — forward-model tables (committed)

Source: Zenodo record **10.5281/zenodo.14262737**, file `starting_kit.tgz`
(md5 `7aaf6b55708a6576ebfa91e43ec80082`, verified on download 2026-10-11), path `starting_kit/model/`.
License: **CC-BY-4.0** — Sidky & Pan, "Data repository for the 2022 AAPM deep-learning spectral CT grand challenge".
Report: Sidky & Pan, Med Phys 51(2):772-785, arXiv:2212.06718.

Only these two small tables are committed (5 KB); **no case data is in git** (`.gitignore`: `data/aapm/`, `*.npy.gz`).

| File | sha256[:32] | Shape | Contents |
|---|---|---|---|
| `model_data_50kVp.npy` | `747e0ab32e42165b7279ee49579b08d0` | (5, 98) | row0 energies 1.25–49.75 keV (0.5 keV); row1 spectrum s (Σ0.5·s=1); rows2-4 μ adipose/fibroglandular/calcification |
| `model_data_80kVp.npy` | `36627ce32822bbda83767453afc9470a` | (5, 158) | same, energies 1.25–79.75 keV |

Forward model (from `starting_kit/model/`, read not executed):
`T = Σ_E 0.5·s_kVp(E)·exp(−μ_a(E)·P_a − μ_f(E)·P_f − μ_c(E)·P_c)`, I₀ = 1, **noiseless**.
Geometry: circular fan beam, flat 1024-bin detector, source radius 50 cm, source-to-detector 100 cm,
512×512 image over 18×18 cm (0.03516 cm pixels), 512 views counter-clockwise, **even = 80 kVp, odd = 50 kVp**
(256 each, offset 2π/512).

UNVERIFIED: the kit never states whether maps are volume fractions or mass densities; volume fraction with
μ in cm⁻¹ is inferred (maxima 1.0, maps sum to 0 or 1, sinogram = map × path length in cm). μ source and
reference densities are not given.
