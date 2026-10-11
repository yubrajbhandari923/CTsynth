# EXP-002 outputs (spectral identifiability toy)

Run: `uv run python projects/p01_material_inference/exp002/run_exp002.py` (config `configs/exp002.yaml`);
figures: `uv run --no-project --with matplotlib --with numpy python projects/p01_material_inference/exp002/figures.py`.

**Limitation (D-002):** the iodine K-edge (33.17 keV) is under-resolved on the 1 keV table; data
generation and estimation share that table. No absolute iodine-concentration accuracy claim is made;
only identifiability, conditioning and noise claims.

Scope: Fisher/CRLB/rank/empirical std/bias are computed on a fixed ray SUBSET (see `metrics.json`
`scope`), 20 realisations each. Full-image (90x72 rays, 1 realisation) reconstructions exist only for the
`full_image` records. Materials are partial mass densities (g/cm^3). Scaling and weighting conventions
are in `metrics.json` (`scaling`, `weighting`). `results/` is regenerable; `timing.json` is kept out of
`metrics.json` so the latter is bit-reproducible.
