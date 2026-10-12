# p03 unsupervised: EXP-004 B0-B4 ladder

Pre-registered experiment (Research OS EXP-004, binding). Question: which added prior, if any, selects
the true member of the 3-from-2 ambiguous family under label-free consistency training (F-014, F-018, F-019).

Files
- `run_exp004.py`: runner. `uv run python projects/p03_unsupervised/run_exp004.py configs/exp004_ladder.yaml`
- `figures.py`: reads only `results/metrics.json` and `results/val_predictions.npz`.
  `uv run --no-project --with matplotlib --with numpy python projects/p03_unsupervised/figures.py projects/p03_unsupervised/results`
- `../../configs/exp004_ladder.yaml`: all parameters (matched budget for every learned rung).
- `results/`: `metrics.json` (deterministic), `timing.json` (wall time, kept out of metrics), `run_meta.json`,
  `manifest.json`, `val_predictions.npz`, `fig_ladder.png`, `fig_maps.png`.

Scope and limits
- SYNTHETIC data only (`ctsynth.aapm.synthetic`, real CC-BY spectra/mu tables), image domain. No fan-beam projector exists.
- Reported metrics are on the VALIDATION split (13 cases; 41 train; the rule-assigned test cases are never loaded).
- Validation is also the best-epoch split. Label-free rungs select on validation consistency; B2 selects on validation
  label MSE (uses labels, disclosed).
- Classical rungs B0/B1 are deterministic (no seeds). Learned rungs use seeds 0, 1, 2.
- Unpre-registered choices, fixed and not tuned: prior weight 1.0, consistency weight 1.0, lam_tv 1e-4 (inherited from p02),
  roi 8 for s2, 32 px images, 80 epochs, lr 3e-3, batch 4, UNet depth 1 width 8.
- Mixed-voxel fraction: fraction of tissue-regime pixels (truth sum = 1) whose largest predicted fraction is below 0.9.
- Null-cos: mean over tissue pixels with nonzero error of |cos(pred - truth, F-014 null)|.

Tests: `uv run pytest tests/p03` (determinism on a tiny config, label-free isolation from truth, null-cos hand case,
null direction vs kit tables, split guard).
