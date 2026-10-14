# Milestone A demo: AAPM images -> U-Net -> materials -> fan-beam forward -> transmission loss

Records live in Research OS (EXP-017, 018, 019, 020; runs RUN-0031, 0029, 0033, 0034-0036). This page is only the how-to.
Interfaces: `ctsynth.aapm.fanbeam.fanbeam_project` (differentiable official projector), `ctsynth.aapm.forward.AapmFanbeamForward`
(fractions -> {50,80} kVp transmissions, even views = 80 kVp), `ctsynth.aapm.physics_loss.projection_loss`.
Conventions: `adapter-spec.md` (the official kit projector is the exact ray-in-pixel chord, computed layerwise).

## Reproduce (Duke; needs `research hpc doctor` green, dataset adopted, split `configs/aapm_split_v1.json`)
```bash
git commit -a                                   # runs need a clean source tree
uv run ctsynth preflight --repo . && uv run ctsynth bundle-check
D=/scratch/railabs/yb107/research-os/shared-data/aapm-dl-spectral-ct-zenodo-14262737
for arm in supervised physics physics_feas; do
  research hpc submit EXP-020 --cluster duke --gpus 1 --cpus 8 --mem 48G --time 04:00:00 --dry-run -- \
    bash scripts/hpc_run_py.sh projects/p05_aapm/milestone_a.py --data $D --split configs/aapm_split_v1.json \
    --arm $arm --n-train 32 --n-val 8 --steps 3000 --batch 2 --views 32 --val-every 100   # then re-run with --yes
done
research hpc sync                               # imports outputs; figures.png/curves.png/results.json land in .research/runs/RUN-*/collected/outputs/
```
Each run saves inputs, truth, predicted maps, error maps, reference/predicted transmissions and residuals (`figures.png`),
loss/grad-norm/validation curves (`curves.png`), `results.json`, `history.json`, `val_pred.npz`, `checkpoint.pt`.

## Reading the result
Gradient flow (finite grads, loss falls) is not recovery. Judge recovery by `s1`, per-material RMSE and `null_dir_error_frac`
in `results.json`, never by projection error alone. Arms B/C (`physics`, `physics_feas`) never see truth in loss, selection or targets.
