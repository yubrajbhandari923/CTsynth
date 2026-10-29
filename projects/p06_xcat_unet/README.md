# p06: COPD XCAT noisy-CT simulator and inverse U-Net (PLAN-008)

Code lives in `src/ctsynth/xcat` (simulation) and `src/ctsynth/seg` (U-Net, loss, dataset, inference); this folder holds scripts, configs and reports. Generated data and checkpoints stay on DRCC under `/home/yb107/datasets/truect_pilot/outputs/xcat_unet/` (never committed).

**Workflow (staged, LEAP never runs during training).** (1) `generate_variants.py` simulates controlled CT variants per patient and writes the compact cache (`npy_i16q` + shared `labels.npy` + `class_samples.npz`, JSON metadata per variant) on node-local NVMe, one process per GPU; (2) persist to `/home`; (3) simulator processes exit (GPU memory returns to 2 MiB); (4) `train_unet.py` in a fresh process reads the cache (resumable, validation-based model selection); (5) `evaluate_unet.py` evaluates every variant incl. held-out acquisition settings, with the intensity-only baseline.

**All cluster jobs go through `scripts/sub.sh`**: node004 only, at most one node of resources in total (8 GPUs / 128 CPUs) including already queued jobs. DRCC environment: `~/ctsynth_xcat/venv` (Python 3.11, torch 2.7.0+cu126) with `PYTHONPATH=~/ctsynth_xcat/src`; LEAP 0c8846f via `LEAP_SRC`.

```bash
rsync -a --delete --exclude __pycache__ src/ctsynth/ plp-drcc-login2:ctsynth_xcat/src/ctsynth/ ; rsync -a projects/p06_xcat_unet/scripts/ plp-drcc-login2:ctsynth_xcat/scripts/
ssh plp-drcc-login2 'bash -lc "cd ~/ctsynth_xcat/scripts && ./sub.sh feas 1 16 96G 01:30:00 feasibility_job.sh 22728 0.8"'   # one-case benchmark
# after the transfer is approved and run (reports/transfer_missing_copd.sh --sbatch), per GPU i of 8:
CUDA_VISIBLE_DEVICES=$i python generate_variants.py --root DATA --patients <ids slice i> --variants clean family:B family:C family:D --cache /tmp/yb107_cache
python train_unet.py --arm C --cache CACHE --train-patients $(jq -r '.train[]' configs/split_seed0.json) --val-patients ... --out RUN
```

Reports: `reports/data_inventory.md` (+ manifest, transfer script), `reports/feasibility_report.md` (Gate B/C), `reports/variation_report.md` (+ figures), `EXPERIMENT_PLAN.md` (pre-registration draft for arms A-D). Status: Gates A-C passed on two patients; Gate D (65 missing phantoms, 181 GB) awaits approval; arms A-D not run.
