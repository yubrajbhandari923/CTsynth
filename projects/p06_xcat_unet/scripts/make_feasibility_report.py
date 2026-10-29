#!/usr/bin/env python
"""Generate feasibility_report.md from measured JSON (sim_report, unet_probe, dataset_cost_estimate). usage: make_feasibility_report.py DIR OUT.md"""

import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
S = json.load(open(d / "sim_report_p22728_0.8mm.json"))
U = json.load(open(d / "unet_probe_p22728_0.8mm.json"))
E = json.load(open(d / "dataset_cost_estimate.json"))
st = S["steps"]
MIB = 1.0
L = []
w = L.append
w("# Feasibility report (Gate B / C): one COPD phantom, one acquisition, node004\n")
w(
    "Generated from measured files (`sim_report_p22728_0.8mm.json`, `unet_probe_p22728_0.8mm.json`, job 203524) plus the node probe (job 203521). **Measured** unless marked ESTIMATE.\n"
)
w("## 1. What the node offers (probed on node004, job 203521)\n")
w("| tier | size / free | write | read | notes |\n|---|---|---|---|---|")
w(
    "| GPU VRAM (8 x RTX A5000) | 24138 MiB usable each | - | - | one job used 1 GPU; process exit returns it to 2 MiB |"
)
w("| System RAM | 503 GiB | - | - | Slurm only reserves what `--mem` asks for |")
w(
    "| `/dev/shm` (RAM-backed tmpfs) | 252 GB | 2.5 GB/s | 6.8 GB/s (warm) | counts against RAM; lost at job end |"
)
w(
    "| `/tmp` and `/local` (node-local NVMe, same device) | 1.8 TB, 1.5 TB free (shared) | 1.1 GB/s | 2.8 GB/s uncached, 13.6 GB/s page-cache | no per-job directory (`SLURM_TMPDIR` unset, Slurm `TmpDisk=0`); not cleaned automatically as far as I can see |"
)
w(
    "| `/cachedata` (second local NVMe) | 7 TB, 3.1 TB free | not tested | not tested | root-owned: I cannot create a directory (other users have admin-made folders) |"
)
w(
    "| `/home` (NFS, shared lab) | 6.3 TB free | 0.69 GB/s | 0.27 GB/s uncached, 12 GB/s cached | durable; no quota tool available |"
)
w("| `/scratch` (NFS) | 1.8 TB free of 74 TB (98% used) | - | - | not used |")
w(
    "\nScheduler: partition `vram24` (node002, node004) has no wall-clock limit set; the project rule of < 10 h per job is kept anyway. `ssh tara` works from compute nodes (not from the laptop).\n"
)
w("## 2. One-case measurements (copd_22728, 0.8 mm, 11000 views, 7 repetitions)\n")
w(
    f"Grid (z,y,x) = {tuple(S['grid_zyx'])}; projection tensor {S['proj_MiB']:.0f} MiB; attenuation volume {S['vol_MiB']:.0f} MiB; native phantom {S['bin_bytes'] / 1e9:.2f} GB.\n"
)
w("| step | median [s] | min-max [s] | peak GPU [MiB] |\n|---|---|---|---|")
names = {
    "01_read_phantom_file_into_ram": "1 read phantom (page-cache warm; cold NFS ESTIMATE ~6 s at 0.27 GB/s)",
    "02_material_table_to_mu": "2 material table -> mu (xlsx + element table; first call 0.33 s imports)",
    "03_native_resolution_mu_conversion": "3 native 0.25 mm mu conversion (GPU LUT, reduced)",
    "04_resample_to_grid": "4 resample to 0.8 mm grid",
    "05_leap_init_and_geometry": "5 LEAP init + geometry (first call 0.28 s)",
    "06_forward_project": "6 forward projection (clean, 11000 views)",
    "07_poisson_noise": "7 Poisson noise (in the count domain)",
    "08a_helical_fbp_noisy": "8a helical FBP (noisy)",
    "08b_helical_fbp_clean": "8b helical FBP (clean)",
    "09a_mu_to_hu_on_gpu": "9a mu -> HU (GPU)",
    "09b_device_to_host_copy": "9b device -> host copy",
}
for k, v in st.items():
    pk = f"{v['peak_device_mib']:.0f}" if v["peak_device_mib"] else "-"
    w(f"| {names[k]} | {v['median_s']:.4f} | {v['min_s']:.4f}-{v['max_s']:.4f} | {pk} |")
w(
    f"\nCache write of the training CT (int16, 0.25 HU step): {S['train_cache_write_s']:.2f} s, {S['train_cache_bytes']['ct_i16'] / 1e6:.0f} MB; labels {S['train_cache_bytes']['labels_u8'] / 1e6:.0f} MB. Host RSS peak of the simulator process: {S['host_rss_peak_MiB_final']:.0f} MiB; training-probe process {U['host_rss_peak_MiB']:.0f} MiB."
)
w(
    f"\n**GPU release.** Same process after `del` + `empty_cache`: {S['gpu_used_MiB_after_del_and_empty_cache_same_process']:.0f} MiB still in use (started at {S['gpu_used_MiB_at_process_start']:.0f}, peak {S['gpu_used_MiB_before_release']:.0f}); PyTorch reports {S['torch_allocated_MiB_after_release']:.0f} MiB allocated. After the simulator process EXITED the job log shows 2 MiB used. `empty_cache` alone does not return LEAP's/CUDA's allocations; process separation does.\n"
)
w(
    "## 3. Cache formats (same CT volume, 3 write repetitions; read-patch = 60 random 96^3 patches, cold = page cache dropped for npy)\n"
)
w(
    "| tier | format | MB | write MB/s | patch96 warm ms | patch96 cold ms | full read s | max error HU |\n|---|---|---|---|---|---|---|---|"
)
for c in S["cache"]:
    if c["tier"] == "shm" and c["fmt"] in ("npy_f16", "zarr_zstd1_i16q"):
        pass
    cold = f"{c['patch96_cold_ms_median']:.2f}" if c["patch96_cold_ms_median"] else "n/a"
    w(
        f"| {c['tier']} | {c['fmt']} | {c['bytes'] / 1e6:.0f} | {c['write_MB_s']:.0f} | {c['patch96_warm_ms_median']:.2f} | {cold} | {c['full_read_s_median']:.3f} | {c['max_abs_hu_err']:.3f} |"
    )
w(
    f"\nCompressed NIfTI of the same volume: {S['nifti_gz_write_s']:.1f} s and {S['nifti_gz_bytes'] / 1e6:.0f} MB (vs {S['cache'][2]['write_s_median']:.2f} s and {S['cache'][2]['bytes'] / 1e6:.0f} MB for `npy_i16q` on shm): about {S['nifti_gz_write_s'] / S['cache'][2]['write_s_median']:.0f}x slower and 1.8x larger."
)
w(
    "\n**Choice: `npy_i16q`** (int16 = round(HU / 0.25), clipped at +-8191 HU): half of float32, max quantisation error 0.125 HU (SD 0.07 HU, far below the 9-54 HU noise), fastest sequential and random-patch reads, memory-mappable by every dataloader worker. float16 is slower to decode and has 0.5 HU error; Zarr (chunk 64^3) is 2-4x slower for patches on local disks and 10-30x on NFS, and compression saves only 35-50% of a volume that is cheap to store.\n"
)
w(
    "## 4. U-Net probe in a fresh process (no LEAP loaded; 1.4 M parameters, base 16, depth 3, bf16)\n"
)
w(
    f"Time to open the cache, read the whole volume once and prepare class samples: {U['time_to_open_cache_and_prepare_s']:.2f} s.\n"
)
w(
    "| patch | batch | step [s] | peak allocated [MiB] | Mvoxel/s | patch load ms/item | GPU idle with 4 workers |\n|---|---|---|---|---|---|---|"
)
for c in U["configs"]:
    idle4 = [x for x in c["loaders"] if x["workers"] == 4][0]["gpu_idle_fraction"]
    w(
        f"| {c['patch']} | {c['batch']} | {c['step_median_s']:.3f} | {c['peak_allocated_MiB']:.0f} | {c['voxels_per_s'] / 1e6:.2f} | {c['patch_load_ms_per_item']:.1f} | {100 * idle4:.1f}% |"
    )
w(
    "\nTraining is compute-bound at about 9 Mvoxel/s for every patch/batch size; with 4 or more loader workers the GPU waits for data only about 1% of the time, so I/O from the cache does not limit training. 3D patches of 96^3 with batch 4 fit in 4.5 GB, so eight independent trainings fit on the 8 GPUs.\n"
)
w(
    "## 5. Dataset cost for all 67 patients (ESTIMATE: measured one-case times scaled per patient by views x sqrt(voxels); see `dataset_cost_estimate.json`)\n"
)
w(
    "| variants / patient | GPU-h: noise variants (one shared projection) | GPU-h: every variant needs its own projection | wall-h on 8 GPUs (ideal) | CT cache int16 | if stored as float32 | if written as .nii.gz | full-coverage epoch of 47 train patients [h] |\n|---|---|---|---|---|---|---|---|"
)
for v, s in E["scenarios"].items():
    w(
        f"| {v} | {s['gpu_hours_noise_only_shared_projection']:.2f} | {s['gpu_hours_each_variant_own_projection']:.2f} | {s['wall_hours_8_gpus_ideal_own_projection']:.2f} | {s['ct_int16_GB']:.0f} GB | {s['ct_float32_GB_if_not_quantised']:.0f} GB | {s['ct_niigz_GB_if_written_as_compressed_nifti']:.0f} GB, {s['niigz_write_hours_if_used']:.1f} h writing | {E['unet'][f'epoch_hours_full_coverage_{v}_variants']:.2f} |"
    )
w(
    f"\nLabels for all patients: {E['scenarios']['1']['labels_GB']:.1f} GB (once). Native phantoms {E['storage_GB']['native_phantoms']:.0f} GB; clean projections for all patients would be {E['storage_GB']['clean_projections_f32_all_patients_if_kept']:.0f} GB (largest single patient {E['storage_GB']['max_single_patient_projections_GB']:.1f} GB): **not kept**."
)
w("\n## 5b. Parallel simulation on one node (measured, job 203529: N simulators, one per GPU, same patient, 4 variants each)\n")
w("| simulators (GPUs) | wall [s] | variants/min | speed-up | efficiency |\n|---|---|---|---|---|")
sc = [dict(kv.split("=") for kv in ln.split()[:3]) for ln in (d / "scaling.txt").read_text().splitlines() if ln.startswith("N=")]
base = float(sc[0]["wall_s"])
for x in sc:
    n, wall = int(x["N"]), float(x["wall_s"])
    w(f"| {n} | {wall:.1f} | {4 * n * 60 / wall:.1f} | {n * base / wall:.2f}x | {100 * base / wall:.0f}% |")
w("\nIndependent simulators scale almost linearly up to 8 GPUs (shared phantom page cache, 8 CPUs each), so NFS and CPU are not bottlenecks at this size. Staging a 2.1 GB cache from `/home` to local NVMe took 14.6 s (0.14 GB/s, many small files; ESTIMATE ~14 min for the full 121 GB cache), persisting it to `/home` took 9.7 s.\n")
w("## 5c. Stage-1 smoke results (two patients: NOT a generalisation result)\n")
ov = json.load(open(d.parent / "smoke" / "overfit_report.json"))
w(f"- Fixed-batch overfit (4 patches of copd_22728 clean, 400 steps, {ov['seconds']:.0f} s): loss {ov['loss_first']:.2f} -> {ov['loss_last']:.2f}, pixel accuracy {100 * ov['pixel_accuracy']:.1f}%, macro Dice over the {len(ov['classes_in_batch'])} classes present {ov['macro_dice_present_classes']:.2f}. The network can learn this task.")
w("- Training with a forced stop at step 200 and resume to step 300 continued from the checkpoint; the conditioned arm D (FiLM, 7 metadata inputs) runs; evaluation, intensity-only baseline and NIfTI export run end to end. After 300 steps on one patient the validation macro Dice on the other patient is about 0.22 for the U-Net and 0.25 for the baseline: undertrained, reported only to show the pipeline works.")
w("\n## 6. Decisions\n")
w(
    "1. **Simulation is cheap in absolute terms, training is the larger cost.** All variants for all 67 patients take 0.4-4.4 GPU-hours (above); a full-coverage training epoch takes 0.1-1.2 h. Re-simulating inside the training loop would still add roughly 1-2x to every epoch (per 51 M-voxel volume: 5.5 s for a noisy variant from shared projections, about 10 s with its own projection, versus about 5.7 s of U-Net compute at 9 Mvoxel/s) and would keep LEAP and a 12 GB FBP working set resident next to the network; caching removes both. It is the right design, but its benefit is a 1-2x saving per epoch plus freedom from LEAP in the trainer, not an order of magnitude."
)
w(
    "2. **Layers:** (A) the resampled attenuation volume is NOT cached to disk (re-deriving it costs 1.1 s, reading a 0.2 GB file ~0.1 s: no gain); (B) clean projections are NOT cached (2.5 GB per patient: writing 1.8 s + reading 0.9 s is comparable to the 3.5 s forward projection); they are kept in GPU memory while all variants of one geometry group are produced; (C) the reconstructed CT (`npy_i16q`) and the shared label volume ARE cached."
)
w(
    "3. **Storage:** the whole training cache (67 patients x 10 variants = 115 GB CT + 6 GB labels) fits in `/tmp` NVMe (1.5 TB free) and, thanks to 503 GiB RAM, stays in the page cache; it also fits in `/dev/shm` (252 GB) but that reserves RAM. Plan: write to node-local NVMe, rsync a durable copy to `/home` after each patient (0.1-0.2 GB per CT; measured NFS write 0.64 GB/s), and stage back if a later job starts on a fresh node. Do not rely on local files surviving the job."
)
w(
    "4. **Processes:** simulator and trainer are separate processes (verified: GPU returns to 2 MiB between them). Up to 8 simulators (one per GPU) or 8 trainings in parallel are possible; whether 8 simulators scale linearly (shared NFS / CPU) is measured before the large run."
)
w(
    "5. **Not yet feasible:** training on held-out patients needs the 65 missing phantoms (181 GB; see `data_inventory.md`, transfer needs your go-ahead). Everything else proceeds with the two available patients (Stage 1) and is labelled a smoke test."
)
Path(sys.argv[2]).write_text("\n".join(L) + "\n")
print("written", sys.argv[2])
