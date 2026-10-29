# Feasibility report (Gate B / C): one COPD phantom, one acquisition, node004

Generated from measured files (`sim_report_p22728_0.8mm.json`, `unet_probe_p22728_0.8mm.json`, job 203524) plus the node probe (job 203521). **Measured** unless marked ESTIMATE.

## 1. What the node offers (probed on node004, job 203521)

| tier | size / free | write | read | notes |
|---|---|---|---|---|
| GPU VRAM (8 x RTX A5000) | 24138 MiB usable each | - | - | one job used 1 GPU; process exit returns it to 2 MiB |
| System RAM | 503 GiB | - | - | Slurm only reserves what `--mem` asks for |
| `/dev/shm` (RAM-backed tmpfs) | 252 GB | 2.5 GB/s | 6.8 GB/s (warm) | counts against RAM; lost at job end |
| `/tmp` and `/local` (node-local NVMe, same device) | 1.8 TB, 1.5 TB free (shared) | 1.1 GB/s | 2.8 GB/s uncached, 13.6 GB/s page-cache | no per-job directory (`SLURM_TMPDIR` unset, Slurm `TmpDisk=0`); not cleaned automatically as far as I can see |
| `/cachedata` (second local NVMe) | 7 TB, 3.1 TB free | not tested | not tested | root-owned: I cannot create a directory (other users have admin-made folders) |
| `/home` (NFS, shared lab) | 6.3 TB free | 0.69 GB/s | 0.27 GB/s uncached, 12 GB/s cached | durable; no quota tool available |
| `/scratch` (NFS) | 1.8 TB free of 74 TB (98% used) | - | - | not used |

Scheduler: partition `vram24` (node002, node004) has no wall-clock limit set; the project rule of < 10 h per job is kept anyway. `ssh tara` works from compute nodes (not from the laptop).

## 2. One-case measurements (copd_22728, 0.8 mm, 11000 views, 7 repetitions)

Grid (z,y,x) = (341, 452, 330); projection tensor 2417 MiB; attenuation volume 194 MiB; native phantom 1.63 GB.

| step | median [s] | min-max [s] | peak GPU [MiB] |
|---|---|---|---|
| 1 read phantom (page-cache warm; cold NFS ESTIMATE ~6 s at 0.27 GB/s) | 0.3371 | 0.3361-0.3392 | - |
| 2 material table -> mu (xlsx + element table; first call 0.33 s imports) | 0.0150 | 0.0133-0.3293 | - |
| 3 native 0.25 mm mu conversion (GPU LUT, reduced) | 0.3464 | 0.3460-0.3468 | - |
| 4 resample to 0.8 mm grid | 1.0996 | 1.0976-1.1023 | 2922 |
| 5 LEAP init + geometry (first call 0.28 s) | 0.0009 | 0.0006-0.2818 | - |
| 6 forward projection (clean, 11000 views) | 3.4607 | 3.4463-3.6620 | 7964 |
| 7 Poisson noise (in the count domain) | 0.4126 | 0.4125-0.4127 | 7758 |
| 8a helical FBP (noisy) | 4.8345 | 4.8168-4.8590 | 12636 |
| 8b helical FBP (clean) | 4.8889 | 4.8707-5.1248 | - |
| 9a mu -> HU (GPU) | 0.0018 | 0.0018-0.0018 | - |
| 9b device -> host copy | 0.1033 | 0.1013-0.1039 | - |

Cache write of the training CT (int16, 0.25 HU step): 0.20 s, 102 MB; labels 51 MB. Host RSS peak of the simulator process: 3840 MiB; training-probe process 6219 MiB.

**GPU release.** Same process after `del` + `empty_cache`: 632 MiB still in use (started at 212, peak 10184); PyTorch reports 8 MiB allocated. After the simulator process EXITED the job log shows 2 MiB used. `empty_cache` alone does not return LEAP's/CUDA's allocations; process separation does.

## 3. Cache formats (same CT volume, 3 write repetitions; read-patch = 60 random 96^3 patches, cold = page cache dropped for npy)

| tier | format | MB | write MB/s | patch96 warm ms | patch96 cold ms | full read s | max error HU |
|---|---|---|---|---|---|---|---|
| shm | npy_f32 | 203 | 1427 | 0.51 | 1.26 | 0.039 | 0.000 |
| shm | npy_f16 | 102 | 991 | 1.81 | 2.31 | 0.108 | 0.500 |
| shm | npy_i16q | 102 | 1368 | 0.67 | 1.26 | 0.076 | 0.125 |
| shm | zarr_raw_i16q | 151 | 956 | 1.90 | n/a | 0.175 | 0.125 |
| shm | zarr_lz4_i16q | 74 | 779 | 3.73 | n/a | 0.164 | 0.125 |
| shm | zarr_zstd1_i16q | 66 | 828 | 3.24 | n/a | 0.158 | 0.125 |
| nvme | npy_f32 | 203 | 1506 | 0.48 | 1.23 | 0.106 | 0.000 |
| nvme | npy_f16 | 102 | 992 | 1.82 | 2.30 | 0.119 | 0.500 |
| nvme | npy_i16q | 102 | 1367 | 0.61 | 1.22 | 0.085 | 0.125 |
| nvme | zarr_raw_i16q | 151 | 871 | 1.68 | n/a | 0.174 | 0.125 |
| nvme | zarr_lz4_i16q | 74 | 734 | 3.15 | n/a | 0.156 | 0.125 |
| nvme | zarr_zstd1_i16q | 66 | 770 | 3.17 | n/a | 0.159 | 0.125 |
| home_nfs | npy_f32 | 203 | 439 | 1.36 | 2.01 | 0.127 | 0.000 |
| home_nfs | npy_f16 | 102 | 537 | 3.15 | 3.74 | 0.152 | 0.500 |
| home_nfs | npy_i16q | 102 | 644 | 1.78 | 1.79 | 0.146 | 0.125 |
| home_nfs | zarr_raw_i16q | 151 | 86 | 16.60 | n/a | 0.594 | 0.125 |
| home_nfs | zarr_lz4_i16q | 74 | 98 | 20.13 | n/a | 0.398 | 0.125 |
| home_nfs | zarr_zstd1_i16q | 66 | 89 | 21.24 | n/a | 0.398 | 0.125 |

Compressed NIfTI of the same volume: 15.6 s and 178 MB (vs 0.15 s and 102 MB for `npy_i16q` on shm): about 105x slower and 1.8x larger.

**Choice: `npy_i16q`** (int16 = round(HU / 0.25), clipped at +-8191 HU): half of float32, max quantisation error 0.125 HU (SD 0.07 HU, far below the 9-54 HU noise), fastest sequential and random-patch reads, memory-mappable by every dataloader worker. float16 is slower to decode and has 0.5 HU error; Zarr (chunk 64^3) is 2-4x slower for patches on local disks and 10-30x on NFS, and compression saves only 35-50% of a volume that is cheap to store.

## 4. U-Net probe in a fresh process (no LEAP loaded; 1.4 M parameters, base 16, depth 3, bf16)

Time to open the cache, read the whole volume once and prepare class samples: 0.59 s.

| patch | batch | step [s] | peak allocated [MiB] | Mvoxel/s | patch load ms/item | GPU idle with 4 workers |
|---|---|---|---|---|---|---|
| 64 | 1 | 0.033 | 349 | 7.85 | 1.4 | 1.4% |
| 64 | 2 | 0.060 | 690 | 8.67 | 2.2 | 0.9% |
| 64 | 4 | 0.113 | 1347 | 9.28 | 1.2 | 0.9% |
| 96 | 1 | 0.104 | 1143 | 8.53 | 2.1 | 1.0% |
| 96 | 2 | 0.200 | 2255 | 8.86 | 2.2 | 1.2% |
| 96 | 4 | 0.392 | 4476 | 9.02 | 2.4 | 0.8% |
| 128 | 1 | 0.247 | 2666 | 8.50 | 3.8 | 0.8% |
| 128 | 2 | 0.472 | 5299 | 8.88 | 3.7 | 1.0% |
| 128 | 4 | 0.925 | 10568 | 9.07 | 3.7 | 0.9% |

Training is compute-bound at about 9 Mvoxel/s for every patch/batch size; with 4 or more loader workers the GPU waits for data only about 1% of the time, so I/O from the cache does not limit training. 3D patches of 96^3 with batch 4 fit in 4.5 GB, so eight independent trainings fit on the 8 GPUs.

## 5. Dataset cost for all 67 patients (ESTIMATE: measured one-case times scaled per patient by views x sqrt(voxels); see `dataset_cost_estimate.json`)

| variants / patient | GPU-h: noise variants (one shared projection) | GPU-h: every variant needs its own projection | wall-h on 8 GPUs (ideal) | CT cache int16 | if stored as float32 | if written as .nii.gz | full-coverage epoch of 47 train patients [h] |
|---|---|---|---|---|---|---|---|
| 1 | 0.44 | 0.44 | 0.05 | 11 GB | 23 GB | 20 GB, 0.4 h writing | 0.12 |
| 3 | 0.73 | 1.31 | 0.16 | 34 GB | 69 GB | 60 GB, 1.3 h writing | 0.37 |
| 5 | 1.02 | 2.18 | 0.27 | 57 GB | 115 GB | 100 GB, 2.1 h writing | 0.62 |
| 10 | 1.75 | 4.36 | 0.55 | 115 GB | 229 GB | 201 GB, 4.2 h writing | 1.24 |

Labels for all patients: 5.7 GB (once). Native phantoms 185 GB; clean projections for all patients would be 189 GB (largest single patient 3.2 GB): **not kept**.

## 5b. Parallel simulation on one node (measured, job 203529: N simulators, one per GPU, same patient, 4 variants each)

| simulators (GPUs) | wall [s] | variants/min | speed-up | efficiency |
|---|---|---|---|---|
| 1 | 37.3 | 6.4 | 1.00x | 100% |
| 2 | 36.4 | 13.2 | 2.05x | 102% |
| 4 | 38.3 | 25.1 | 3.90x | 97% |
| 8 | 40.8 | 47.0 | 7.31x | 91% |

Independent simulators scale almost linearly up to 8 GPUs (shared phantom page cache, 8 CPUs each), so NFS and CPU are not bottlenecks at this size. Staging a 2.1 GB cache from `/home` to local NVMe took 14.6 s (0.14 GB/s, many small files; ESTIMATE ~14 min for the full 121 GB cache), persisting it to `/home` took 9.7 s.

## 5c. Stage-1 smoke results (two patients: NOT a generalisation result)

- Fixed-batch overfit (4 patches of copd_22728 clean, 400 steps, 158 s): loss 3.56 -> 0.17, pixel accuracy 97.8%, macro Dice over the 14 classes present 0.92. The network can learn this task.
- Training with a forced stop at step 200 and resume to step 300 continued from the checkpoint; the conditioned arm D (FiLM, 7 metadata inputs) runs; evaluation, intensity-only baseline and NIfTI export run end to end. After 300 steps on one patient the validation macro Dice on the other patient is about 0.22 for the U-Net and 0.25 for the baseline: undertrained, reported only to show the pipeline works.

## 6. Decisions

1. **Simulation is cheap in absolute terms, training is the larger cost.** All variants for all 67 patients take 0.4-4.4 GPU-hours (above); a full-coverage training epoch takes 0.1-1.2 h. Re-simulating inside the training loop would still add roughly 1-2x to every epoch (per 51 M-voxel volume: 5.5 s for a noisy variant from shared projections, about 10 s with its own projection, versus about 5.7 s of U-Net compute at 9 Mvoxel/s) and would keep LEAP and a 12 GB FBP working set resident next to the network; caching removes both. It is the right design, but its benefit is a 1-2x saving per epoch plus freedom from LEAP in the trainer, not an order of magnitude.
2. **Layers:** (A) the resampled attenuation volume is NOT cached to disk (re-deriving it costs 1.1 s, reading a 0.2 GB file ~0.1 s: no gain); (B) clean projections are NOT cached (2.5 GB per patient: writing 1.8 s + reading 0.9 s is comparable to the 3.5 s forward projection); they are kept in GPU memory while all variants of one geometry group are produced; (C) the reconstructed CT (`npy_i16q`) and the shared label volume ARE cached.
3. **Storage:** the whole training cache (67 patients x 10 variants = 115 GB CT + 6 GB labels) fits in `/tmp` NVMe (1.5 TB free) and, thanks to 503 GiB RAM, stays in the page cache; it also fits in `/dev/shm` (252 GB) but that reserves RAM. Plan: write to node-local NVMe, rsync a durable copy to `/home` after each patient (0.1-0.2 GB per CT; measured NFS write 0.64 GB/s), and stage back if a later job starts on a fresh node. Do not rely on local files surviving the job.
4. **Processes:** simulator and trainer are separate processes (verified: GPU returns to 2 MiB between them). Up to 8 simulators (one per GPU) or 8 trainings in parallel are possible; whether 8 simulators scale linearly (shared NFS / CPU) is measured before the large run.
5. **Not yet feasible:** training on held-out patients needs the 65 missing phantoms (181 GB; see `data_inventory.md`, transfer needs your go-ahead). Everything else proceeds with the two available patients (Stage 1) and is labelled a smoke test.
