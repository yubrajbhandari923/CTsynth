# AAPM DL-Spectral CT — full-data acquisition plan (HPC)

> Status 2026-10-13: still a plan (task T-017, not executed). Use the native protocol in `docs/hpc.md` (`research hpc datasets` for
> placement on the cluster; do not create a project-specific dataset registry).

**Not executed on the laptop.** Per `docs/hpc.md`, HPC is run-only: no code is developed there.
Nothing in this plan has been run; it is a specification. Bulk files are gitignored (`data/aapm/`, `*.npy.gz`).

## Source
Zenodo **10.5281/zenodo.14262737**, CC-BY-4.0 — Sidky & Pan, "Data repository for the 2022 AAPM
deep-learning spectral CT grand challenge". Report: Med Phys 51(2):772–785, arXiv:2212.06718.
**Attribution is required** in any publication using this data.

## Files (sizes and md5 read from the Zenodo API 2026-10-11; verify again at download time)
| File | Bytes | md5 |
|---|---|---|
| `Phantom_Adipose.npy.gz` | 102,183,683 | `1d0c7239fa747a285c7ef1bcf684e3da` |
| `Phantom_Fibroglandular.npy.gz` | 113,764,766 | `746641e2910c231a39441d242bdfd458` |
| `Phantom_Calcification.npy.gz` | 6,937,257 | `2e04dc154dc36087a2404007c1b36baa` |
| `lowkVpImages.npy.gz` | 717,374,937 | `8afab98fef99400d4f293f57281fa6a3` |
| `highkVpImages.npy.gz` | 716,205,150 | `f600ba6546c0f8fab3b227e3d470c5ed` |
| `lowkVpTransmission.npy.gz` | 850,758,740 | `8c48d814d3db4c34f1cb96348cd1c9d9` |
| `highkVpTransmission.npy.gz` | 842,546,719 | `57011ed1b0346d1062627a6ba86f6b7e` |
| `starting_kit.tgz` | 102,706,887 | `7aaf6b55708a6576ebfa91e43ec80082` ✅ already verified locally |
| **Total** | **3,452,478,139 (3.22 GiB)** | |

**Decompressed footprint:** each `(1000,512,512)` float32 array is 1000·512·512·4 B ≈ **0.98 GiB**;
transmission `(1000,256,1024)` float32 ≈ **0.98 GiB** each. Six large arrays ≈ **5.9 GiB** uncompressed,
plus ~3.2 GiB of archives. **Budget ≥ 12 GiB** of scratch to hold both during decompression.

## Why this cannot be subset
Each `.npy.gz` is a single gzip stream holding one 1000-case array: there is **no random access**.
Reading even one case requires streaming the whole file. Hence: no partial download, and the laptop
plan is "synthetic only" (user decision, 2026-10-11).

## Procedure (on HPC, from a clean checkout at a recorded commit)
```bash
mkdir -p "$SCRATCH/ctsynth_data/aapm" && cd "$SCRATCH/ctsynth_data/aapm"
uv run python scripts/fetch_aapm.py --dest .            # downloads + verifies md5, refuses on mismatch
# decompress only what a given experiment needs:
gunzip -k Phantom_Adipose.npy.gz                        # keeps the .gz
```
Then point configs at the directory via `AAPM_DATA_DIR`; the loader memory-maps per case
(`src/ctsynth/aapm/loader.py`), so peak RAM is O(one case), not O(dataset).

`scripts/fetch_aapm.py` streams each file, computes md5 while writing, compares against the table above
(fetched live from the Zenodo API), and **deletes any file whose hash does not match**. It is resumable
by skipping files that already verify.

## Decompression-free alternative (preferred if scratch is tight)
`gzip` streams decode sequentially, so a one-pass converter can read each `.npy.gz` and write per-case
`.npy` shards (or a single `.zarr`/`.h5` with chunking) without ever holding the full array:
1000 shards × ~1 MiB per phantom material. This also restores random access for shuffled training.
**Not implemented** — it is the recommended first HPC task if full-data training is approved.

## What the laptop can and cannot claim
- Laptop: synthetic-data smoke benchmark only. **No AAPM-dependent number may be reported from it.**
- Full-resolution (512², 1000 cases) results and any comparison to the published leaderboard
  (winner s1 = 6.8e-7) require this acquisition plus an HPC run recorded with `run_meta.json`.
- Reduced-resolution results are **not** comparable to published numbers: the metric s1 divides by
  `3n` with `n = 512²`, and resizing changes both the geometry and the per-pixel error statistics.
