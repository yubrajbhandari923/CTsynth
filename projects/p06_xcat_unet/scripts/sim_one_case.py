#!/usr/bin/env python
"""Process A of the one-case feasibility benchmark: simulate ONE COPD phantom at ONE acquisition and measure every step.

Steps (each timed with CUDA syncs, 7 repetitions unless noted): read phantom, material table -> mu, native-resolution mu conversion,
resample, LEAP init, forward projection, Poisson noise, helical FBP (noisy and clean), HU conversion, device->host copy, cache write per
format x location, reopen, random 96^3 patch reads (cold = page cache dropped, warm), compressed NIfTI for comparison, simulator release.
Leaves a training cache (int16-quantised HU + group labels) on the fast tier for process B (unet_probe.py), then EXITS so the CUDA
context and LEAP's allocations are returned to the driver (verified by nvidia-smi in the job script).
usage: sim_one_case.py --root DATA --patient 22728 --vox 0.8 --fast /dev/shm/x --nvme /tmp/x --home DIR --out report.json
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.xcat import cache as C
from ctsynth.xcat.cases import find_copd_case
from ctsynth.xcat.leap_sim import (
    Acquisition,
    fbp,
    forward_project,
    make_leap,
    plan_scan,
    projection_bytes,
)
from ctsynth.xcat.materials import build_material_model, crosscheck_dukesim
from ctsynth.xcat.noise import apply_poisson_noise, make_generator
from ctsynth.xcat.phantom import make_grid, mu_to_hu, open_phantom, resample_phantom
from ctsynth.xcat.profiling import (
    MIB,
    device_used_bytes,
    drop_file_cache,
    host_rss_peak_mib,
    sync,
    timed,
)

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--patient", default="22728")
ap.add_argument("--vox", type=float, default=0.8)
ap.add_argument("--fast", required=True, help="RAM-backed tier (/dev/shm/...)")
ap.add_argument("--nvme", required=True, help="node-local NVMe tier (/tmp/...)")
ap.add_argument("--home", required=True, help="durable NFS tier")
ap.add_argument("--out", required=True)
ap.add_argument("--n0", type=float, default=1e5)
ap.add_argument("--reps", type=int, default=7)
a = ap.parse_args()
dev = torch.device("cuda:0")
R: dict = {
    "patient": a.patient,
    "vox_mm": a.vox,
    "n0": a.n0,
    "steps": {},
    "host": os.uname().nodename,
    "torch": torch.__version__,
}
used0 = device_used_bytes()
R["gpu_used_MiB_at_process_start"] = used0 / MIB


def rec(t):
    R["steps"][t.op] = t.summary()
    s = R["steps"][t.op]
    print(
        f"{t.op}: median {s['median_s']:.4f} s (n={s['n']}) peak_dev={s['peak_device_mib']}",
        flush=True,
    )


case = find_copd_case(a.root, a.patient)
R["bin_bytes"] = case.bin_path.stat().st_size
t, _ = timed(
    "01_read_phantom_file_into_ram",
    lambda: np.fromfile(case.bin_path, dtype=np.uint8),
    a.reps,
    warmup=0,
)
rec(t)
vol = open_phantom(case.bin_path, case.shape_zyx)
t, model = timed(
    "02_material_table_to_mu", lambda: build_material_model(case.xlsx, 70.0), a.reps, warmup=0
)
rec(t)
R["material_crosscheck"] = crosscheck_dukesim(model, case.lac_bin, case.conversion_in)
lut = torch.from_numpy(model.mu_mm).to(dev)


def native_mu():
    tot = torch.zeros((), device=dev)
    for z0 in range(0, case.shape_zyx[0], 64):
        tot += lut[torch.from_numpy(np.array(vol[z0 : z0 + 64])).to(dev).long()].sum()
    return tot


t, _ = timed("03_native_resolution_mu_conversion", native_mu, a.reps)
rec(t)
grid = make_grid(case.shape_zyx, a.vox)
t, res = timed(
    "04_resample_to_grid", lambda: resample_phantom(vol, grid, model, dev), a.reps, memory=True
)
rec(t)
acq = Acquisition()
plan = plan_scan(grid, acq)
R["grid_zyx"], R["n_views"], R["proj_MiB"] = (
    list(grid.shape),
    plan.n_views,
    projection_bytes(plan) / MIB,
)
R["vol_MiB"] = grid.n_voxels * 4 / MIB
t, leap = timed("05_leap_init_and_geometry", lambda: make_leap(grid, plan), a.reps, warmup=0)
rec(t)
t, p_clean = timed(
    "06_forward_project", lambda: forward_project(leap, res.mu_mm, plan), a.reps, memory=True
)
rec(t)
noisy = torch.empty_like(p_clean)
t, _ = timed(
    "07_poisson_noise",
    lambda: apply_poisson_noise(p_clean, a.n0, make_generator(dev, 0, 1), out=noisy),
    a.reps,
    memory=True,
)
rec(t)
t, mu_noisy = timed("08a_helical_fbp_noisy", lambda: fbp(leap, noisy, grid), a.reps, memory=True)
rec(t)
t, mu_clean = timed("08b_helical_fbp_clean", lambda: fbp(leap, p_clean, grid), a.reps)
rec(t)
mu_w = model.mu_water_mm
t, hu_t = timed("09a_mu_to_hu_on_gpu", lambda: mu_to_hu(mu_noisy, mu_w), a.reps)
rec(t)
t, hu = timed("09b_device_to_host_copy", lambda: hu_t.cpu().numpy(), a.reps)
rec(t)
R["host_rss_peak_MiB_after_sim"] = host_rss_peak_mib()

# noise sweep: SD of (noisy - clean) HU in eroded muscle (pure noise, deterministic structure cancels)
mus = [k for k, v in model.groups.items() if v["name"] == "muscle"][0]
mask = (res.groups == mus).float()[None, None]
mask = (-torch.nn.functional.max_pool3d(-mask, 7, 1, 3))[0, 0] > 0.5
hu_clean = mu_to_hu(mu_clean, mu_w)
R["noise_sweep"] = []
for n0 in (3e5, 1e5, 3e4, 1e4, 3e3):
    apply_poisson_noise(p_clean, n0, make_generator(dev, 0, 2), out=noisy)
    d = (mu_to_hu(fbp(leap, noisy, grid), mu_w) - hu_clean)[mask]
    R["noise_sweep"].append(
        {
            "n0": n0,
            "muscle_noise_sd_hu": float(d.std()),
            "muscle_bias_hu": float(d.mean()),
            "n_voxels": int(mask.sum()),
        }
    )
    print(R["noise_sweep"][-1], flush=True)

# cache matrix: formats x tiers
tiers = {"shm": Path(a.fast), "nvme": Path(a.nvme), "home_nfs": Path(a.home)}
rng = np.random.default_rng(0)
P = 96
R["cache"] = []
for tier, base in tiers.items():
    for fmt in C.FORMATS:
        d = base / "matrix"
        w_t, bytes_ = [], 0
        for rep in range(3):
            w = C.write_hu(d / f"ct_{fmt}", hu, fmt, chunks=(64, 64, 64))
            w_t.append(w.seconds)
            bytes_ = w.bytes
        o_t = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            rd = C.VolumeReader(w.path, fmt)
            o_t.append(time.perf_counter() - t0)
        origins = [tuple(int(rng.integers(0, s - P)) for s in rd.shape) for _ in range(60)]
        warm = []
        for o in origins:
            t0 = time.perf_counter()
            rd.read_patch(o, (P, P, P))
            warm.append(time.perf_counter() - t0)
        cold = []
        if fmt.startswith("npy"):
            for o in origins[:30]:
                drop_file_cache(w.path)
                rd2 = C.VolumeReader(w.path, fmt)
                t0 = time.perf_counter()
                rd2.read_patch(o, (P, P, P))
                cold.append(time.perf_counter() - t0)
        full = []
        for _ in range(3):
            if fmt.startswith("npy"):
                drop_file_cache(w.path)
            t0 = time.perf_counter()
            C.VolumeReader(w.path, fmt).read_all()
            full.append(time.perf_counter() - t0)
        patch_mib = P**3 * (4 if fmt == "npy_f32" else 2) / MIB
        e = {
            "tier": tier,
            "fmt": fmt,
            "bytes": bytes_,
            "write_s_median": float(np.median(w_t)),
            "write_MB_s": hu.nbytes / np.median(w_t) / 1e6,
            "reopen_s_median": float(np.median(o_t)),
            "patch96_warm_ms_median": 1e3 * float(np.median(warm)),
            "patch96_cold_ms_median": 1e3 * float(np.median(cold)) if cold else None,
            "patch96_MiB": patch_mib,
            "full_read_s_median": float(np.median(full)),
            "max_abs_hu_err": float(np.abs(rd.read_all() - hu).max()),
        }
        R["cache"].append(e)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}, flush=True)
        shutil.rmtree(d, ignore_errors=True)

# nii.gz for comparison (what the earlier pipeline wrote)
import nibabel as nib  # noqa: E402
from nibabel import openers  # noqa: E402

openers.Opener.default_compresslevel = 6
from ctsynth.xcat.phantom import affine_from_axcodes  # noqa: E402

t0 = time.perf_counter()
nib.save(
    nib.Nifti1Image(hu, affine_from_axcodes(grid.shape, grid.vox_mm)),
    str(Path(a.nvme) / "compare.nii.gz"),
)
R["nifti_gz_write_s"] = time.perf_counter() - t0
R["nifti_gz_bytes"] = (Path(a.nvme) / "compare.nii.gz").stat().st_size
(Path(a.nvme) / "compare.nii.gz").unlink()

# the training cache for process B: compact int16 CT + group labels on the fast tier
fast = Path(a.fast) / f"p{a.patient}"
t0 = time.perf_counter()
w = C.write_hu(fast / "ct_noisy_n0_1e5", hu, "npy_i16q")
np.save(fast / "labels.npy", res.groups.cpu().numpy())
R["train_cache_write_s"] = time.perf_counter() - t0
R["train_cache_bytes"] = {
    "ct_i16": w.bytes,
    "labels_u8": (fast / "labels.npy").stat().st_size,
    "native_uint8_phantom": R["bin_bytes"],
    "clean_projections_f32": projection_bytes(plan),
    "attenuation_f32": grid.n_voxels * 4,
}
R["ct_hu_stats"] = {
    "min": float(hu.min()),
    "max": float(hu.max()),
    "mean_muscle": float(hu_t[mask].mean()),
}
R["group_names"] = {str(k): v["name"] for k, v in model.groups.items()}
R["host_rss_peak_MiB_final"] = host_rss_peak_mib()

# release: drop every reference we hold; report what the process still holds before exit
R["gpu_used_MiB_before_release"] = device_used_bytes() / MIB
del leap, p_clean, noisy, mu_noisy, mu_clean, hu_t, hu_clean, res, mask, lut
gc.collect()
torch.cuda.empty_cache()
sync()
R["gpu_used_MiB_after_del_and_empty_cache_same_process"] = device_used_bytes() / MIB
R["torch_allocated_MiB_after_release"] = torch.cuda.memory_allocated() / MIB
R["nvidia_smi_process_view"] = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
    capture_output=True,
    text=True,
).stdout.strip()
Path(a.out).write_text(json.dumps(R, indent=1, default=str))
print("wrote", a.out, flush=True)
