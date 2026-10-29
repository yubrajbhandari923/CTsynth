#!/usr/bin/env python
"""Process B of the feasibility benchmark: a FRESH process (no LEAP) that loads the cached CT and trains a 3D U-Net for a few steps.

Measures: cache open + class-sample time (time to get data into the training process), GPU memory per (patch, batch), optimiser step time
(7 repetitions after warm-up), patch-loading cost, DataLoader throughput for several worker counts, and GPU idle fraction caused by I/O.
usage: unet_probe.py --cache /dev/shm/x/p22728 --out report.json
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.seg.dataset import PatchDataset, VolumeRecord, class_voxel_samples
from ctsynth.seg.losses import dice_ce_loss
from ctsynth.seg.unet3d import UNet3D
from ctsynth.xcat.cache import VolumeReader
from ctsynth.xcat.profiling import MIB, MemSampler, device_used_bytes, host_rss_peak_mib, sync

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--patches", type=int, nargs="+", default=[64, 96, 128])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--workers", type=int, nargs="+", default=[0, 4, 8, 16])
ap.add_argument("--base", type=int, default=16)
ap.add_argument("--reps", type=int, default=7)
a = ap.parse_args()
dev = torch.device("cuda:0")
R: dict = {
    "torch": torch.__version__,
    "gpu_used_MiB_at_process_start": device_used_bytes() / MIB,
    "model": {"base": a.base, "depth": 3},
}
cache = Path(a.cache)
ct_path = next(cache.glob("ct_*.npy"))
t0 = time.perf_counter()
labels = np.load(cache / "labels.npy")
rd = VolumeReader(ct_path, "npy_i16q")
_ = rd.read_all()  # touch the whole volume: pulls it into this process / the page cache
samples = class_voxel_samples(labels)
R["time_to_open_cache_and_prepare_s"] = time.perf_counter() - t0
rec = VolumeRecord("probe", str(ct_path), "npy_i16q", str(cache / "labels.npy"), (), samples)
R["volume_shape"], R["n_classes"] = list(labels.shape), int(labels.max()) + 1
n_classes = 15
R["configs"] = []
for patch in a.patches:
    for bs in a.batches:
        torch.manual_seed(0)
        net = UNet3D(1, n_classes, base=a.base, depth=3).to(dev)
        opt = torch.optim.AdamW(net.parameters(), 1e-3)
        ds = PatchDataset([rec], patch, 64, seed=0)
        x = torch.stack([ds[i][0] for i in range(bs)]).to(dev)
        y = torch.stack([ds[i][1] for i in range(bs)]).to(dev)
        e: dict = {"patch": patch, "batch": bs, "params": sum(p.numel() for p in net.parameters())}

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = dice_ce_loss(net(x), y)
            loss.backward()
            opt.step()
            return loss

        try:
            for _ in range(2):
                step()
            sync()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(a.reps):
                sync()
                t0 = time.perf_counter()
                loss = step()
                sync()
                ts.append(time.perf_counter() - t0)
            e.update(
                step_median_s=float(np.median(ts)),
                step_std_s=float(np.std(ts)),
                step_min_s=min(ts),
                step_max_s=max(ts),
                peak_allocated_MiB=torch.cuda.max_memory_allocated() / MIB,
                peak_reserved_MiB=torch.cuda.max_memory_reserved() / MIB,
                loss_finite=bool(torch.isfinite(loss)),
                voxels_per_s=bs * patch**3 / float(np.median(ts)),
            )
            # patch loading (CPU): one item, then loader throughput by worker count
            t0 = time.perf_counter()
            for i in range(20):
                ds[i]
            e["patch_load_ms_per_item"] = 1e3 * (time.perf_counter() - t0) / 20
            e["loaders"] = []
            for w in a.workers:
                n_items = 4 * bs * 8
                dl = torch.utils.data.DataLoader(
                    PatchDataset([rec], patch, n_items, seed=1),
                    batch_size=bs,
                    num_workers=w,
                    pin_memory=True,
                    persistent_workers=False,
                )
                t0 = time.perf_counter()
                wait = busy = 0.0
                it = iter(dl)
                n = 0
                while True:
                    t1 = time.perf_counter()
                    try:
                        bx, by, _ = next(it)
                    except StopIteration:
                        break
                    wait += time.perf_counter() - t1
                    t2 = time.perf_counter()
                    x, y = bx.to(dev, non_blocking=True), by.to(dev, non_blocking=True)
                    step()
                    sync()
                    busy += time.perf_counter() - t2
                    n += 1
                wall = time.perf_counter() - t0
                e["loaders"].append(
                    {
                        "workers": w,
                        "batches": n,
                        "wall_s": wall,
                        "s_per_batch": wall / n,
                        "gpu_idle_fraction": wait / wall,
                        "gpu_busy_fraction": busy / wall,
                    }
                )
        except torch.OutOfMemoryError as err:
            e["oom"] = str(err).split("\n")[0][:200]
        R["configs"].append(e)
        print({k: v for k, v in e.items() if k != "loaders"}, flush=True)
        for lw in e.get("loaders", []):
            print("   ", lw, flush=True)
        del net, opt, x, y
        gc.collect()
        torch.cuda.empty_cache()
R["host_rss_peak_MiB"] = host_rss_peak_mib()
with MemSampler() as ms:
    pass
R["gpu_used_MiB_at_end"] = ms.base / MIB
Path(a.out).write_text(json.dumps(R, indent=1, default=str))
print("wrote", a.out)
