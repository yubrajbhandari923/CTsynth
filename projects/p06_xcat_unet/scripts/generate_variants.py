#!/usr/bin/env python
"""Simulate controlled CT variants of COPD XCAT phantoms and cache them compactly (resumable).

For each patient: resample the phantom once per energy, forward-project once per (energy, pitch, view decimation) group, then for every
variant in the group add Poisson noise to the clean projections (or not), apply channel apodisation, run LEAP helical FBP, convert to HU and
write int16-quantised .npy + JSON metadata. Optional --metrics computes error / noise / sharpness metrics against the ideal image.
usage: generate_variants.py --root DATA --patients 22728 21535 --variants clean n0_1e5 ... --cache DIR [--metrics] [--seed 0]
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from ctsynth.seg.dataset import class_voxel_samples
from ctsynth.xcat import cache as C
from ctsynth.xcat.cases import find_copd_case
from ctsynth.xcat.leap_sim import SOD_MM, fbp, forward_project, make_leap, plan_scan
from ctsynth.xcat.materials import build_material_model
from ctsynth.xcat.metrics import edge_width_mm, erode, noise_autocorrelation, region_errors
from ctsynth.xcat.noise import apply_poisson_noise, make_generator
from ctsynth.xcat.phantom import make_grid, mu_to_hu, open_phantom, resample_phantom
from ctsynth.xcat.profiling import MIB, MemSampler, host_rss_peak_mib, sync
from ctsynth.xcat.variants import VariantSpec, apodise_channels, family_variants, sample_combined

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--patients", nargs="+", required=True)
ap.add_argument("--variants", nargs="*", default=["clean"])
ap.add_argument(
    "--combos", type=int, default=0, help="number of family-E combined variants per patient"
)
ap.add_argument("--vox", type=float, default=0.8)
ap.add_argument("--cache", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--metrics", action="store_true")
ap.add_argument("--fmt", default="npy_i16q")
a = ap.parse_args()
dev = torch.device("cuda:0")
cache = Path(a.cache)
fam = family_variants()
specs: list[VariantSpec] = []
for v in a.variants:
    specs += [s for s in fam.values() if s.family == v[7:]] if v.startswith("family:") else [fam[v]]
specs += [sample_combined(a.seed, i) for i in range(a.combos)]
print("variants:", [s.name for s in specs], flush=True)
logf = open(cache / "generation_log.jsonl", "a") if cache.exists() else None


def log(rec):
    global logf
    if logf is None:
        cache.mkdir(parents=True, exist_ok=True)
        logf = open(cache / "generation_log.jsonl", "a")
    logf.write(json.dumps(rec, default=str) + "\n")
    logf.flush()


for pi, pid in enumerate(a.patients):
    t_patient = time.perf_counter()
    case = find_copd_case(a.root, pid)
    pdir = cache / f"p{pid}"
    pdir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not (pdir / f"ct_{s.name}.json").exists()]
    if not todo:
        print(f"p{pid}: all variants cached, skipping", flush=True)
        continue
    t0 = time.perf_counter()
    vol = open_phantom(case.bin_path, case.shape_zyx)
    grid = make_grid(case.shape_zyx, a.vox)
    per_energy: dict[float, tuple] = {}

    def get_energy(e):
        if e not in per_energy:
            per_energy.clear()  # keep one energy resident (GPU memory)
            gc.collect()
            torch.cuda.empty_cache()
            m = build_material_model(case.xlsx, e)
            res = resample_phantom(vol, grid, m, dev)
            per_energy[e] = (m, res, mu_to_hu(res.mu_mm, m.mu_water_mm))
        return per_energy[e]

    m70, res70, _ = get_energy(70.0)
    labels = res70.groups.cpu().numpy()
    if not (pdir / "labels.npy").exists():
        np.save(pdir / "labels.npy", labels)
        np.savez(
            pdir / "class_samples.npz",
            **{str(k): v for k, v in class_voxel_samples(labels).items()},
        )
    names = {k: v["name"] for k, v in m70.groups.items()}
    json.dump(names, open(pdir / "group_names.json", "w"))
    t_prep = time.perf_counter() - t0
    groups: dict[tuple, list[VariantSpec]] = {}
    for s in todo:
        groups.setdefault(s.group_key(), []).append(s)
    for key, members in groups.items():
        energy = key[0]
        m, res, ideal = get_energy(energy)
        plan = plan_scan(grid, members[0].acq)
        leap = make_leap(grid, plan)
        sync()
        t0 = time.perf_counter()
        p_clean = forward_project(leap, res.mu_mm, plan)
        sync()
        t_fwd = time.perf_counter() - t0
        work = torch.empty_like(p_clean)
        clean_by_sigma: dict[tuple[float, float], torch.Tensor] = {}
        if a.metrics:
            nz = grid.shape[0]
            zc = (torch.arange(nz, device=dev).float() + 0.5) * grid.vox_mm - nz * grid.vox_mm / 2
            yy = torch.arange(grid.shape[1], device=dev)[:, None].float()
            xx = torch.arange(grid.shape[2], device=dev)[None, :].float()
            rad = torch.sqrt(
                ((yy + 0.5) * grid.vox_mm - grid.shape[1] * grid.vox_mm / 2) ** 2
                + ((xx + 0.5) * grid.vox_mm - grid.shape[2] * grid.vox_mm / 2) ** 2
            )
            fov_r = SOD_MM * np.sin(0.5 * 900 / 1050.0)
            valid = (
                ((zc.abs() <= plan.scan_length_mm / 2 - 20.0)[:, None, None])
                & (rad < 0.95 * fov_r)[None]
            )
            grp = res.groups
            body = erode(grp > 0, 2) & valid
            lung_ids = [
                k for k, n in names.items() if n in ("lung", "lung_parenchyma", "emphysema")
            ]
            lung = (
                erode(torch.isin(grp, torch.tensor(lung_ids, device=dev, dtype=torch.uint8)), 2)
                & valid
            )
            muscle = erode(grp == [k for k, n in names.items() if n == "muscle"][0], 3) & valid
            lung_centroid = torch.nonzero(lung).float().mean(0).cpu().numpy()
        # clean variants first so noisy ones can reuse the same-kernel clean reconstruction
        for s in sorted(members, key=lambda s: (s.n0 is not None, s.apod_sigma_px, s.lowpass_w)):
            vi = [x.name for x in specs].index(s.name)
            if (
                a.metrics
                and s.n0 is not None
                and (s.apod_sigma_px, s.lowpass_w) not in clean_by_sigma
            ):
                # same-kernel clean reconstruction = noise-free reference for this variant's noise SD / texture
                work.copy_(p_clean)
                apodise_channels(work, s.apod_sigma_px)
                leap.set_FBPlowpass(s.lowpass_w)
                clean_by_sigma[(s.apod_sigma_px, s.lowpass_w)] = mu_to_hu(
                    fbp(leap, work, grid, inplace=True), m.mu_water_mm
                )
                leap.set_FBPlowpass(1.0)
            t0 = time.perf_counter()
            if s.n0 is None:
                work.copy_(p_clean)
            else:
                apply_poisson_noise(
                    p_clean, s.n0, make_generator(dev, a.seed, int(pid), vi), out=work
                )
            apodise_channels(work, s.apod_sigma_px)
            sync()
            t_pre = time.perf_counter() - t0
            with MemSampler() as ms:
                t0 = time.perf_counter()
                leap.set_FBPlowpass(s.lowpass_w)
                mu = fbp(leap, work, grid, inplace=True)
                leap.set_FBPlowpass(1.0)
                sync()
                t_fbp = time.perf_counter() - t0
            hu = mu_to_hu(mu, m.mu_water_mm)
            if s.n0 is None and (a.metrics):
                clean_by_sigma[(s.apod_sigma_px, s.lowpass_w)] = hu.clone()
            t0 = time.perf_counter()
            hu_np = hu.cpu().numpy()
            w = C.write_hu(pdir / f"ct_{s.name}", hu_np, a.fmt)
            t_write = time.perf_counter() - t0
            meta = {
                "patient": pid,
                "variant": s.name,
                "family": s.family,
                "energy_keV": s.acq.energy_kev,
                "pitch": s.acq.pitch,
                "view_decimation": s.acq.view_decimation,
                "n0": s.n0,
                "apod_sigma_px": s.apod_sigma_px,
                "lowpass_w": s.lowpass_w,
                "seed": a.seed,
                "stream": [int(pid), vi],
                "vox_mm": a.vox,
                "grid_zyx": list(grid.shape),
                "n_views": plan.n_views,
                "scan_plan": plan.to_dict(),
                "cond": list(s.cond_vector()),
                "fmt": a.fmt,
                "hu_step": C.HU_STEP,
                "bytes": w.bytes,
                "times_s": {
                    "noise_and_apodise": t_pre,
                    "fbp": t_fbp,
                    "hu_and_write": t_write,
                    "group_forward": t_fwd,
                    "patient_prep": t_prep,
                },
                "peak_device_MiB_fbp": ms.peak / MIB,
                "host_rss_peak_MiB": host_rss_peak_mib(),
                "mu_water_mm": m.mu_water_mm,
            }
            if a.metrics:
                ref_hu = ideal
                rec = {
                    "all_valid": region_errors(hu, ref_hu, valid),
                    "body": region_errors(hu, ref_hu, body),
                    "lung": region_errors(hu, ref_hu, lung),
                    "per_group": {},
                }
                for k, nm in names.items():
                    mk = erode(grp == k, 1) & valid
                    if int(mk.sum()) > 200:
                        rec["per_group"][nm] = region_errors(hu, ref_hu, mk)
                if s.n0 is not None and (s.apod_sigma_px, s.lowpass_w) in clean_by_sigma:
                    noise = hu - clean_by_sigma[(s.apod_sigma_px, s.lowpass_w)]
                    rec["noise_muscle"] = {
                        "sd_hu": float(noise[muscle].std()),
                        "mean_hu": float(noise[muscle].mean()),
                        **{"acf_" + k: v for k, v in noise_autocorrelation(noise, muscle).items()},
                    }
                rec["edge"] = edge_width_mm(hu, tuple(float(v) for v in lung_centroid), a.vox)
                rec["edge_ideal_reference"] = edge_width_mm(
                    ref_hu, tuple(float(v) for v in lung_centroid), a.vox
                )
                meta["metrics"] = rec
                zc_, yc_, xc_ = (int(round(v)) for v in lung_centroid)
                np.savez_compressed(
                    pdir / f"slices_{s.name}.npz",
                    axial=hu_np[zc_].astype(np.float16),
                    coronal=hu_np[:, yc_].astype(np.float16),
                    ideal_axial=ref_hu[zc_].cpu().numpy().astype(np.float16),
                    ideal_coronal=ref_hu[:, yc_].cpu().numpy().astype(np.float16),
                )
            (pdir / f"ct_{s.name}.json").write_text(json.dumps(meta, indent=1, default=str))
            log(
                {
                    "patient": pid,
                    "variant": s.name,
                    "fbp_s": t_fbp,
                    "bytes": w.bytes,
                    "t": time.time(),
                }
            )
            print(
                f"p{pid} {s.name}: fbp {t_fbp:.2f}s pre {t_pre:.2f}s write {t_write:.2f}s peak {ms.peak / MIB:.0f} MiB"
                + (f" body MAE {meta['metrics']['body']['mae_hu']:.1f}" if a.metrics else ""),
                flush=True,
            )
            del mu, hu
        del leap, p_clean, work, clean_by_sigma
        gc.collect()
        torch.cuda.empty_cache()
    print(f"p{pid} done in {time.perf_counter() - t_patient:.1f}s", flush=True)
