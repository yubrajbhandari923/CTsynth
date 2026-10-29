#!/usr/bin/env python
"""Scale the one-case measurements (sim_report, unet_probe) to all patients in data_manifest.csv: time, storage, GPU-hours.

First-order scaling per patient i: s_i = (views_i / 11000) * sqrt(voxels_i / 50.86e6) relative to copd_22728 at 0.8 mm. This is
an ESTIMATE; forward ~ views (measured 3.0x time for 3x views), FBP ~ views x voxels^0.5 (measured 4.2x for 3x views, 2.7x voxels).
usage: estimate_dataset_cost.py MANIFEST SIM_REPORT UNET_PROBE OUT_JSON
"""

import csv
import json
import math
import sys

man = list(csv.DictReader(open(sys.argv[1])))
S = json.load(open(sys.argv[2]))["steps"]
cache = {(c["tier"], c["fmt"]): c for c in json.load(open(sys.argv[2]))["cache"]}
REF_VIEWS, REF_VOX = 11000, 341 * 452 * 330
t = {k: S[k]["median_s"] for k in S}
cold_nfs_MBps = 270.0  # measured O_DIRECT read of /home on node004 (node probe job 203521)
rows = []
for r in man:
    z, y, x = (int(v) for v in r["grid_0p8mm_zyx"].split("x"))
    vox, views = z * y * x, int(r["views_margin40"])
    s = (views / REF_VIEWS) * math.sqrt(vox / REF_VOX)
    sv = views / REF_VIEWS
    rows.append(
        dict(
            pid=r["patient_id"],
            s=s,
            sv=sv,
            vox=vox,
            views=views,
            bin=int(r["bytes"]),
            proj=views * 64 * 900 * 4,
            ct16=vox * 2,
            lab=vox,
            att=vox * 4,
        )
    )
tot = lambda k: sum(r[k] for r in rows)  # noqa: E731
GB = 1e9
out = {
    "n_patients": len(rows),
    "mean_scale": tot("s") / len(rows),
    "max_scale": max(r["s"] for r in rows),
}
shared = lambda r: (
    r["sv"] * (t["04_resample_to_grid"] * 0 + t["06_forward_project"])
    + r["s"] * t["04_resample_to_grid"]
    + r["bin"] / (cold_nfs_MBps * 1e6)
)  # noqa: E731
pervar = lambda r: (
    r["s"] * (t["07_poisson_noise"] + t["08a_helical_fbp_noisy"] + t["09b_device_to_host_copy"])
    + r["ct16"] / (cache[("nvme", "npy_i16q")]["write_MB_s"] * 1e6)
)  # noqa: E731
out["per_patient_22728_s"] = {
    "shared": t["04_resample_to_grid"] + t["06_forward_project"],
    "per_noise_variant": t["07_poisson_noise"]
    + t["08a_helical_fbp_noisy"]
    + t["09b_device_to_host_copy"]
    + 0.1,
}
out["scenarios"] = {}
for v in (1, 3, 5, 10):
    t_noise_only = sum(
        shared(r) + v * pervar(r) for r in rows
    )  # variants differ only in photon statistics: one forward projection each
    t_all_geom = sum(
        v * (shared(r) + pervar(r)) for r in rows
    )  # every variant needs its own projection (acquisition varies)
    out["scenarios"][v] = {
        "gpu_hours_noise_only_shared_projection": t_noise_only / 3600,
        "gpu_hours_each_variant_own_projection": t_all_geom / 3600,
        "wall_hours_8_gpus_ideal_noise_only": t_noise_only / 3600 / 8,
        "wall_hours_8_gpus_ideal_own_projection": t_all_geom / 3600 / 8,
        "ct_int16_GB": v * tot("ct16") / GB,
        "ct_float32_GB_if_not_quantised": 2 * v * tot("ct16") / GB,
        "labels_GB": tot("lab") / GB,
        "ct_niigz_GB_if_written_as_compressed_nifti": v * 0.875 * 2 * tot("ct16") / GB * 0
        + v * tot("ct16") / 101727248 * 178150929 / GB,
        "niigz_write_hours_if_used": v
        * len(rows)
        * 15.6
        * (sum(r["s"] for r in rows) / len(rows))
        / 3600,
    }
out["storage_GB"] = {
    "native_phantoms": tot("bin") / GB,
    "attenuation_f32_all_patients_transient": tot("att") / GB,
    "clean_projections_f32_all_patients_if_kept": tot("proj") / GB,
    "max_single_patient_projections_GB": max(r["proj"] for r in rows) / GB,
}
out["tiers_free_GB"] = {
    "/dev/shm (RAM-backed, counts against job memory)": 252,
    "/tmp NVMe (shared, node004)": 1500,
    "/home NFS (shared)": 6300,
}
P = json.load(open(sys.argv[3]))["configs"]
b96 = [c for c in P if c["patch"] == 96 and c["batch"] == 4][0]
out["unet"] = {
    "step_s_96_bs4": b96["step_median_s"],
    "Mvox_per_s": b96["voxels_per_s"] / 1e6,
    "peak_allocated_MiB_96_bs4": b96["peak_allocated_MiB"],
}
for v in (1, 3, 5, 10):
    n_train_vol = round(0.7 * len(rows)) * v
    vox_epoch = n_train_vol * tot("vox") / len(rows)
    out["unet"][f"epoch_hours_full_coverage_{v}_variants"] = (
        vox_epoch / (b96["voxels_per_s"]) / 3600
    )
json.dump(out, open(sys.argv[4], "w"), indent=1)
print(json.dumps(out, indent=1))
