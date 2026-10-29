#!/usr/bin/env python
"""Sanity checks on the generated cache before any training. usage: validate_cache.py CACHE PLAN.json OUT.json"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

cache, plan = Path(sys.argv[1]), json.load(open(sys.argv[2]))


def check(pid):
    pdir = cache / f"p{pid}"
    res = {"patient": pid, "problems": []}
    lab = np.load(pdir / "labels.npy", mmap_mode="r")
    res["labels_shape"] = list(lab.shape)
    cnt = np.bincount(np.asarray(lab).ravel(), minlength=15)
    res["classes_present"] = [int(i) for i in np.flatnonzero(cnt)]
    if len(res["classes_present"]) != 15:
        res["problems"].append(f"classes present {len(res['classes_present'])}/15")
    res["group_names"] = json.load(open(pdir / "group_names.json"))
    res["variants"] = {}
    for spec in plan["patients"][pid]["variants"]:
        n = spec["name"]
        mp = pdir / f"ct_{n}.json"
        if not mp.exists() or not (pdir / f"ct_{n}.npy").exists():
            res["problems"].append(f"missing {n}")
            continue
        meta = json.load(open(mp))
        ct = np.load(pdir / f"ct_{n}.npy", mmap_mode="r")
        hu = np.asarray(ct)[::3, ::3, ::3].astype(np.float32) * meta["hu_step"]
        body = np.asarray(lab)[::3, ::3, ::3] > 0
        v = {
            "shape_ok": list(ct.shape) == res["labels_shape"],
            "dtype": str(ct.dtype),
            "finite": bool(np.isfinite(hu).all()),
            "hu_min": float(hu.min()),
            "hu_max": float(hu.max()),
            "frac_at_int16_limit": float((np.abs(np.asarray(ct)[::3, ::3, ::3]) >= 32767).mean()),
            "body_mean_hu": float(hu[body].mean()),
            "air_mean_hu": float(hu[~body].mean()),
            "cond_len": len(meta["cond"]),
            "spec_matches": bool(
                abs((meta["n0"] or 0) - (spec["n0"] or 0)) < 1e-6 * max(1, spec["n0"] or 1)
                and meta["energy_keV"] == spec["acq"]["energy_kev"]
            ),
        }
        if (
            not (v["shape_ok"] and v["finite"] and v["spec_matches"])
            or v["frac_at_int16_limit"] > 0
        ):
            res["problems"].append(f"{n}: {v}")
        res["variants"][n] = v
    return res


with ProcessPoolExecutor(8) as ex:
    results = list(ex.map(check, sorted(plan["patients"])))
names = [json.dumps(r["group_names"], sort_keys=True) for r in results]
summary = {
    "patients": len(results),
    "volumes": sum(len(r["variants"]) for r in results),
    "patients_with_problems": [r["patient"] for r in results if r["problems"]],
    "group_names_identical_across_patients": len(set(names)) == 1,
    "group_names": json.loads(names[0]),
    "hu_min_overall": min(v["hu_min"] for r in results for v in r["variants"].values()),
    "hu_max_overall": max(v["hu_max"] for r in results for v in r["variants"].values()),
    "air_mean_hu_range": [
        min(v["air_mean_hu"] for r in results for v in r["variants"].values()),
        max(v["air_mean_hu"] for r in results for v in r["variants"].values()),
    ],
    "body_mean_hu_clean_range": [
        min(r["variants"]["clean"]["body_mean_hu"] for r in results),
        max(r["variants"]["clean"]["body_mean_hu"] for r in results),
    ],
}
json.dump({"summary": summary, "per_patient": results}, open(sys.argv[3], "w"), indent=1)
print(json.dumps(summary, indent=1))
for r in results:
    if r["problems"]:
        print("PROBLEM", r["patient"], r["problems"][:3])
