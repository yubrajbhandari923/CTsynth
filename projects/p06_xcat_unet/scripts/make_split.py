#!/usr/bin/env python
"""Reproducible patient-level split (70/15/15) of the COPD patients in data_manifest.csv, with optional patients forced into TRAIN
(patients already used to develop/analyse the simulator must not be test patients).
usage: make_split.py MANIFEST OUT.json [--seed 0] [--force-train 22728 ...]"""

import argparse
import csv
import hashlib
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--force-train", nargs="*", default=[])
a = ap.parse_args()
ids = sorted(r["patient_id"] for r in csv.DictReader(open(a.manifest)))
rest = [i for i in ids if i not in a.force_train]
r = np.random.default_rng(np.random.SeedSequence(a.seed, spawn_key=(11,)))
perm = [rest[i] for i in r.permutation(len(rest))]
n_va = n_te = round(0.15 * len(ids))
split = {
    "seed": a.seed,
    "n_patients": len(ids),
    "forced_train": a.force_train,
    "test": sorted(perm[:n_te]),
    "val": sorted(perm[n_te : n_te + n_va]),
    "train": sorted(perm[n_te + n_va :] + list(a.force_train)),
}
assert len(set(split["train"]) | set(split["val"]) | set(split["test"])) == len(ids)
assert (
    not set(split["train"]) & set(split["val"])
    and not set(split["train"]) & set(split["test"])
    and not set(split["val"]) & set(split["test"])
)
split["sha256_of_sorted_ids"] = hashlib.sha256(
    json.dumps({k: split[k] for k in ("train", "val", "test")}, sort_keys=True).encode()
).hexdigest()
json.dump(split, open(a.out, "w"), indent=1)
print({k: (len(v) if isinstance(v, list) else v) for k, v in split.items()})
for p in ("22728", "21535"):
    print(p, [k for k in ("train", "val", "test") if p in split[k]])
