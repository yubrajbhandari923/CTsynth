#!/usr/bin/env python
"""Reproducible patient-level split (70/15/15) of the COPD patients in data_manifest.csv. usage: make_split.py MANIFEST OUT.json [SEED]"""

import csv
import json
import sys

import numpy as np

ids = sorted(r["patient_id"] for r in csv.DictReader(open(sys.argv[1])))
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
r = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(11,)))
perm = [ids[i] for i in r.permutation(len(ids))]
n_tr, n_va = round(0.7 * len(ids)), round(0.15 * len(ids))
split = {"seed": seed, "n_patients": len(ids), "train": sorted(perm[:n_tr]), "val": sorted(perm[n_tr : n_tr + n_va]), "test": sorted(perm[n_tr + n_va :])}
assert len(set(split["train"]) | set(split["val"]) | set(split["test"])) == len(ids)
json.dump(split, open(sys.argv[2], "w"), indent=1)
print({k: (len(v) if isinstance(v, list) else v) for k, v in split.items()})
for p in ("22728", "21535"):
    print(p, [k for k in ("train", "val", "test") if p in split[k]])
