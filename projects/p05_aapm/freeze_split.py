"""T-069: freeze the case-disjoint 800/100/100 split and audit leakage (duplicates, support).

Split = exact sizes via one seeded permutation (NOT the binomial per-case rule in aapm.splits).
Writes split_v1.json (case ids, seed, sizes, sha256 of id lists) to the run dir and, with --out-dir,
next to the dataset. Test ids are listed for the freeze record only; never used for tuning.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ctsynth.aapm.loader import CaseStore
from ctsynth.seeding import rng

SEED = 20261013
SIZES = {"train": 800, "val": 100, "test": 100}


def h(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args()
    run = Path(os.environ.get("RESEARCH_RUN_DIR", "."))
    st = CaseStore(a.data / "files")
    n = len(st)
    assert n == sum(SIZES.values()), n
    ar = st.arrays
    ph_h, img_h, sup_h, calc = [], [], [], []
    for i in range(n):
        p = np.stack(
            [
                np.asarray(ar[f"Phantom_{m}"][i])
                for m in ("Adipose", "Fibroglandular", "Calcification")
            ]
        )
        ph_h.append(h(p))
        img_h.append(h(np.asarray(ar["lowkVpImages"][i])) + h(np.asarray(ar["highkVpImages"][i])))
        sup_h.append(h(p.sum(0) > 1e-3))
        calc.append(float((p[2] > 1e-3).mean()))
    audit = {
        "n": n,
        "duplicate_phantoms": n - len(set(ph_h)),
        "duplicate_image_pairs": n - len(set(img_h)),
        "distinct_support_masks": len(set(sup_h)),
        "cases_with_calcification": int(sum(c > 0 for c in calc)),
    }
    perm = rng(SEED).permutation(n)
    ids, lo = {}, 0
    for k, sz in SIZES.items():
        ids[k] = sorted(int(x) for x in perm[lo : lo + sz])
        lo += sz
    assert not (
        set(ids["train"]) & set(ids["val"])
        | set(ids["train"]) & set(ids["test"])
        | set(ids["val"]) & set(ids["test"])
    )
    audit["calc_cases_per_split"] = {k: int(sum(calc[i] > 0 for i in v)) for k, v in ids.items()}
    out = {
        "version": "v1",
        "seed": SEED,
        "method": "ctsynth.seeding.rng(seed).permutation(1000); sizes 800/100/100; ids sorted",
        "note": "internal split of the public 1000-case release; NOT the official challenge split",
        "sizes": SIZES,
        **ids,
        "sha256": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in ids.items()},
        "audit": audit,
    }
    out["manifest_sha256"] = hashlib.sha256(
        json.dumps({k: out[k] for k in ("seed", *SIZES)}, sort_keys=True).encode()
    ).hexdigest()
    (run / "split_v1.json").write_text(json.dumps(out, indent=1))
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "split_v1.json").write_text(json.dumps(out, indent=1))
    with open(run / "artifacts.jsonl", "a") as f:
        f.write(
            json.dumps({"path": str((run / "split_v1.json").resolve()), "name": "split_v1.json"})
            + "\n"
        )
    with open(run / "metrics.jsonl", "a") as f:
        for k, v in audit.items():
            if isinstance(v, int):
                f.write(json.dumps({"name": k, "value": v}) + "\n")
    print(json.dumps(out["audit"], indent=1), out["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
