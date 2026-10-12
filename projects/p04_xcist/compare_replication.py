"""Compare an imported HPC replication against the laptop record of the same cell.
`uv run python projects/p04_xcist/compare_replication.py RUN_DIR DATASET_DIR`"""

import json
import sys
from pathlib import Path

import numpy as np

run, ds = Path(sys.argv[1]), Path(sys.argv[2])
out = run / "collected" / "outputs"
rec = ds / "G1__kvp120_ref"
meta = json.loads((out / "replication.json").read_text())
p_h, p_l = np.load(out / "prep.npy"), np.load(rec / "prep.npy")
r_h, r_l = np.load(out / "recon.npy"), np.load(rec / "recon.npy")
local_truth = np.load(rec / "truth.npy")
import hashlib  # noqa: E402

res = {
    "truth_sha256_equal": hashlib.sha256(local_truth.tobytes()).hexdigest() == meta["truth_sha256"],
    "prep_bit_identical": bool(np.array_equal(p_h, p_l)),
    "prep_max_abs_diff": float(np.abs(p_h - p_l).max()),
    "prep_rms_diff": float(np.sqrt(np.mean((p_h - p_l) ** 2))),
    "prep_max": float(p_l.max()),
    "recon_max_abs_diff_HU": float(np.abs(r_h - r_l).max()),
    "recon_rms_diff_HU": float(np.sqrt(np.mean((r_h - r_l) ** 2))),
    "runtime_s_hpc": meta["runtime_s"],
    "host": meta["host"],
    "gecatsim": meta["provenance"]["gecatsim"],
    "numpy_hpc": meta["provenance"]["numpy"],
}
print(json.dumps(res, indent=1))
(run / "replication_comparison.json").write_text(json.dumps(res, indent=1))
