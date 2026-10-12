"""HPC replication of one dataset-v0 cell (G1, 120 kVp, reference) with XCIST.

Runs inside a SLURM job via `bash scripts/hpc_xcist_replicate.sh`; writes prep/recon/metadata
into $RESEARCH_RUN_DIR. The laptop then compares it with its own record
(projects/p04_xcist/compare_replication.py)."""

import hashlib
import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from generate import BASE, DX, PHANTOMS, XN, N  # type: ignore[import-not-found]

from ctsynth.paired.phantoms import fractions
from ctsynth.paired.xcist import run_xcist

run_dir = Path(os.environ["RESEARCH_RUN_DIR"])
f = fractions(PHANTOMS["G1"], N, DX)
acq = replace(BASE, kvp=120)
t0 = time.time()
out = run_xcist(run_dir / "_work", f, XN, DX, acq, recon=True)
rt = time.time() - t0
np.save(run_dir / "prep.npy", out["prep"])
np.save(run_dir / "recon.npy", out["recon"])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
meta = {
    "host": platform.node(),
    "platform": platform.platform(),
    "runtime_s": round(rt, 2),
    "provenance": out["provenance"],
    "truth_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
    "prep_sha256": sha(run_dir / "prep.npy"),
    "recon_sha256": sha(run_dir / "recon.npy"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
(run_dir / "replication.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
with open(run_dir / "artifacts.jsonl", "a") as a:
    for n in ("replication.json", "prep.npy", "recon.npy"):
        a.write(json.dumps({"path": str((run_dir / n).resolve()), "name": n}) + "\n")
with open(run_dir / "metrics.jsonl", "a") as m:
    m.write(json.dumps({"name": "xcist_runtime_s", "value": rt}) + "\n")
print(json.dumps(meta, indent=1))
