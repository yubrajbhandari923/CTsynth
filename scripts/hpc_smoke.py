"""HPC smoke test: stdlib-only, Python 3.8-compatible. Not scientific; checks the handoff, not readiness for GPU/scale."""
import hashlib
import json
import os
import platform
import socket
import sys

run_dir = os.environ["RESEARCH_RUN_DIR"]
n = 1_000_000
ssq = sum(i * i for i in range(n))  # exact integer: (n-1)n(2n-1)/6
assert ssq == (n - 1) * n * (2 * n - 1) // 6
h = hashlib.sha256(b"ctsynth-smoke" * 1000).hexdigest()
res = {
    "python": sys.version.split()[0], "platform": platform.platform(), "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "run_id": os.environ.get("RESEARCH_RUN_ID"),
    "sum_squares": ssq, "sha256": h,
    "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
}
out = os.path.join(run_dir, "smoke_result.json")
with open(out, "w") as f:
    json.dump(res, f, indent=2, sort_keys=True)
with open(os.path.join(run_dir, "metrics.jsonl"), "a") as f:
    f.write(json.dumps({"name": "sum_squares_exact_ok", "value": 1.0, "step": 0}) + "\n")
with open(os.path.join(run_dir, "artifacts.jsonl"), "a") as f:
    f.write(json.dumps({"path": out, "name": "smoke_result.json", "type": "json", "tags": ["smoke"]}) + "\n")
print(json.dumps(res, indent=2))
