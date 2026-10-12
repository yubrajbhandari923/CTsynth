"""Run XCIST's shipped Sim_Recon_Sample.py UNMODIFIED in a scratch copy; record versions, hashes, runtime.

Run inside the optional XCIST env (.venv-xcist): `research run exec EXP-… -- .venv-xcist/bin/python <this>`.
Deviation handling: on macOS the shipped FDK library links /opt/homebrew gcc libgomp; if XCIST_LIBGOMP_DIR
is given we set DYLD_FALLBACK_LIBRARY_PATH for the child only (no system change) and record it.
(A DYLD_* variable cannot be passed through /bin/sh launchers: SIP strips it, hence the other name.)
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("XCIST_OUT", "projects/p04_xcist/results/official_example"))
OUT.mkdir(parents=True, exist_ok=True)
import gecatsim  # noqa: E402

pkg = Path(gecatsim.__file__).parent
work = Path(tempfile.mkdtemp(prefix="xcist_official_"))
shutil.copytree(pkg / "examples", work / "examples")
env = {**os.environ, "MPLBACKEND": "Agg"}
deviation = None
if os.environ.get("XCIST_LIBGOMP_DIR"):
    env["DYLD_FALLBACK_LIBRARY_PATH"] = os.environ["XCIST_LIBGOMP_DIR"]
    deviation = (
        f"DYLD_FALLBACK_LIBRARY_PATH={os.environ['XCIST_LIBGOMP_DIR']} (libgomp.1.dylib symlink)"
    )
t0 = time.time()
p = subprocess.run(
    [sys.executable, "Sim_Recon_Sample.py"],
    cwd=work / "examples",
    env=env,
    capture_output=True,
    text=True,
)
runtime = time.time() - t0
(OUT / "stdout.log").write_text(p.stdout)
(OUT / "stderr_tail.log").write_text(p.stderr[-4000:])
ex = work / "examples"
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()  # noqa: E731
img_path = ex / "test_512x512x4.raw"
meta = {
    "gecatsim": metadata.version("gecatsim"),
    "numpy": np.__version__,
    "python": sys.version.split()[0],
    "exit_code": p.returncode,
    "runtime_s": round(runtime, 1),
    "deviation": deviation,
    "script_sha256": sha(ex / "Sim_Recon_Sample.py"),
    "cfg_sha256": {f.name: sha(f) for f in sorted((ex / "cfg").glob("*.cfg"))},
}
if img_path.exists():
    img = np.fromfile(img_path, dtype="<f4").reshape(4, 512, 512)
    np.save(OUT / "recon.npy", img)
    meta |= {
        "recon_sha256": sha(img_path),
        "recon_min_HU": float(img.min()),
        "recon_max_HU": float(img.max()),
        "recon_median_HU_slice2": float(np.median(img[2])),
    }
    for ext in ("air", "offset", "scan", "prep"):
        meta[f"{ext}_sha256"] = sha(ex / f"test.{ext}")
        meta[f"{ext}_bytes"] = (ex / f"test.{ext}").stat().st_size
(OUT / "official_example.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
print(json.dumps(meta, indent=2))
sys.exit(p.returncode)
