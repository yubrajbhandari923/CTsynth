"""T-017: fetch (md5-verified), decompress, schema-validate and manifest the AAPM 1000-case dataset.

Idempotent: archives that verify are not re-downloaded; a decompressed .npy whose recorded sha256
matches is not rewritten. Raw data live under --dest (cluster shared-data), never in the run dir.
Usage (in a job): acquire.py --dest <shared-data>/aapm-dl-spectral-ct-zenodo-14262737
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ctsynth.aapm import schema as S
from ctsynth.aapm.loader import CaseStore

REPO = Path(__file__).resolve().parents[2]
CHUNK = 1 << 24
STEMS = [
    "Phantom_Adipose", "Phantom_Fibroglandular", "Phantom_Calcification",
    "lowkVpImages", "highkVpImages", "lowkVpTransmission", "highkVpTransmission",
]  # fmt: skip


def gunzip(src: Path, dst: Path) -> str:
    """Stream-decompress to dst (via .part, atomic rename); return decompressed sha256."""
    h, part = hashlib.sha256(), dst.with_suffix(".part")
    with gzip.open(src, "rb") as fi, part.open("wb") as fo:
        while b := fi.read(CHUNK):
            h.update(b)
            fo.write(b)
    part.rename(dst)
    return h.hexdigest()


def case_hashes(arr: np.ndarray) -> list[str]:
    return [
        hashlib.sha1(np.ascontiguousarray(arr[i]).tobytes()).hexdigest() for i in range(len(arr))
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--chunk-cases", type=int, default=25)
    ap.add_argument("--skip-fetch", action="store_true", help="archives already in place (tests)")
    a = ap.parse_args()
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR", "."))
    arch, files = a.dest / "archives", a.dest / "files"
    arch.mkdir(parents=True, exist_ok=True)
    files.mkdir(exist_ok=True)
    t0 = time.time()

    fetch = [sys.executable, str(REPO / "scripts/fetch_aapm.py"), "--dest", str(arch)]
    names = [f"{s}.npy.gz" for s in STEMS] + ["starting_kit.tgz"]
    if not a.skip_fetch:  # one fetch process per file: downloads are network-bound and independent
        with ThreadPoolExecutor(len(names)) as ex:
            rcs = list(ex.map(lambda n: subprocess.run([*fetch, "--only", n]).returncode, names))
        if any(rcs):
            print("fetch failed; nothing was decompressed", file=sys.stderr)
            return 1

    prev = {}
    if (a.dest / "manifest.json").exists():
        prev = {
            f["name"]: f["sha256"]
            for f in json.loads((a.dest / "manifest.json").read_text())["files"]
        }

    def decompress(stem: str) -> str:
        dst = files / f"{stem}.npy"
        if dst.exists() and stem in prev:
            return prev[stem]
        sha = gunzip(arch / f"{stem}.npy.gz", dst)  # zlib releases the GIL: threads parallelise
        print("decompressed", stem, flush=True)
        return sha

    with ThreadPoolExecutor(len(STEMS)) as ex:
        sha = dict(zip(STEMS, ex.map(decompress, STEMS), strict=True))

    store = CaseStore(files)  # headers: shape/dtype/case counts
    n = len(store)
    stats = {
        "n_cases": n,
        "sum_max": 0.0,
        "T50_mean": 0.0,
        "T80_mean": 0.0,
        "calc_support_frac": 0.0,
    }
    tr = {"T_min": 1.0, "T_max": 0.0}
    part = {"air": 0, "tissue": 0, "band": 0}
    for lo in range(0, n, a.chunk_cases):
        sl = slice(lo, min(n, lo + a.chunk_cases))
        ar = store.arrays
        ph = np.stack(
            [
                np.asarray(ar[f"Phantom_{m}"][sl])
                for m in ("Adipose", "Fibroglandular", "Calcification")
            ],
            1,
        )
        tx = {
            50: np.asarray(ar["lowkVpTransmission"][sl]),
            80: np.asarray(ar["highkVpTransmission"][sl]),
        }
        im = {50: np.asarray(ar["lowkVpImages"][sl]), 80: np.asarray(ar["highkVpImages"][sl])}
        S.validate_dataset(ph, tx, im, S.REAL_GEOMETRY)  # raises SchemaError on any violation
        w = ph.shape[0]
        tot = ph.sum(1, dtype=np.float64)
        stats["sum_max"] = max(stats["sum_max"], float(tot.max()))
        stats["T50_mean"] += float(tx[50].mean()) * w / n
        stats["T80_mean"] += float(tx[80].mean()) * w / n
        stats["calc_support_frac"] += float((ph[:, 2] > S.CALC_SUPPORT_THRESH).mean()) * w / n
        tr["T_min"] = min(tr["T_min"], float(min(tx[50].min(), tx[80].min())))
        tr["T_max"] = max(tr["T_max"], float(max(tx[50].max(), tx[80].max())))
        part["air"] += int((tot < 1e-3).sum())
        part["tissue"] += int((np.abs(tot - 1) < 1e-3).sum())
        part["band"] += int(tot.size - (tot < 1e-3).sum() - (np.abs(tot - 1) < 1e-3).sum())
        print("validated cases", sl.start, sl.stop, flush=True)
    npx = n * S.REAL_GEOMETRY.n_pix**2
    part = {k: v / npx for k, v in part.items()}

    # Duplicate cases, and overlap with the 10-case starting kit (split leakage).
    h_img = case_hashes(store.arrays["lowkVpImages"])
    dup = len(h_img) - len(set(h_img))
    kit_overlap: list[int] = []
    with tarfile.open(arch / "starting_kit.tgz") as tf:
        member = next(m for m in tf.getmembers() if Path(m.name).name == "lowkVpImages.npy")
        kit = np.load(io.BytesIO(tf.extractfile(member).read()))  # type: ignore[union-attr]
    kit_h = set(case_hashes(kit))
    kit_overlap = [i for i, h in enumerate(h_img) if h in kit_h]

    manifest = {
        "schema": 1,
        "name": "AAPM DL-Spectral CT Grand Challenge training set (1000 cases)",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "https://doi.org/10.5281/zenodo.14262737",
        "license": "CC-BY-4.0 (attribution required: Sidky & Pan, Med Phys 2023)",
        "layout": "files/<stem>.npy float32; archives/*.npy.gz kept for provenance",
        "files": [
            {"name": s, "sha256": sha[s], "bytes": (files / f"{s}.npy").stat().st_size}
            for s in STEMS
        ],
        "validation": {
            "schema": "ctsynth.aapm.schema.validate_dataset",
            "passed": True,
            **stats,
            **tr,
            "partition_frac": part,
            "duplicate_cases": dup,
            "kit_overlap_indices": kit_overlap,
        },
    }
    (a.dest / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    report = {
        **manifest["validation"],
        "runtime_s": round(time.time() - t0, 1),
        "dest": str(a.dest),
    }
    (run_dir / "acquisition_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    with open(run_dir / "artifacts.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "path": str((run_dir / "acquisition_report.json").resolve()),
                    "name": "acquisition_report.json",
                }
            )
            + "\n"
        )
    with open(run_dir / "metrics.jsonl", "a") as f:
        for k in ("n_cases", "duplicate_cases", "sum_max", "T_min", "T_max"):
            f.write(json.dumps({"name": k, "value": report[k]}) + "\n")
        f.write(json.dumps({"name": "kit_overlap_count", "value": len(kit_overlap)}) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
