"""Download the AAPM DL-Spectral CT dataset with md5 verification. For HPC; not run on the laptop.

uv run python scripts/fetch_aapm.py --dest "$SCRATCH/ctsynth_data/aapm"
  [--only Phantom_Adipose.npy.gz ...]
Hashes come from the Zenodo API at run time, so this never trusts a hash hard-coded here.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORD = "https://zenodo.org/api/records/14262737"
CHUNK = 1 << 20


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324  # Zenodo publishes md5 only; integrity check, not security
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="file names; default all")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(RECORD) as r:  # noqa: S310  # fixed https Zenodo URL
        files = json.load(r)["files"]
    wanted = [f for f in files if args.only is None or f["key"] in args.only]
    if args.only and len(wanted) != len(args.only):
        print(f"unknown file(s); available: {[f['key'] for f in files]}", file=sys.stderr)
        return 2

    bad = 0
    for f in wanted:
        want = f["checksum"].removeprefix("md5:")
        out = args.dest / f["key"]
        if out.exists() and _md5(out) == want:
            print(f"ok (cached)   {f['key']}")
            continue
        print(f"downloading   {f['key']}  ({f['size'] / 2**20:.0f} MiB)", flush=True)
        with urllib.request.urlopen(f["links"]["self"]) as src, out.open("wb") as dst:  # noqa: S310
            while chunk := src.read(CHUNK):
                dst.write(chunk)
        got = _md5(out)
        if got == want:
            print(f"ok (verified) {f['key']}")
        else:
            out.unlink()  # a corrupt file must never be left where an experiment could read it
            print(f"MD5 MISMATCH  {f['key']}: want {want} got {got} — deleted", file=sys.stderr)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
