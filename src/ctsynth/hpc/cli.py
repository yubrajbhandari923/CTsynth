"""`ctsynth preflight | postcheck`: guards around `research hpc` (the native HPC protocol).

Exit codes: 0 ok; 1 refusal or failed check (reasons on stderr); 2 usage error.

    ctsynth preflight --repo . --config configs/x.yaml     # prints JSON fingerprints
    research experiment create … --param git_commit=… --param config_sha256=…   # record them
    research hpc submit EXP-… --yes -- uv run python …     # native
    research hpc sync
    ctsynth postcheck .research/runs/RUN-… --expect fp.json --require metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

from ctsynth.hpc.check import PreflightError, output_hashes, postcheck, preflight


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ctsynth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight", help="refuse a dirty/untracked source tree; print fingerprints")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--config", type=Path, required=True)
    q = sub.add_parser("postcheck", help="verify an imported run against preflight fingerprints")
    q.add_argument("run_dir", type=Path)
    q.add_argument("--expect", type=Path, required=True, help="JSON written from preflight output")
    q.add_argument("--require", action="append", default=[], metavar="FILE")
    q.add_argument("--failed", action="store_true", help="run is expected to have failed")
    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        try:
            print(json.dumps(preflight(a.repo, a.config), indent=2))
        except PreflightError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    problems = postcheck(a.run_dir, json.loads(a.expect.read_text()), a.require, failed=a.failed)
    for m in problems:
        print(m, file=sys.stderr)
    if not problems:
        print(json.dumps({"ok": True, "outputs_sha256": output_hashes(a.run_dir)}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
