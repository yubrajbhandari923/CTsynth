"""`ctsynth` guards around `research hpc` (the native HPC protocol) and around run outputs.

    preflight / postcheck   source-clean + identity fingerprints before/after a cluster run
    bundle-check            what `research hpc bundle` would package (F-038)
    paired-check DIR...     paired-ct records load and every sha256 verifies
    xcist-check             optional XCIST stack usable (simulate / FDK recon)

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

from ctsynth.hpc.check import PreflightError, bundle_check, output_hashes, postcheck, preflight
from ctsynth.paired.schema import load_record
from ctsynth.paired.xcist import environment_report


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
    b = sub.add_parser("bundle-check", help="refuse a bundle that would ship envs, data or results")
    b.add_argument("--repo", type=Path, default=Path("."))
    b.add_argument("--max-mb", type=float, default=50.0)
    c = sub.add_parser("paired-check", help="verify paired-ct record directories")
    c.add_argument("records", type=Path, nargs="+")
    sub.add_parser("xcist-check", help="report whether the optional XCIST stack is usable")
    a = ap.parse_args(argv)
    if a.cmd == "bundle-check":
        rep = bundle_check(a.repo, a.max_mb)
        print(json.dumps({k: v for k, v in rep.items() if k != "largest"}, indent=2))
        for m in rep["problems"]:
            print(m, file=sys.stderr)
        return 1 if rep["problems"] else 0
    if a.cmd == "paired-check":
        failed = 0
        for d in a.records:
            try:
                rec = load_record(d)  # verifies every array sha256
                print(f"ok  {d}  {rec.phantom_id}/{rec.acquisition_id}")
            except Exception as exc:  # any load/verify failure is a failed check, with its reason
                failed += 1
                print(f"FAIL {d}: {exc}", file=sys.stderr)
        return 1 if failed else 0
    if a.cmd == "xcist-check":
        rep = environment_report()
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok_simulate"] and rep["ok_recon"] else 1
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
