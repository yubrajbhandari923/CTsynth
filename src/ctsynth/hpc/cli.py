"""`ctsynth` command line: prepare | verify | export | import | report (stdlib argparse).

Exit codes: 0 ok; 1 validation failure or refusal (reasons on stderr, one per line);
2 usage error (argparse). No subcommand opens a network connection.
Every command is also available under `ctsynth experiment` (the runbook spelling).
"""

import argparse
import sys
from pathlib import Path

from ctsynth.hpc.bundle import HARDWARE_FIELDS, BundleError, export_result, prepare
from ctsynth.hpc.importer import ImportRefused, import_result, research_attach
from ctsynth.hpc.report import build_report, write_report
from ctsynth.hpc.verify import verify


def _add_commands(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    p = sub.add_parser("prepare", help="freeze a bundle from a committed config")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--require-output", action="append", default=[], metavar="FILE")
    p.add_argument("--data", action="append", default=[], type=Path, metavar="FILE")
    p.add_argument("--checkpoint", action="append", default=[], metavar="PATH[:external]")
    p.add_argument("--hw", action="append", default=[], metavar="KEY=VAL")
    p.add_argument("--env-setup", action="append", default=[], metavar="LINE")
    p.add_argument("--entry", nargs=argparse.REMAINDER, required=True, metavar="ARGV")

    v = sub.add_parser("verify", help="validate a bundle or result directory")
    v.add_argument("path", type=Path)
    v.add_argument("--repo", type=Path)
    v.add_argument("--data-root", type=Path)

    e = sub.add_parser("export", help="(worker) seal a finished job's output dir")
    e.add_argument("--bundle", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--exit-code", type=int, required=True)
    e.add_argument("--started", required=True)
    e.add_argument("--ended", required=True)
    e.add_argument("--slurm-job", default=None)

    i = sub.add_parser("import", help="verify and register a result with Research OS")
    i.add_argument("result", type=Path)
    i.add_argument("--repo", type=Path, required=True)
    i.add_argument("--runs-root", type=Path, required=True)
    i.add_argument("--research-bin", type=Path)
    i.add_argument("--research-root", type=Path)
    i.add_argument("--label")

    r = sub.add_parser("report", help="write a markdown summary of an imported run")
    r.add_argument("run_id")
    r.add_argument("--runs-root", type=Path, required=True)


def _parse_checkpoint(spec: str) -> dict[str, object]:
    if spec.endswith(":external"):
        return {"path": spec[: -len(":external")], "external": True}
    return {"path": spec, "external": False}


def _run(args: argparse.Namespace) -> int:
    if args.cmd == "prepare":
        hw: dict[str, str] = {}
        for item in args.hw:
            key, sep, val = item.partition("=")
            if not sep or key not in HARDWARE_FIELDS:
                print(f"bad --hw {item!r}; keys are {', '.join(HARDWARE_FIELDS)}", file=sys.stderr)
                return 1
            hw[key] = val
        bundle_dir = prepare(
            args.repo, args.config, args.out_root,
            entry_point=args.entry, required_outputs=args.require_output,
            data_files=args.data, checkpoints=[_parse_checkpoint(c) for c in args.checkpoint],
            hardware=hw, env_setup=args.env_setup,
        )  # fmt: skip
        print(bundle_dir)
        return 0
    if args.cmd == "verify":
        rep = verify(args.path, repo=args.repo, data_root=args.data_root)
        if rep.ok:
            print(f"OK {rep.kind} {args.path}")
            return 0
        print("\n".join(rep.problems), file=sys.stderr)
        return 1
    if args.cmd == "export":
        out = export_result(
            args.bundle, args.out, args.exit_code,
            started_utc=args.started, ended_utc=args.ended, slurm_job_id=args.slurm_job,
        )  # fmt: skip
        print(out)
        return 0
    if args.cmd == "import":
        research_bin = args.research_bin or args.repo / ".research" / "bin" / "research"
        result = import_result(
            args.result, repo=args.repo, runs_root=args.runs_root,
            attach=research_attach(research_bin, args.research_root), label=args.label,
        )  # fmt: skip
        print(f"imported {result.run_id} ({result.status}) -> {result.dest}")
        return 0
    if args.cmd == "report":
        path = write_report(args.run_id, args.runs_root)
        print(build_report(args.run_id, args.runs_root), end="")
        print(f"wrote {path}", file=sys.stderr)
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    """Subcommands (all paths explicit, nothing defaults to a cluster):
      prepare --repo R --config C --out-root D --entry ARGV... [--require-output F]...
              [--data F]... [--checkpoint PATH[:external]]... [--hw KEY=VAL]...
              [--env-setup LINE]...      -> prints the bundle dir
      verify PATH [--repo R] [--data-root D]
      export --bundle B --out O --exit-code N --started T --ended T [--slurm-job J]
      import RESULT --repo R --runs-root D [--research-bin P] [--research-root P] [--label L]
              -> prints run id and destination
      report RUN_ID --runs-root D        -> prints markdown, writes <RUN_ID>.report.md
    `--entry` is last and takes the remainder (argparse.REMAINDER)."""
    parser = argparse.ArgumentParser(prog="ctsynth", description="HPC experiment bundles")
    top = parser.add_subparsers(dest="cmd", required=True)
    _add_commands(top)
    group = top.add_parser("experiment", help="same commands, runbook spelling")
    _add_commands(group.add_subparsers(dest="cmd", required=True))
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        return _run(args)
    except (BundleError, ImportRefused, FileNotFoundError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
