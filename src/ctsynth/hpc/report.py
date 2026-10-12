"""Readable summary of an imported run."""

import json
from pathlib import Path
from typing import Any


def _numeric_leaves(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key in obj:
            out.update(_numeric_leaves(obj[key], f"{prefix}{key}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = obj
    return out


def build_report(run_id: str, runs_root: Path) -> str:
    """Markdown for `<runs_root>/<run_id>/`: experiment, FULL git commit (the commit the run was
    made from, not the repo HEAD), config_sha256, status (FAILED stated loudly with exit code),
    started/ended, slurm job, a metrics table (numeric leaves of metrics.json, nested keys
    dotted, sorted), figure paths (*.png/*.pdf/*.svg), checkpoint refs (path, bytes, sha256,
    external flag, location). Raises FileNotFoundError for an unknown run id."""
    run = Path(runs_root) / run_id
    result_file = run / "result.json"
    if not result_file.is_file():
        raise FileNotFoundError(f"no imported run {run_id} under {runs_root}")
    res = json.loads(result_file.read_text(encoding="utf-8"))
    bundle = json.loads((run / "bundle.json").read_text(encoding="utf-8"))
    lines = [f"# {run_id}: {res['experiment_id']}", ""]
    if res["status"] == "failed":
        lines.append(f"**STATUS: FAILED (exit code {res['exit_code']})**")
    else:
        lines.append(f"Status: {res['status']} (exit code {res['exit_code']})")
    lines += [
        "",
        f"- Git commit (run source): `{res['git_commit']}`",
        f"- Config sha256: `{res['config_sha256']}`",
        f"- Bundle: `{res['bundle_id']}`",
        f"- Started: {res['started_utc']}",
        f"- Ended: {res['ended_utc']}",
        f"- SLURM job: {res['slurm_job_id'] or 'none'}",
        "",
    ]
    metrics_file = run / "metrics.json"
    lines += ["## Metrics", ""]
    if metrics_file.is_file():
        leaves = _numeric_leaves(json.loads(metrics_file.read_text(encoding="utf-8")))
        lines += ["| metric | value |", "| --- | --- |"]
        lines += [f"| {k} | {leaves[k]} |" for k in sorted(leaves)]
    else:
        lines.append("No metrics.json in this run.")
    figures = sorted(
        p.relative_to(run).as_posix()
        for p in run.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".pdf", ".svg")
    )
    lines += ["", "## Figures", ""]
    lines += [f"- {f}" for f in figures] or ["None."]
    lines += ["", "## Checkpoints", ""]
    if res["checkpoints"]:
        lines += [
            "| path | bytes | sha256 | external | location |",
            "| --- | --- | --- | --- | --- |",
        ]
        for ck in res["checkpoints"]:
            ext = "yes" if ck["external"] else "no"
            lines.append(
                f"| {ck['path']} | {ck['bytes']} | `{ck['sha256']}` | {ext} | {ck['location']} |"
            )
    else:
        lines.append("None.")
    declared = [c["path"] for c in bundle.get("checkpoints", [])]
    if declared:
        lines += ["", f"Declared checkpoints: {', '.join(declared)}"]
    return "\n".join(lines) + "\n"


def write_report(run_id: str, runs_root: Path) -> Path:
    """Write build_report to `<runs_root>/<run_id>.report.md` (outside the run dir, which stays
    byte-identical to what was verified). Overwriting an older report of the same run is fine."""
    path = Path(runs_root) / f"{run_id}.report.md"
    path.write_text(build_report(run_id, Path(runs_root)), encoding="utf-8")
    return path
