"""Import a verified result directory on the laptop and register it with Research OS via
`research run attach` (Research OS stays the system of record; no second tracker)."""

import contextlib
import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ctsynth.hpc.bundle import RESULT_MANIFEST, resolve_entry, sha256_file
from ctsynth.hpc.verify import verify


class ImportRefused(Exception):
    """Import refused (validation failed, already imported, or target exists). Nothing was
    registered and nothing was left on disk."""


@dataclass(frozen=True)
class AttachRequest:
    experiment_id: str
    status: str  # "completed" | "failed"
    exit_code: int
    stdout: Path  # paths inside the imported copy under runs_imported/
    stderr: Path
    label: str
    command: str  # shlex-joined resolved entry point
    started: str
    ended: str
    slurm_job: str | None
    params: dict[str, str] = field(default_factory=dict)  # git_commit, bundle_id, config_sha256


Attach = Callable[[AttachRequest], str]  # returns the Research OS run id, e.g. "RUN-0004"


@dataclass(frozen=True)
class ImportResult:
    run_id: str
    dest: Path
    status: str
    fingerprint: str


def build_attach_argv(
    req: AttachRequest, *, research_bin: Path, root: Path | None = None
) -> list[str]:
    """Pure: [research_bin, --json, (--root R,) run, attach, EXP, --status S, --exit-code N,
    --stdout P, --stderr P, --label L, --started T, --ended T, (--slurm-job J,) (--param K=V)*,
    --, *shlex.split(command)]. `--json` is a GLOBAL flag and goes before `run`; the command
    after `--` is mandatory for the real CLI."""
    argv = [str(research_bin), "--json"]
    if root is not None:
        argv += ["--root", str(root)]
    argv += [
        "run", "attach", req.experiment_id,
        "--status", req.status,
        "--exit-code", str(req.exit_code),
        "--stdout", str(req.stdout),
        "--stderr", str(req.stderr),
        "--label", req.label,
        "--started", req.started,
        "--ended", req.ended,
    ]  # fmt: skip
    if req.slurm_job:
        argv += ["--slurm-job", req.slurm_job]
    for key, value in req.params.items():
        argv += ["--param", f"{key}={value}"]
    return [*argv, "--", *shlex.split(req.command)]


def research_attach(research_bin: Path, root: Path | None = None) -> Attach:
    """Return an Attach that shells out (subprocess list argv, no shell) to build_attach_argv
    and parses the run id from the JSON `id` field. Non-zero exit -> ImportRefused carrying the
    CLI's stderr (exit 3 policy blocks are not worked around)."""

    def attach(req: AttachRequest) -> str:
        argv = build_attach_argv(req, research_bin=research_bin, root=root)
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise ImportRefused(
                f"research run attach failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        try:
            run_id = str(json.loads(proc.stdout)["id"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ImportRefused(f"cannot read run id from research output: {exc}") from exc
        return run_id

    return attach


def import_result(
    result_dir: Path,
    *,
    repo: Path,
    runs_root: Path,
    attach: Attach,
    label: str | None = None,
) -> ImportResult:
    """verify(result_dir, repo=repo) -> refuse on any problem; fingerprint = sha256 of the
    RESULT.sha256 file; refuse if `<runs_root>/.imported/<fingerprint>.json` exists (idempotency
    marker). Then: copy to `<runs_root>/.staging-<fingerprint>/` skipping external checkpoints
    (they are never in the dir; their hash+location stay in result.json), call attach, rename
    staging -> `<runs_root>/<run_id>/` (refuse, remove staging, and keep the existing target
    untouched if it exists), write the marker (run_id, imported_utc, git_commit, bundle_id).
    Any failure after staging removes staging. A failed job (status "failed") IS importable and
    is registered with status failed + its exit code. Default label: `<bundle_id> <sha8>`
    plus ` FAILED` when failed. Never executes anything from result_dir."""
    result_dir, runs_root = Path(result_dir), Path(runs_root)
    rep = verify(result_dir, repo=repo)
    if not rep.ok:
        raise ImportRefused("verification failed: " + "; ".join(rep.problems))
    res, bundle = rep.result, rep.bundle
    fingerprint = sha256_file(result_dir / RESULT_MANIFEST)
    marker = runs_root / ".imported" / f"{fingerprint}.json"
    if marker.exists():
        prior = json.loads(marker.read_text(encoding="utf-8")).get("run_id", "?")
        raise ImportRefused(f"already imported as {prior} (marker {marker.name})")

    status = str(res["status"])
    bundle_id = str(res["bundle_id"])
    if label is None:
        label = f"{bundle_id} {str(res['git_commit'])[:8]}"
    if status == "failed" and "FAILED" not in label:
        label += " FAILED"
    staging = runs_root / f".staging-{fingerprint}"
    external = {ck["path"] for ck in res["checkpoints"] if ck["external"]}
    runs_root.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(
        result_dir,
        staging,
        ignore=lambda d, names: [n for n in names if Path(d) == result_dir and n in external],
    )
    try:
        command = shlex.join(resolve_entry(bundle, Path("config.yaml"), Path(".")))
        req = AttachRequest(
            experiment_id=str(res["experiment_id"]),
            status=status,
            exit_code=int(res["exit_code"]),
            stdout=staging / "stdout.log",
            stderr=staging / "stderr.log",
            label=label,
            command=command,
            started=str(res["started_utc"]),
            ended=str(res["ended_utc"]),
            slurm_job=str(res["slurm_job_id"]) if res["slurm_job_id"] else None,
            params={
                "git_commit": str(res["git_commit"]),
                "bundle_id": bundle_id,
                "config_sha256": str(res["config_sha256"]),
            },
        )
        run_id = attach(req)
        dest = runs_root / run_id
        if dest.exists():
            raise ImportRefused(f"destination {dest} already exists; never overwritten")
        os.rename(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    marker.parent.mkdir(parents=True, exist_ok=True)
    # shortcut: attach registers the staging paths (the run id is unknown until attach returns),
    # so the staging name becomes a relative symlink to the final dir to keep them resolvable.
    # Upgrade: have research return the run dir, or pre-allocate the id, then drop the symlink.
    with contextlib.suppress(OSError):
        os.symlink(dest.name, staging)
    marker.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "imported_utc": res["ended_utc"],
                "git_commit": res["git_commit"],
                "bundle_id": bundle_id,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ImportResult(run_id=run_id, dest=dest, status=status, fingerprint=fingerprint)
