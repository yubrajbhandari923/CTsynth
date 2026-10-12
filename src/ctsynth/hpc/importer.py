"""Import a verified result directory on the laptop and register it with Research OS via
`research run attach` (Research OS stays the system of record; no second tracker)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


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
    raise NotImplementedError


def research_attach(research_bin: Path, root: Path | None = None) -> Attach:
    """Return an Attach that shells out (subprocess list argv, no shell) to build_attach_argv
    and parses the run id from the JSON `id` field. Non-zero exit -> ImportRefused carrying the
    CLI's stderr (exit 3 policy blocks are not worked around)."""
    raise NotImplementedError


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
    raise NotImplementedError
