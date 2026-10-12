"""Bundle creation (laptop), SLURM rendering, result export (worker). See docs/hpc-bundle-spec.md.

Nothing here touches the network. `export_result` runs on the worker and must not need git,
Research OS, or write anywhere except its `out_dir`.
"""

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
UNSET = "unset"  # documented default of every SLURM/hardware field
BUNDLE_MANIFEST = "MANIFEST.sha256"  # covers every file in a bundle dir except itself
RESULT_MANIFEST = "RESULT.sha256"  # same, for a result dir
HARDWARE_FIELDS = (
    "partition", "account", "gpus", "cpus", "mem", "time", "job_name", "scratch",
)  # fmt: skip
# bundle.json top-level keys (see spec for types)
BUNDLE_KEYS = (
    "schema_version", "bundle_id", "experiment_id", "created_utc", "git_commit",
    "git_dirty_source", "config", "config_sha256", "seeds", "data", "uv_lock_sha256",
    "hardware", "env_setup", "entry_point", "required_outputs", "checkpoints",
)  # fmt: skip
# result.json top-level keys
RESULT_KEYS = (
    "schema_version", "bundle_id", "experiment_id", "git_commit", "config_sha256", "status",
    "exit_code", "started_utc", "ended_utc", "slurm_job_id", "outputs", "checkpoints",
)  # fmt: skip


class BundleError(Exception):
    """prepare/export refused. The message names the precise reason."""


def sha256_file(path: Path) -> str:
    # Same pattern as projects/p02_aapm_benchmark/evaluate.py (src/ cannot import projects/).
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(directory: Path, name: str) -> Path:
    """Write `<directory>/<name>`: sorted `<sha256>  <posix relpath>` lines (sha256sum format)
    for every regular file under `directory` except the manifest itself."""
    raise NotImplementedError


def check_manifest(directory: Path, name: str) -> list[str]:
    """Return problems (empty = ok): manifest missing/malformed, listed file missing, hash
    mismatch, file present but unlisted, absolute/`..`/symlink paths. Pure read; never raises
    on bad content."""
    raise NotImplementedError


def prepare(
    repo: Path,
    config_path: Path,
    out_root: Path,
    *,
    entry_point: Sequence[str],
    required_outputs: Sequence[str],
    data_files: Sequence[Path] = (),
    checkpoints: Sequence[Mapping[str, Any]] = (),
    hardware: Mapping[str, str] | None = None,
    env_setup: Sequence[str] = (),
    now: datetime | None = None,
) -> Path:
    """Create `<out_root>/<EXP>-<sha8>-<YYYYmmddTHHMMSSZ>/` and return it.

    Contents: bundle.json, config.yaml (byte copy of config_path), run.sbatch, MANIFEST.sha256.
    entry_point is argv with `{config}` / `{out}` placeholders (see resolve_entry); its first
    element is a bare command such as "python"; the path is relative to the source root.
    checkpoints: [{"path": str, "external": bool}] declared in advance (sizes/hashes are
    unknown until the run; they are measured by export_result).
    hardware: any subset of HARDWARE_FIELDS; missing ones become UNSET.
    Raises BundleError (and creates NOTHING on disk) if: git_dirty_source is true (same
    pathspec as runmeta: src tests configs, untracked files included); repo has no uv.lock;
    config fails ctsynth.config.load_config; experiment_id empty; the target directory already
    exists (never overwrite; `now` is injectable for tests); a data file is missing.
    Writes the bundle into a sibling temp dir and renames it into place (atomic).
    """
    raise NotImplementedError


def resolve_entry(bundle: Mapping[str, Any], config_path: Path, out_dir: Path) -> list[str]:
    """bundle['entry_point'] with `{config}` -> str(config_path), `{out}` -> str(out_dir)."""
    raise NotImplementedError


def render_sbatch(bundle: Mapping[str, Any]) -> str:
    """Render templates/slurm.sh.j2 (str.format, no new deps) from a bundle dict.

    Each hardware field set to a real value -> `#SBATCH --<flag>=<value>` (job_name ->
    --job-name; scratch -> `SCRATCH_DIR=<value>` line, not an #SBATCH). A field equal to UNSET
    -> `# TODO(unset): <flag>` and NO active directive / assignment. env_setup lines are
    injected verbatim; empty -> a `# TODO(unset): no environment setup` comment. The entry
    point is shlex-quoted with `{config}` -> "$BUNDLE_DIR/config.yaml", `{out}` -> "$OUT".
    The output must contain no cluster-specific literal that did not come from `bundle`.
    """
    raise NotImplementedError


def export_result(
    bundle_dir: Path,
    out_dir: Path,
    exit_code: int,
    *,
    started_utc: str,
    ended_utc: str,
    slurm_job_id: str | None = None,
) -> Path:
    """Worker-side: turn the job's `out_dir` into a self-describing result directory.

    out_dir already holds the job outputs plus stdout.log / stderr.log. This copies
    bundle.json + config.yaml in, scans every other file into result.json `outputs`
    ({path, bytes, sha256}), measures each declared checkpoint into `checkpoints`
    ({path, bytes, sha256, external, location}; external ones are hashed in place and NOT copied;
    a declared checkpoint that does not exist is omitted), sets status = "completed" iff
    exit_code == 0 else "failed", and writes RESULT.sha256. Returns out_dir.
    Refuses (BundleError) if out_dir already contains result.json (never overwrite a completed
    run). Does not read git, Research OS, or anything outside bundle_dir/out_dir/checkpoints.
    """
    raise NotImplementedError
