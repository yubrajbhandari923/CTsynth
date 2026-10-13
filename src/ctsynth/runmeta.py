"""Machine-readable run metadata, written beside every run's outputs.

Complements `research run exec` (laptop). HPC runs bypass it, so each run dir carries its own
record that `research sync` / manual registration can ingest later.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ctsynth import __version__
from ctsynth.config import ExperimentConfig, config_sha256

# The one definition of "source" for citability (D-008): what determines a run's result.
# Shared by the run metadata and `ctsynth.hpc.check.preflight`, so local and HPC paths agree.
SOURCE_PATHS = ("src", "tests", "configs", "scripts", "projects", "pyproject.toml", "uv.lock")
# Outputs under projects/ (incl. tracked curated evidence such as exp002/results) are not source.
_OUTPUT_EXCLUDE = ":(exclude,glob)projects/**/results*/**"


def _git(*args: str, repo: str | Path | None = None) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=repo
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dirty_source(repo: str | Path | None = None) -> str | None:
    """`git status --porcelain` lines for modified, staged or untracked source ('' = clean).

    Untracked files count because Research OS packages the working tree, not the commit.
    None means git could not answer (no repository): unknown, never reported as clean.
    """
    return _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *SOURCE_PATHS,
        _OUTPUT_EXCLUDE,
        repo=repo,
    )


def _lock_sha256(repo: str | Path | None) -> str | None:
    """Hash of uv.lock: without it the environment is not pinned in the evidence."""
    root = _git("rev-parse", "--show-toplevel", repo=repo)
    lock = Path(root) / "uv.lock" if root else Path(repo or ".") / "uv.lock"
    try:
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    except OSError:
        return None


def collect(cfg: ExperimentConfig, repo: str | Path | None = None) -> dict[str, object]:
    import numpy

    dirty = dirty_source(repo)
    return {
        "experiment_id": cfg.experiment_id,
        "uv_lock_sha256": _lock_sha256(repo),
        "config_sha256": config_sha256(cfg),
        "config": cfg.model_dump(),
        "git_commit": _git("rev-parse", "HEAD", repo=repo),
        "git_dirty": bool(_git("status", "--porcelain", repo=repo)),
        # Research OS writes its own event log while the run executes, so git_dirty is always
        # true for a `research run exec` run. Citability depends on the SOURCE being clean
        # (SOURCE_PATHS); None = git unavailable, which postcheck treats as not citable.
        "git_dirty_source": None if dirty is None else bool(dirty),
        "ctsynth_version": __version__,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "utc_time": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write(cfg: ExperimentConfig, out_dir: str | Path, repo: str | Path | None = None) -> Path:
    path = Path(out_dir) / "run_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collect(cfg, repo), indent=2, sort_keys=True))
    return path
