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
from ctsynth.config import ExperimentConfig
from ctsynth.hashing import config_sha256


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _lock_sha256() -> str | None:
    """Hash of uv.lock: without it the environment is not pinned in the evidence."""
    root = _git("rev-parse", "--show-toplevel")
    lock = Path(root) / "uv.lock" if root else Path("uv.lock")
    try:
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    except OSError:
        return None


def collect(cfg: ExperimentConfig) -> dict[str, object]:
    import numpy

    return {
        "experiment_id": cfg.experiment_id,
        "uv_lock_sha256": _lock_sha256(),
        "config_sha256": config_sha256(cfg),
        "config": cfg.model_dump(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        # Research OS writes its own event log while the run executes, so git_dirty is always
        # true for a `research run exec` run. Citability depends on the SOURCE being clean.
        "git_dirty_source": bool(_git("status", "--porcelain", "--", "src", "tests", "configs")),
        "ctsynth_version": __version__,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "utc_time": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write(cfg: ExperimentConfig, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "run_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collect(cfg), indent=2, sort_keys=True))
    return path
