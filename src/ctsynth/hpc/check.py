"""Scientific guards around Research OS native HPC (it owns bundle, submit, sync, import).

`preflight` runs on the laptop before `research hpc submit`: refuses a dirty or untracked source
tree and returns the fingerprints to record as experiment parameters. `postcheck` runs after
`research hpc sync` and verifies the imported run against them. Neither opens a connection.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

from ctsynth.config import config_sha256, load_config
from ctsynth.hashing import sha256_file
from ctsynth.runmeta import dirty_source


class PreflightError(Exception):
    pass


def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise PreflightError(f"git {' '.join(args)} failed in {repo}: {p.stderr.strip()}")
    return p.stdout.strip()


def preflight(repo: Path, config_path: Path) -> dict[str, str]:
    """Fingerprints of a clean source tree; raises PreflightError if anything source-like is dirty.

    Untracked (non-ignored) files count: Research OS packages the working tree, not the commit.
    """
    repo = Path(repo)
    dirty = dirty_source(repo)
    if dirty is None:
        raise PreflightError(f"git_dirty_source unknown: {repo} is not a git repository")
    if dirty:
        raise PreflightError("git_dirty_source: commit or remove first:\n" + dirty)
    commit = _git(repo, "rev-parse", "HEAD")
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # any load/validation failure is a refusal, with its reason
        raise PreflightError(f"config {config_path} is invalid: {exc}") from exc
    return {
        "experiment_id": cfg.experiment_id,
        "git_commit": commit,
        "config_sha256": config_sha256(cfg),
        "uv_lock_sha256": sha256_file(repo / "uv.lock"),
    }


def postcheck(
    run_dir: Path, expect: dict[str, str], required_outputs: list[str], *, failed: bool = False
) -> list[str]:
    """Problems with an imported run (empty list = ok). `run_dir` is `.research/runs/RUN-…`.

    Reads collected/outputs/run_meta.json (written by ctsynth.runmeta in the job). A failed run need
    not have its required outputs, but if it wrote run_meta its identity is still checked.
    """
    out = Path(run_dir) / "collected" / "outputs"
    problems: list[str] = []
    meta_path = out / "run_meta.json"
    if meta_path.is_file():
        meta: dict[str, Any] = json.loads(meta_path.read_text())
        for key in ("git_commit", "config_sha256", "uv_lock_sha256", "experiment_id"):
            if key in expect and meta.get(key) != expect[key]:
                problems.append(f"{key}: run has {meta.get(key)!r}, expected {expect[key]!r}")
        if meta.get("git_dirty_source") is not False:
            problems.append(
                f"run_meta git_dirty_source={meta.get('git_dirty_source')!r}: not citable"
            )
    elif not failed:
        problems.append("run_meta.json missing from collected outputs")
    if not failed:
        problems += [
            f"required output missing: {n}" for n in required_outputs if not (out / n).is_file()
        ]
    return problems


def output_hashes(run_dir: Path) -> dict[str, str]:
    out = Path(run_dir) / "collected" / "outputs"
    return {p.name: sha256_file(p) for p in sorted(out.glob("*")) if p.is_file()}
