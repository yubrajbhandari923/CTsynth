import json
import subprocess
from pathlib import Path

import pytest

from ctsynth.hpc.check import PreflightError, output_hashes, postcheck, preflight


def _run(repo: Path, *a: str) -> None:
    subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "configs").mkdir()
    (r / "src" / "a.py").write_text("x = 1\n")
    (r / "uv.lock").write_text("lock\n")
    (r / "configs" / "c.yaml").write_text("experiment_id: EXP-009\nname: t\nparams: {a: 1}\n")
    _run(r, "init", "-q")
    _run(r, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _run(r, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return r


def test_preflight_clean_returns_fingerprints(repo: Path) -> None:
    fp = preflight(repo, repo / "configs" / "c.yaml")
    assert fp["experiment_id"] == "EXP-009" and len(fp["git_commit"]) == 40
    assert len(fp["config_sha256"]) == 64 and len(fp["uv_lock_sha256"]) == 64


def test_preflight_refuses_modified_tracked(repo: Path) -> None:
    (repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(PreflightError, match="git_dirty_source"):
        preflight(repo, repo / "configs" / "c.yaml")


def test_preflight_refuses_untracked_source(repo: Path) -> None:
    (repo / "src" / "new.py").write_text("y = 1\n")  # would enter a native bundle
    with pytest.raises(PreflightError, match="new.py"):
        preflight(repo, repo / "configs" / "c.yaml")


def test_preflight_refuses_bad_config(repo: Path) -> None:
    (repo / "configs" / "bad.yaml").write_text("experiment_id: E\nname: t\ntypo: 1\n")
    _run(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _run(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "bad")
    with pytest.raises(PreflightError, match="invalid"):
        preflight(repo, repo / "configs" / "bad.yaml")


def _imported(tmp_path: Path, meta: dict[str, object] | None, files: tuple[str, ...]) -> Path:
    out = tmp_path / "RUN-0001" / "collected" / "outputs"
    out.mkdir(parents=True)
    if meta is not None:
        (out / "run_meta.json").write_text(json.dumps(meta))
    for f in files:
        (out / f).write_text("data")
    return tmp_path / "RUN-0001"


EXPECT = {"git_commit": "a" * 40, "config_sha256": "b" * 64, "uv_lock_sha256": "c" * 64}
GOOD = {**EXPECT, "git_dirty_source": False}


def test_postcheck_ok(tmp_path: Path) -> None:
    run = _imported(tmp_path, GOOD, ("metrics.json",))
    assert postcheck(run, EXPECT, ["metrics.json"]) == []


def test_postcheck_detects_identity_mismatch_and_dirty(tmp_path: Path) -> None:
    run = _imported(tmp_path, {**GOOD, "config_sha256": "z", "git_dirty_source": True}, ())
    msgs = postcheck(run, EXPECT, [])
    assert any("config_sha256" in m for m in msgs) and any("not citable" in m for m in msgs)


def test_postcheck_missing_output_and_meta(tmp_path: Path) -> None:
    run = _imported(tmp_path, None, ())
    msgs = postcheck(run, EXPECT, ["metrics.json"])
    assert any("run_meta.json missing" in m for m in msgs)
    assert any("metrics.json" in m for m in msgs)


def test_postcheck_failed_run_needs_no_outputs(tmp_path: Path) -> None:
    run = _imported(tmp_path, None, ())
    assert postcheck(run, EXPECT, ["metrics.json"], failed=True) == []


def test_output_hashes_change_when_modified(tmp_path: Path) -> None:
    run = _imported(tmp_path, GOOD, ("metrics.json",))
    before = output_hashes(run)
    (run / "collected" / "outputs" / "metrics.json").write_text("tampered")
    assert output_hashes(run) != before
