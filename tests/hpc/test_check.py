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


def test_preflight_without_config_returns_commit_and_lock_only(repo: Path) -> None:
    fp = preflight(repo)
    assert set(fp) == {"git_commit", "uv_lock_sha256"}


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


# --- local (runmeta) and HPC (preflight) paths must make the same cleanliness decision ---------
from ctsynth.config import load_config  # noqa: E402
from ctsynth.runmeta import SOURCE_PATHS, collect  # noqa: E402


def _both(repo: Path) -> tuple[bool, bool]:
    """(runmeta says dirty, preflight refuses) for the same tree."""
    cfg_path = repo / "configs" / "c.yaml"
    local = bool(collect(load_config(cfg_path), repo)["git_dirty_source"])
    try:
        preflight(repo, cfg_path)
        return local, False
    except PreflightError:
        return local, True


def test_clean_tree_agrees(repo: Path) -> None:
    assert _both(repo) == (False, False)


@pytest.mark.parametrize(
    "path", [p for p in SOURCE_PATHS if p not in ("uv.lock", "pyproject.toml")]
)
def test_untracked_file_in_each_source_dir_is_dirty_on_both_paths(repo: Path, path: str) -> None:
    d = repo / path
    d.mkdir(exist_ok=True)
    (d / "new_file.py").write_text("x = 1\n")
    assert _both(repo) == (True, True)


@pytest.mark.parametrize("name", ["uv.lock", "pyproject.toml"])
def test_modified_root_file_is_dirty_on_both_paths(repo: Path, name: str) -> None:
    (repo / name).write_text("changed\n")
    assert _both(repo) == (True, True)


def test_staged_change_is_dirty_on_both_paths(repo: Path) -> None:
    (repo / "src" / "a.py").write_text("x = 3\n")
    _run(repo, "add", "src/a.py")
    assert _both(repo) == (True, True)


def test_project_results_are_outputs_not_source(repo: Path) -> None:
    out = repo / "projects" / "p" / "results_x"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text("{}")
    assert _both(repo) == (False, False)  # untracked output
    _run(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-f", "-A")
    _run(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "curated results")
    (out / "metrics.json").write_text('{"changed": 1}')
    assert _both(repo) == (False, False)  # tracked curated evidence (exp002) modified
    (repo / "projects" / "p" / "run.py").write_text("x = 1\n")
    assert _both(repo) == (True, True)  # a driver next to it still counts


def test_not_a_repository_is_unknown_never_clean(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.yaml").write_text("experiment_id: EXP-009\nname: t\n")
    assert (
        collect(load_config(tmp_path / "configs" / "c.yaml"), tmp_path)["git_dirty_source"] is None
    )
    with pytest.raises(PreflightError, match="unknown"):
        preflight(tmp_path, tmp_path / "configs" / "c.yaml")


def test_postcheck_rejects_unknown_dirty_state(tmp_path: Path) -> None:
    run = _imported(tmp_path, {**EXPECT, "git_dirty_source": None}, ())
    assert any("not citable" in m for m in postcheck(run, EXPECT, []))
