"""bundle-check, paired-check and xcist-check: the fast guards behind the Research OS hooks."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ctsynth.hpc.check import bundle_check
from ctsynth.hpc.cli import main
from ctsynth.paired.mock import make_mock_record


def _repo(tmp_path: Path, exclude: list[str]) -> Path:
    (tmp_path / ".research").mkdir()
    (tmp_path / ".research" / "config.yaml").write_text(
        yaml.safe_dump({"hpc": {"bundle": {"exclude": exclude}}})
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    return tmp_path


EXCLUDE = [".venv*", ".research", "data/aapm", "projects/*/results*"]


def test_bundle_check_clean_tree(tmp_path: Path) -> None:
    rep = bundle_check(_repo(tmp_path, EXCLUDE))
    assert rep["problems"] == [] and rep["bytes"] == len("x = 1\n")


def test_bundle_check_flags_leaked_environment_f038(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [".research"])  # exclude list that forgot .venv* (the F-038 mistake)
    (repo / ".venv-xcist" / "lib" / "site-packages").mkdir(parents=True)
    (repo / ".venv-xcist" / "lib" / "site-packages" / "m.py").write_text("y = 1\n")
    rep = bundle_check(repo)
    assert any(".venv-xcist" in p for p in rep["problems"])
    assert main(["bundle-check", "--repo", str(repo)]) == 1


def test_bundle_check_flags_oversize_and_empty_exclude(tmp_path: Path) -> None:
    repo = _repo(tmp_path, EXCLUDE)
    (repo / "big.bin").write_bytes(b"0" * 2_000_000)
    assert any("MB" in p for p in bundle_check(repo, max_mb=1.0)["problems"])
    (tmp_path / "e").mkdir()
    empty = _repo(tmp_path / "e", [])
    assert any("empty" in p for p in bundle_check(empty)["problems"])


def _record(tmp: Path) -> Path:
    return make_mock_record(
        tmp / "R1",
        seed=1,
        phantom_id="P1",
        acquisition_id="A1",
        energies_keV=np.array([40.0, 60.0, 80.0]),
        fluence=np.array([1e5, 2e5, 1e5]),
        kappa=np.array([[0.27, 0.205, 0.184], [4.0, 1.2, 0.6]]),
        names=("water", "calcium"),
        ref_density=np.array([1.0, 1.55]),
    )


def test_paired_check_ok_and_detects_tampering(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = _record(tmp_path)
    assert main(["paired-check", str(rec)]) == 0
    assert "P1/A1" in capsys.readouterr().out
    arr = next(rec.glob("*.npy"))
    data = np.load(arr)
    np.save(arr, data + 1)
    assert main(["paired-check", str(rec)]) == 1


def test_xcist_check_reports_without_raising(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["xcist-check"])
    rep = json.loads(capsys.readouterr().out)
    assert set(rep) == {"gecatsim", "ok_simulate", "ok_recon", "problems"}
    assert rc == (0 if rep["ok_simulate"] and rep["ok_recon"] else 1)
    if rep["gecatsim"] is None:  # core env: absence is reported, not an exception
        assert rep["problems"] and rc == 1


def _artifacts(repo: Path, rows: list[dict]) -> None:
    (repo / ".research").mkdir(exist_ok=True)
    (repo / ".research" / "artifacts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def _meta_row(repo: Path, dirty: object, exp: str = "EXP-1") -> dict:
    (repo / "m").mkdir(exist_ok=True)
    (repo / "m" / "run_meta.json").write_text(json.dumps({"git_dirty_source": dirty}))
    return {"id": "A-1", "experiment_id": exp, "name": "run_meta.json", "path": "m/run_meta.json"}


def test_run_check_flags_non_citable_registered_run_meta(tmp_path: Path) -> None:
    _artifacts(tmp_path, [_meta_row(tmp_path, True)])
    assert main(["run-check", "--repo", str(tmp_path), "--experiment", "EXP-1"]) == 1
    _artifacts(tmp_path, [_meta_row(tmp_path, None)])  # unknown is not citable either
    assert main(["run-check", "--repo", str(tmp_path), "--experiment", "EXP-1"]) == 1


def test_run_check_ok_and_missing_meta_is_only_a_note(tmp_path: Path) -> None:
    _artifacts(tmp_path, [_meta_row(tmp_path, False)])
    assert main(["run-check", "--repo", str(tmp_path), "--experiment", "EXP-1"]) == 0
    assert main(["run-check", "--repo", str(tmp_path), "--experiment", "EXP-2"]) == 0  # no run_meta


def test_run_check_requires_an_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_EXPERIMENT", raising=False)
    _artifacts(tmp_path, [])
    assert main(["run-check", "--repo", str(tmp_path)]) == 2


def test_bundle_check_does_not_flag_files_merely_named_like_results(tmp_path: Path) -> None:
    repo = _repo(tmp_path, EXCLUDE)
    (repo / "src" / "results.py").write_text("x = 1\n")
    (repo / "projects").mkdir()
    (repo / "projects" / "results_utils.py").write_text("y = 1\n")
    assert bundle_check(repo)["problems"] == []
    (repo / "analysis" / "results_v2").mkdir(
        parents=True
    )  # an unexcluded results DIRECTORY is flagged
    (repo / "analysis" / "results_v2" / "m.npy").write_bytes(b"0")
    assert any("results_v2" in p for p in bundle_check(repo)["problems"])
