"""End-to-end laptop -> worker -> laptop workflow, with a fake `research run attach`."""

import json
import shutil
from pathlib import Path

import pytest
from conftest import FIXED_NOW, REQUIRED, FakeAttach, Laptop, git, rewrite_json, run_on_worker

from ctsynth.hpc.bundle import RESULT_MANIFEST, BundleError, write_manifest
from ctsynth.hpc.importer import ImportRefused, import_result
from ctsynth.hpc.report import build_report, write_report
from ctsynth.hpc.verify import verify


def _import(done: Path, laptop: Laptop, attach: FakeAttach, **kw: object):  # type: ignore[no-untyped-def]
    return import_result(
        done,
        repo=laptop.repo,
        runs_root=laptop.root / "runs_imported",
        attach=attach,
        **kw,  # type: ignore[arg-type]
    )


# (a) happy path ---------------------------------------------------------------------------
def test_happy_path(laptop: Laptop, worker_dir: Path, fake_attach: FakeAttach) -> None:
    bundle_dir = laptop.prepare()
    assert bundle_dir.name == f"EXP-001-{laptop.sha[:8]}-20260102T030405Z"
    b = json.loads((bundle_dir / "bundle.json").read_text())
    assert b["git_commit"] == laptop.sha and b["git_dirty_source"] is False
    assert verify(bundle_dir, repo=laptop.repo).ok  # standalone-verifiable before transfer
    done = run_on_worker(laptop, worker_dir, bundle_dir)
    rep = verify(done, repo=laptop.repo)
    assert rep.ok and rep.kind == "result", rep.problems
    res = _import(done, laptop, fake_attach)
    assert res.run_id == "RUN-0001" and res.status == "completed"
    assert (res.dest / "metrics.json").read_bytes() == (done / "metrics.json").read_bytes()
    (req,) = fake_attach.calls
    assert req.experiment_id == "EXP-001" and req.exit_code == 0 and req.status == "completed"
    assert req.stdout.is_relative_to(laptop.root / "runs_imported") and req.stdout.exists()
    assert req.params["git_commit"] == laptop.sha
    md = write_report(res.run_id, laptop.root / "runs_imported").read_text()
    for needle in (laptop.sha, b["config_sha256"], "completed", "rmse", "0.125", "bias.water",
                   "figure.png"):  # fmt: skip
        assert needle in md


# (b) missing required output ----------------------------------------------------------------
def test_missing_required_output(laptop: Laptop, worker_dir: Path, fake_attach: FakeAttach) -> None:
    done = run_on_worker(laptop, worker_dir, laptop.prepare(laptop.write_config(skip_figure=True)))
    rep = verify(done, repo=laptop.repo)
    assert not rep.ok and any("figure.png" in p and "missing" in p.lower() for p in rep.problems)
    with pytest.raises(ImportRefused, match="figure.png"):
        _import(done, laptop, fake_attach)
    assert not fake_attach.calls and not (laptop.root / "runs_imported").exists()


# (c) corrupted file -------------------------------------------------------------------------
def test_corrupted_file(laptop: Laptop, done: Path, fake_attach: FakeAttach) -> None:
    f = done / "metrics.json"
    f.write_bytes(
        f.read_bytes().replace(b"0.125", b"0.126")
    )  # same length: only the hash catches it
    rep = verify(done, repo=laptop.repo)
    assert not rep.ok and any("metrics.json" in p and "sha256" in p for p in rep.problems)
    with pytest.raises(ImportRefused):
        _import(done, laptop, fake_attach)
    assert not fake_attach.calls


# (d) failed job -----------------------------------------------------------------------------
def test_failed_job_is_importable_but_marked(
    laptop: Laptop, worker_dir: Path, fake_attach: FakeAttach
) -> None:
    done = run_on_worker(laptop, worker_dir, laptop.prepare(laptop.write_config(fail=True)))
    r = json.loads((done / "result.json").read_text())
    assert r["status"] == "failed" and r["exit_code"] == 3
    assert verify(done, repo=laptop.repo).ok  # failed run needs no required outputs
    res = _import(done, laptop, fake_attach)
    (req,) = fake_attach.calls
    assert (req.status, req.exit_code, res.status) == ("failed", 3, "failed")
    assert "boom" in req.stderr.read_text() and "FAILED" in req.label
    assert "FAILED" in build_report(res.run_id, laptop.root / "runs_imported")


def test_inconsistent_status_rejected(laptop: Laptop, done: Path) -> None:
    rewrite_json(done / "result.json", exit_code=3)  # claims completed but exit 3
    write_manifest(done, RESULT_MANIFEST)  # attacker also fixes the manifest
    assert any("exit_code" in p for p in verify(done, repo=laptop.repo).problems)


# (e) repeated import / target exists --------------------------------------------------------
def test_repeated_import_rejected(laptop: Laptop, done: Path, fake_attach: FakeAttach) -> None:
    res = _import(done, laptop, fake_attach)
    before = (res.dest / "metrics.json").read_bytes()
    with pytest.raises(ImportRefused, match="already imported"):
        _import(done, laptop, fake_attach)
    assert len(fake_attach.calls) == 1 and (res.dest / "metrics.json").read_bytes() == before


def test_existing_target_never_overwritten(
    laptop: Laptop, done: Path, fake_attach: FakeAttach
) -> None:
    target = laptop.root / "runs_imported" / "RUN-0001"
    target.mkdir(parents=True)
    (target / "precious").write_text("keep")
    with pytest.raises(ImportRefused, match="exists"):
        _import(done, laptop, fake_attach)
    assert (target / "precious").read_text() == "keep"
    assert not list((laptop.root / "runs_imported").glob(".staging-*"))  # staging cleaned up


def test_attach_failure_leaves_nothing(laptop: Laptop, done: Path) -> None:
    def boom(req: object) -> str:
        raise ImportRefused("policy blocked")

    with pytest.raises(ImportRefused):
        _import(done, laptop, boom)  # type: ignore[arg-type]
    runs = laptop.root / "runs_imported"
    assert not runs.exists() or not [p for p in runs.iterdir() if p.name != ".imported"]


# (f) dirty source ---------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["modify_tracked", "add_untracked"])
def test_prepare_refuses_dirty_source(laptop: Laptop, how: str) -> None:
    cfg = laptop.write_config()
    if how == "modify_tracked":
        (laptop.repo / "src" / "job.py").write_text("# edited\n")
    else:
        (laptop.repo / "src" / "new.py").write_text("x = 1\n")
    with pytest.raises(BundleError, match="git_dirty_source"):
        laptop.prepare(cfg)
    assert not (laptop.root / "bundles").exists()  # nothing half-written


def test_prepare_never_overwrites_bundle(laptop: Laptop) -> None:
    cfg = laptop.write_config()
    first = laptop.prepare(cfg)
    stamp = (first / "bundle.json").read_bytes()
    with pytest.raises(BundleError, match="exists"):
        laptop.prepare(cfg, now=FIXED_NOW)  # same commit + same second => same id
    assert (first / "bundle.json").read_bytes() == stamp


# (g) ambiguous / unknown commit -------------------------------------------------------------
def _retarget_commit(bundle_dir: Path, sha: str) -> None:
    rewrite_json(bundle_dir / "bundle.json", git_commit=sha)
    write_manifest(
        bundle_dir, "MANIFEST.sha256"
    )  # make the manifest consistent: isolate the commit check


@pytest.mark.parametrize(
    ("sha", "word"),
    [("short", "ambiguous"), ("0123456789abcdef0123456789abcdef01234567", "unknown")],
)
def test_bad_commit_rejected(laptop: Laptop, sha: str, word: str) -> None:
    b = laptop.prepare()
    _retarget_commit(b, laptop.sha[:8] if word == "ambiguous" else sha)
    probs = verify(b, repo=laptop.repo).problems
    assert any(word in p.lower() for p in probs), probs


def test_commit_check_skipped_without_repo_is_explicit(laptop: Laptop) -> None:
    b = laptop.prepare()
    _retarget_commit(b, "0123456789abcdef0123456789abcdef01234567")
    # Worker has no git repo: format-valid sha passes there; the laptop import is the real gate.
    assert verify(b, repo=None).ok


# (h) config mutated after prepare -----------------------------------------------------------
def test_config_mutation_detected_by_manifest(laptop: Laptop) -> None:
    b = laptop.prepare()
    cfg = b / "config.yaml"
    cfg.write_text(cfg.read_text().replace('"seed": 7', '"seed": 8'))
    assert any("config.yaml" in p for p in verify(b, repo=laptop.repo).problems)


def test_config_mutation_detected_even_if_manifest_forged(laptop: Laptop) -> None:
    b = laptop.prepare()
    cfg = b / "config.yaml"
    cfg.write_text(cfg.read_text().replace('"seed": 7', '"seed": 8'))
    write_manifest(b, "MANIFEST.sha256")
    assert any("config_sha256" in p for p in verify(b, repo=laptop.repo).problems)


# (i) external checkpoint --------------------------------------------------------------------
def test_external_checkpoint_recorded_not_copied(
    laptop: Laptop, worker_dir: Path, fake_attach: FakeAttach
) -> None:
    ckpt = worker_dir / "scratch" / "model.pt"
    cfg = laptop.write_config(ckpt_out=str(ckpt))
    done = run_on_worker(
        laptop, worker_dir, laptop.prepare(cfg, checkpoints=[{"path": str(ckpt), "external": True}])
    )
    assert ckpt.exists() and not list(done.rglob("model.pt"))  # never entered the result dir
    (c,) = json.loads((done / "result.json").read_text())["checkpoints"]
    assert c["external"] is True and c["bytes"] == 7000 and c["location"] == str(ckpt)
    res = _import(done, laptop, fake_attach)
    assert not list(res.dest.rglob("model.pt"))
    md = build_report(res.run_id, laptop.root / "runs_imported")
    assert c["sha256"] in md and str(ckpt) in md and "external" in md.lower()


def test_missing_declared_checkpoint_on_completed_run_rejected(
    laptop: Laptop, worker_dir: Path
) -> None:
    ckpt = worker_dir / "scratch" / "never_written.pt"
    done = run_on_worker(
        laptop, worker_dir, laptop.prepare(checkpoints=[{"path": str(ckpt), "external": True}])
    )
    assert any("never_written.pt" in p for p in verify(done, repo=laptop.repo).problems)


# (j) old commit still imports ---------------------------------------------------------------
def test_old_commit_result_imports_and_reports_old_sha(
    laptop: Laptop, worker_dir: Path, fake_attach: FakeAttach
) -> None:
    done = run_on_worker(laptop, worker_dir, laptop.prepare())
    old = json.loads((done / "result.json").read_text())["git_commit"]
    (laptop.repo / "src" / "later.py").write_text("y = 2\n")  # laptop moves on
    git(laptop.repo, "add", "-A")
    git(laptop.repo, "commit", "-qm", "later")
    assert git(laptop.repo, "rev-parse", "HEAD") != old
    res = _import(done, laptop, fake_attach)
    assert fake_attach.calls[0].params["git_commit"] == old
    md = build_report(res.run_id, laptop.root / "runs_imported")
    assert old in md and git(laptop.repo, "rev-parse", "HEAD") not in md


# data provenance ----------------------------------------------------------------------------
def test_data_manifest_detects_changed_input(laptop: Laptop, tmp_path: Path) -> None:
    data = tmp_path / "datadir"
    data.mkdir()
    (data / "in.bin").write_bytes(b"0123")
    b = laptop.prepare(data_files=[data / "in.bin"])
    entry = json.loads((b / "bundle.json").read_text())["data"][0]
    assert entry["bytes"] == 4 and len(entry["sha256"]) == 64
    worker_data = tmp_path / "worker_data"
    shutil.copytree(data, worker_data)
    assert verify(b, repo=laptop.repo, data_root=worker_data).ok
    (worker_data / "in.bin").write_bytes(b"0124")
    assert any("in.bin" in p for p in verify(b, repo=laptop.repo, data_root=worker_data).problems)


def test_reexport_over_completed_result_refused(
    laptop: Laptop, worker_dir: Path, done: Path
) -> None:
    from ctsynth.hpc.bundle import export_result

    with pytest.raises(BundleError):
        export_result(worker_dir / "bundle", worker_dir / "scratch" / "out", 0,
                      started_utc="a", ended_utc="b")  # fmt: skip


def test_result_required_list_matches(laptop: Laptop) -> None:
    b = json.loads((laptop.prepare() / "bundle.json").read_text())
    assert b["required_outputs"] == REQUIRED and b["seeds"] == {"seed": 7}
