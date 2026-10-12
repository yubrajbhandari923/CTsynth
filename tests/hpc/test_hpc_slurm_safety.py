"""SLURM rendering (k), verify-never-executes (l), and Research OS attach plumbing."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import Laptop, run_on_worker

from ctsynth.hpc.bundle import HARDWARE_FIELDS, UNSET, render_sbatch, write_manifest
from ctsynth.hpc.cli import main
from ctsynth.hpc.importer import AttachRequest, build_attach_argv
from ctsynth.hpc.verify import verify

# Real cluster vocabulary that must never be baked into the template or defaults.
BANNED = ("duke", "dcc", "scavenger", "common", "/work/", "/hpc/", "gpu-common", "rtx", "a100")


def _bundle(laptop: Laptop, **kw: object) -> dict[str, object]:
    return json.loads((laptop.prepare(**kw) / "bundle.json").read_text())  # type: ignore[no-any-return]


# (k) SLURM ---------------------------------------------------------------------------------
def test_sbatch_all_unset_is_todo_not_default(laptop: Laptop) -> None:
    b = _bundle(laptop)
    assert all(b["hardware"][f] == UNSET for f in HARDWARE_FIELDS)  # type: ignore[index]
    text = render_sbatch(b)
    # No active #SBATCH directive at all: nothing may be silently inherited.
    assert not [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
    for flag in ("partition", "account", "gpus", "cpus", "mem", "time", "job-name"):
        assert re.search(rf"# TODO\(unset\).*{flag}", text), flag
    assert re.search(r"# TODO\(unset\).*scratch", text, re.I)
    assert re.search(r"# TODO\(unset\).*environment", text, re.I)
    # Scan everything EXCEPT the commit sha: "dcc" and "a100" are spellable in hex, so a random
    # 40-hex sha trips this ~0.9% of the time (measured). The sha is required content, not a
    # cluster default, so excluding it keeps the check strict without being flaky.
    scanned = text.lower().replace(str(b["git_commit"]).lower(), "")
    assert not any(w in scanned for w in BANNED)
    assert b["git_commit"] in text and "ctsynth export" in text  # type: ignore[operator]


def test_sbatch_set_fields_render_and_only_those(laptop: Laptop) -> None:
    hw = {"partition": "P1", "account": "ACC", "gpus": "1", "time": "00:10:00", "scratch": "/s/x"}
    text = render_sbatch(
        _bundle(laptop, hardware=hw, env_setup=["module load foo", "export BAR=1"])
    )
    assert "#SBATCH --partition=P1" in text and "#SBATCH --account=ACC" in text
    assert "#SBATCH --time=00:10:00" in text and "SCRATCH_DIR=/s/x" in text
    assert "module load foo\nexport BAR=1" in text
    active = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
    assert not any("--cpus" in ln or "--mem" in ln or "--job-name" in ln for ln in active)
    assert re.search(r"# TODO\(unset\).*mem", text)


def test_sbatch_entry_point_is_pinned_and_quoted(laptop: Laptop) -> None:
    text = render_sbatch(_bundle(laptop))
    assert 'python src/job.py --config "$BUNDLE_DIR/config.yaml" --out "$OUT"' in text


# (l) verify is inert on hostile directories -----------------------------------------------
def test_verify_never_executes_directory_contents(
    laptop: Laptop, worker_dir: Path, tmp_path: Path
) -> None:
    done = run_on_worker(laptop, worker_dir, laptop.prepare())
    sentinel = tmp_path / "PWNED"
    evil = f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('x')\n"
    # Everything Python or a shell could auto-load from a directory, plus a stdlib shadow:
    for name in ("sitecustomize.py", "usercustomize.py", "json.py", "yaml.py", "conftest.py",
                 "evil.pth", "git"):  # fmt: skip
        (done / name).write_text(evil)
    (done / "run.sbatch").write_text(f"#!/bin/bash\ntouch {sentinel}\n")
    (done / "git").chmod(0o755)
    write_manifest(done, "RESULT.sha256")  # hashes are valid: only inertness is under test
    env_before = dict(os.environ)
    rep = verify(done, repo=laptop.repo)
    assert not sentinel.exists()
    assert dict(os.environ) == env_before and rep.kind == "result"
    # cwd inside the hostile dir must not matter either (no CWD-relative imports/executables)
    cwd = Path.cwd()
    os.chdir(done)
    try:
        verify(done, repo=laptop.repo)
    finally:
        os.chdir(cwd)
    assert not sentinel.exists()


def test_verify_rejects_unlisted_files_and_traversal(laptop: Laptop) -> None:
    b = laptop.prepare()
    (b / "extra.txt").write_text("sneaky")
    assert any("extra.txt" in p for p in verify(b, repo=laptop.repo).problems)
    (b / "extra.txt").unlink()
    m = b / "MANIFEST.sha256"
    m.write_text(m.read_text() + f"{'0' * 64}  ../outside\n")
    assert any("outside" in p for p in verify(b, repo=laptop.repo).problems)


def test_verify_on_garbage_does_not_raise(tmp_path: Path) -> None:
    assert not verify(tmp_path / "nope").ok
    d = tmp_path / "d"
    d.mkdir()
    (d / "bundle.json").write_text("{not json")
    rep = verify(d)
    assert not rep.ok and rep.kind in {"bundle", "unknown"}


# Research OS plumbing ----------------------------------------------------------------------
def test_build_attach_argv() -> None:
    req = AttachRequest(
        "EXP-001", "failed", 3, Path("/r/o.log"), Path("/r/e.log"), "lbl", "python a.py --x 1",
        "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", "99", {"git_commit": "abc"},
    )  # fmt: skip
    argv = build_attach_argv(req, research_bin=Path("/bin/research"), root=Path("/proj"))
    assert argv[:2] == ["/bin/research", "--json"] and argv[2:4] == ["--root", "/proj"]
    i = argv.index("attach")
    assert argv[i - 1] == "run" and argv[i + 1] == "EXP-001"
    assert argv[argv.index("--status") + 1] == "failed"
    assert argv[argv.index("--exit-code") + 1] == "3"
    assert argv[argv.index("--slurm-job") + 1] == "99"
    assert "git_commit=abc" in argv
    assert argv[argv.index("--") + 1 :] == ["python", "a.py", "--x", "1"]


RESEARCH_BIN = Path(__file__).resolve().parents[2] / ".research" / "bin" / "research"


@pytest.mark.skipif(not RESEARCH_BIN.exists(), reason="project-local research CLI not installed")
def test_cli_end_to_end_against_throwaway_research_root(
    laptop: Laptop, worker_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Whole CLI, real `research run attach`, but --research-root points at a temp project, so
    the real .research/ of this repo is never touched."""
    root = tmp_path / "rproj"
    root.mkdir()
    sh = lambda *a: subprocess.run(  # noqa: E731
        [str(RESEARCH_BIN), "--root", str(root), *a],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,  # `init` initialises the cwd; --root alone is not enough for it
    )
    sh("init")
    sh("experiment", "create", "tmp experiment")  # becomes EXP-001, matching the fixture config
    cfg = laptop.write_config()
    assert main(["prepare", "--repo", str(laptop.repo), "--config", str(cfg),
                 "--out-root", str(laptop.root / "bundles"), "--require-output", "metrics.json",
                 "--require-output", "figure.png", "--entry", "python", "src/job.py",
                 "--config", "{config}", "--out", "{out}"]) == 0  # fmt: skip
    bundle_dir = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert main(["verify", str(bundle_dir), "--repo", str(laptop.repo)]) == 0
    done = run_on_worker(laptop, worker_dir, bundle_dir)
    runs = laptop.root / "runs_imported"
    args = ["import", str(done), "--repo", str(laptop.repo), "--runs-root", str(runs),
            "--research-bin", str(RESEARCH_BIN), "--research-root", str(root)]  # fmt: skip
    assert main(args) == 0
    run_id = re.search(r"RUN-\d+", capsys.readouterr().out).group(0)  # type: ignore[union-attr]
    assert (runs / run_id / "metrics.json").exists()
    assert run_id in sh("run", "list").stdout  # registered in the TEMP research root
    assert main(args) == 1  # idempotent refusal surfaces as exit code 1
    assert main(["report", run_id, "--runs-root", str(runs)]) == 0
    assert (runs / f"{run_id}.report.md").exists()
    shutil.rmtree(root)
