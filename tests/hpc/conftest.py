"""Simulated HPC boundary: two temp dirs (laptop, worker) and NO cluster.

The worker never sees the laptop repo: it gets a `git archive` of the pinned commit plus a
byte-copy of the bundle, runs the entry point with its own cwd, and the result dir is copied back.
"""

import io
import json
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ctsynth.hpc.bundle import export_result, prepare, resolve_entry
from ctsynth.hpc.importer import AttachRequest

JOB = """\
import argparse, json, sys
from pathlib import Path
import yaml
ap = argparse.ArgumentParser()
ap.add_argument("--config"); ap.add_argument("--out")
a = ap.parse_args()
cfg = yaml.safe_load(Path(a.config).read_text())
p = cfg["params"]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if p.get("fail"):
    print("boom", file=sys.stderr); sys.exit(3)
(out / "metrics.json").write_text(json.dumps({"rmse": 0.125, "bias": {"water": -0.5}, "n": 4}))
if not p.get("skip_figure"):
    (out / "figure.png").write_bytes(b"\\x89PNG fake")
if p.get("ckpt_out"):
    Path(p["ckpt_out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(p["ckpt_out"]).write_bytes(b"WEIGHTS" * 1000)
print("done")
"""
ENTRY = ["python", "src/job.py", "--config", "{config}", "--out", "{out}"]
REQUIRED = ["metrics.json", "figure.png", "stdout.log", "stderr.log"]
FIXED_NOW = datetime(
    2026, 1, 2, 3, 4, 5, tzinfo=UTC
)  # injectable clock: bundle ids are reproducible


def git(repo: Path, *args: str) -> str:
    # -c identity flags: tests must not depend on the developer's global git config
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


@dataclass
class Laptop:
    root: Path
    repo: Path
    sha: str

    def write_config(self, name: str = "exp.yaml", **params: object) -> Path:
        """Write + commit a config so the source tree stays clean (citable)."""
        path = self.repo / "configs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps({"experiment_id": "EXP-001", "name": "t", "seed": 7, "params": params})
        )  # JSON is valid YAML; keeps the fixture independent of yaml dump formatting
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", f"config {name}")
        self.sha = git(self.repo, "rev-parse", "HEAD")
        return path

    def prepare(self, config: Path | None = None, **kw: object) -> Path:
        kw.setdefault("entry_point", ENTRY)
        kw.setdefault("required_outputs", REQUIRED)
        kw.setdefault("now", FIXED_NOW)
        return prepare(self.repo, config or self.write_config(), self.root / "bundles", **kw)  # type: ignore[arg-type]


@pytest.fixture
def laptop(tmp_path: Path) -> Laptop:
    root = tmp_path / "laptop"
    repo = root / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "job.py").write_text(JOB)
    (repo / "uv.lock").write_text("# lock\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    return Laptop(root, repo, git(repo, "rev-parse", "HEAD"))


@pytest.fixture
def worker_dir(tmp_path: Path) -> Path:
    return tmp_path / "worker"


def tree_hashes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def run_on_worker(laptop: Laptop, worker: Path, bundle_dir: Path) -> Path:
    """Manual-transfer simulation. Returns the result dir copied back to the laptop."""
    bundle = json.loads((bundle_dir / "bundle.json").read_text())
    shutil.copytree(bundle_dir, worker / "bundle")  # "scp" bundle out
    src = worker / "src_checkout"
    src.mkdir(parents=True)
    tar = subprocess.run(
        ["git", "archive", "--format=tar", bundle["git_commit"]],
        cwd=laptop.repo, capture_output=True, check=True,
    ).stdout  # fmt: skip
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(src, filter="data")  # "clean checkout"
    before = tree_hashes(src)
    out = worker / "scratch" / "out"
    out.mkdir(parents=True)
    argv = resolve_entry(bundle, worker / "bundle" / "config.yaml", out)
    argv[0] = sys.executable
    with (out / "stdout.log").open("wb") as so, (out / "stderr.log").open("wb") as se:
        rc = subprocess.run(argv, cwd=src, stdout=so, stderr=se).returncode
    assert tree_hashes(src) == before, "job modified source on the worker"
    export_result(worker / "bundle", out, rc, started_utc="2026-01-02T03:04:05Z",
                  ended_utc="2026-01-02T03:05:05Z", slurm_job_id="12345")  # fmt: skip
    dest = laptop.root / "incoming" / "result"
    shutil.copytree(out, dest)  # "scp" results back
    return dest


class FakeAttach:
    """Injectable stand-in for `research run attach`: the real Research OS is never touched."""

    def __init__(self) -> None:
        self.calls: list[AttachRequest] = []

    def __call__(self, req: AttachRequest) -> str:
        self.calls.append(req)
        return f"RUN-{len(self.calls):04d}"


@pytest.fixture
def fake_attach() -> FakeAttach:
    return FakeAttach()


@pytest.fixture
def done(laptop: Laptop, worker_dir: Path) -> Path:
    """A completed, exported result dir sitting on the laptop."""
    return run_on_worker(laptop, worker_dir, laptop.prepare())


def rewrite_json(path: Path, **updates: object) -> None:
    d = json.loads(path.read_text())
    d.update(updates)
    path.write_text(json.dumps(d, indent=2))
