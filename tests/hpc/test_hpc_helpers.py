"""Self-tests of the test harness itself (pass without any implementation)."""

import subprocess
import sys
from pathlib import Path

from conftest import ENTRY, Laptop, git

from ctsynth.hpc.bundle import sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "a"
    f.write_bytes(b"abc")
    # NIST test vector for "abc"
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_fixture_repo_is_clean_and_job_runs(laptop: Laptop, tmp_path: Path) -> None:
    cfg = laptop.write_config()
    assert git(laptop.repo, "status", "--porcelain") == ""
    out = tmp_path / "o"
    argv = [sys.executable, "src/job.py", "--config", str(cfg), "--out", str(out)]
    assert ENTRY[1] == "src/job.py"
    subprocess.run(argv, cwd=laptop.repo, check=True, capture_output=True)
    assert (out / "metrics.json").exists() and (out / "figure.png").exists()
