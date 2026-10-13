"""The XCIST HPC scripts must resolve the env path from the environment, not from a user name.

No cluster is contacted: these only execute the path helper and `bash -n` (syntax) the scripts.
"""

import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _resolve(home: str, override: str | None) -> str:
    env = {"HOME": home, "PATH": "/usr/bin:/bin"}
    if override is not None:
        env["CTSYNTH_XCIST_ENV"] = override
    out = subprocess.run(
        ["bash", "-c", f'source "{SCRIPTS}/xcist_env_path.sh"; printf %s "$CTSYNTH_XCIST_ENV"'],
        capture_output=True, text=True, check=True, env=env,
    )  # fmt: skip
    return out.stdout


def test_default_is_under_home() -> None:
    assert _resolve("/home/someone", None) == "/home/someone/research-os/projects/CTsynth/env"


def test_override_wins() -> None:
    assert _resolve("/home/someone", "/scratch/x/env") == "/scratch/x/env"


@pytest.mark.parametrize("name", ["hpc_setup_xcist_env.sh", "hpc_xcist_replicate.sh"])
def test_scripts_parse_and_source_the_helper(name: str) -> None:
    subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)
    text = (SCRIPTS / name).read_text()
    assert "xcist_env_path.sh" in text and "CTSYNTH_XCIST_ENV" in text


def test_no_maintained_script_hardcodes_a_user_path() -> None:
    for p in SCRIPTS.iterdir():
        if p.suffix in {".sh", ".py"} and p.name not in {"hpc_smoke.py", "hpc_env_probe.py"}:
            assert "/home/yb107" not in p.read_text(), p.name
