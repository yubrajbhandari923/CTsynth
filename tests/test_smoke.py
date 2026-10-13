import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ctsynth import config, device, runmeta, seeding

SMOKE = Path(__file__).parent.parent / "configs" / "smoke.yaml"


def test_config_loads_and_rejects_typos(tmp_path: Path) -> None:
    cfg = config.load_config(SMOKE)
    assert cfg.seed == 123 and cfg.device == "cpu"
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment_id: EXP-1\nname: x\nsedd: 1\n")
    with pytest.raises(ValidationError):
        config.load_config(bad)


def test_rng_is_deterministic_and_streams_independent() -> None:
    a, b = seeding.rng(7, 0).random(5), seeding.rng(7, 0).random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, seeding.rng(7, 1).random(5))


def test_run_metadata_roundtrip(tmp_path: Path) -> None:
    cfg = config.load_config(SMOKE)
    meta = json.loads(runmeta.write(cfg, tmp_path).read_text())
    assert meta["experiment_id"] == "EXP-000" and len(meta["config_sha256"]) == 64
    # git_dirty covers the whole tree (always true under `research run exec`, which logs as it
    # runs); git_dirty_source covers only src/tests/configs, which is what citability needs.
    assert isinstance(meta["git_dirty_source"], bool)
    # uv.lock hash pins the environment in the evidence (HPC bundle requirement)
    assert meta["uv_lock_sha256"] is not None and len(meta["uv_lock_sha256"]) == 64
    assert runmeta.collect(cfg)["config_sha256"] == meta["config_sha256"]


def test_device_cpu_default_and_unavailable_fails_loudly() -> None:
    assert device.pick_device() == "cpu"
    with pytest.raises((RuntimeError, ValueError)):
        device.pick_device("tpu")


def test_sha256_file_matches_hashlib_and_config_hash_is_stable(tmp_path: Path) -> None:
    import hashlib

    from ctsynth.hashing import sha256_file

    f = tmp_path / "x.bin"
    f.write_bytes(b"ctsynth" * 300_000)  # > 1 MiB: exercises chunking
    assert sha256_file(f) == hashlib.sha256(f.read_bytes()).hexdigest()
    cfg = config.ExperimentConfig(experiment_id="EXP-000", name="t")
    assert config.config_sha256(cfg) == hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()
