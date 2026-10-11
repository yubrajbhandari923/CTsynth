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
    assert runmeta.collect(cfg)["config_sha256"] == meta["config_sha256"]


def test_device_cpu_default_and_unavailable_fails_loudly() -> None:
    assert device.pick_device() == "cpu"
    with pytest.raises((RuntimeError, ValueError)):
        device.pick_device("tpu")
