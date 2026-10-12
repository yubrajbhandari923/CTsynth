"""Single home for the content hashes used in provenance (files, configs)."""

import hashlib
from pathlib import Path

from ctsynth.config import ExperimentConfig


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sha256(cfg: ExperimentConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()
