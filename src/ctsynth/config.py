"""Structured experiment configuration: YAML -> validated, immutable pydantic model."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)  # typos in YAML must fail loudly

    experiment_id: str  # Research OS ID, e.g. EXP-001
    name: str
    seed: int = 0
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    params: dict[str, float | int | str | bool] = {}


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))
