"""One validated configuration snapshot per test process, copied before each use."""

from __future__ import annotations

from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.utils.config import read_config


@cache
def _base() -> DictConfig:
    return read_config(str(Path(__file__).resolve().parents[2] / "configs"))


def fixture_config(**branches: Any) -> DictConfig:
    """Clone the real production configuration; fixture mutations never leak."""
    return cast(DictConfig, OmegaConf.merge(_base(), OmegaConf.create(branches)) if branches else deepcopy(_base()))
