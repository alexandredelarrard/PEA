"""Shared config plumbing for the OFFLINE data_extract tests (tmp_path fake contexts).

The real `configs/config.yml` and `configs/gpt.yml` are loaded rather than a hand-written stub, so
cache folders, filenames and model defaults stay in one place.
"""

from __future__ import annotations

from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

_PATHS_YML = Path(__file__).resolve().parents[2] / "configs" / "config.yml"
_GPT_YML = Path(__file__).resolve().parents[2] / "configs" / "gpt.yml"


@cache
def _base() -> DictConfig:
    root = _PATHS_YML.parent
    return cast(DictConfig, OmegaConf.merge(*(OmegaConf.load(root / name) for name in ("config.yml", "gpt.yml", "data.yml", "data_extract.yml"))))


def extract_config(**branches) -> DictConfig:
    """The config a fake data_extract `Context` needs: the real `local` tree (paths and
    filenames) and the real `gpt` tree (model / threads / cache defaults), plus whatever
    per-test branches the caller adds -- `extract_config(data_extract={"years_history": 15})`."""
    return cast(DictConfig, OmegaConf.merge(_base(), OmegaConf.create(branches)) if branches else deepcopy(_base()))
