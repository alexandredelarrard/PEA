"""Shared config plumbing for the OFFLINE data_extract tests (tmp_path fake contexts).

The real `configs/paths.yml` and `configs/gpt.yml` are loaded rather than a hand-written stub, so
cache folders, filenames and model defaults stay in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

_PATHS_YML = Path(__file__).resolve().parents[2] / "configs" / "paths.yml"
_GPT_YML = Path(__file__).resolve().parents[2] / "configs" / "gpt.yml"


def extract_config(**branches) -> DictConfig:
    """The config a fake data_extract `Context` needs: the real `local` tree (paths and
    filenames) and the real `gpt` tree (model / max_chars / cache defaults), plus whatever
    per-test branches the caller adds -- `extract_config(data_extract={"years_history": 15})`."""
    cfg = OmegaConf.merge(OmegaConf.load(_PATHS_YML), OmegaConf.load(_GPT_YML))
    return cast(DictConfig, OmegaConf.merge(cfg, OmegaConf.create(branches)) if branches else cfg)
