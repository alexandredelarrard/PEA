"""Shared config plumbing for the OFFLINE data_extract tests (tmp_path fake contexts).

`run_manifest._manifest_path` resolves the checkpoint's filename through
`context.config.local.filename.extraction` rather than a module constant, so a fake context
carrying only `paths` raises `AttributeError` before any assertion in the test body runs --
which reads as 17 unrelated failures across the manifest, edgar-driver and DEF 14A suites.

The real `configs/paths.yml` is loaded rather than a hand-written stub so the filename stays in
ONE place: a test that pinned its own copy would keep passing after the config moved the file,
while the pipeline wrote somewhere else entirely.
"""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_PATHS_YML = Path(__file__).resolve().parents[2] / "configs" / "paths.yml"


def extract_config(**branches) -> DictConfig:
    """The config a fake data_extract `Context` needs: the real `local` tree (paths and
    filenames), plus whatever per-test branches the caller adds --
    `extract_config(data_extract={"years_history": 15})`."""
    cfg = OmegaConf.load(_PATHS_YML)
    return OmegaConf.merge(cfg, OmegaConf.create(branches)) if branches else cfg
