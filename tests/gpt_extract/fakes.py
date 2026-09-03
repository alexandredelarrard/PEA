"""Shared fakes for the gpt_extract suite: a config, a context and a stub provider.

Nothing here touches the network or the store, so the whole threaded path can be exercised
at zero spend.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path
from typing import Sequence

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

_GPT_YML = Path(__file__).resolve().parents[2] / "configs" / "gpt.yml"


class Answer(BaseModel):
    """A tiny schema, so the tests assert on plumbing rather than on a 16k-char schema."""

    name: str = ""
    value: int = 0


def gpt_config(**overrides) -> DictConfig:
    """The real `configs/gpt.yml`, so a config drift breaks a test instead of the run."""
    cfg = OmegaConf.load(_GPT_YML)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"gpt": overrides}))
    return cfg


def fake_context(store=None):
    return types.SimpleNamespace(
        store=store, log=logging.getLogger("test"), config_dir=Path("."),
    )


class StubProvider:
    """A `_Provider` that answers from a canned list. No network, no key, no spend."""

    name = "stub"
    structured = True

    def __init__(self, model: str = "stub-model", api_key: str | None = None,
                 answers=None, fail_on=None, delay=0.0, **_) -> None:
        self.model = model
        self.api_key = api_key
        self.calls: list[tuple[str, str]] = []
        self._answers = answers
        self._fail_on = set(fail_on or ())
        self._delay = delay

    def parse(self, schema, system: str, user: str):
        import time

        self.calls.append((system, user))
        if self._delay:
            time.sleep(self._delay)
        for marker in self._fail_on:
            if marker in user:
                raise RuntimeError(f"stub failure on {marker}")
        parsed = schema(name=user[-40:], value=len(user))
        return parsed, {"input_tokens": 100, "output_tokens": 10, "cached_input_tokens": 40}

    def embed(self, texts: Sequence[str], model: str):
        return [[float(len(t)), 1.0, 0.0] for t in texts]


@pytest.fixture
def answer_schema():
    return Answer
