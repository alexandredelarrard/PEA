"""Shared fixtures for the data_extract tests."""

from __future__ import annotations

import pytest

from src.data_extract.utils.common import sec_io


@pytest.fixture(autouse=True)
def _sec_io_without_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh `sec_io` state per test whose retry waits are skipped: the policy (3 attempts) still
    applies, but no test sleeps between SEC retries and no configured policy leaks between tests."""
    monkeypatch.setattr(sec_io, "_STATE", sec_io._State(sleep=lambda seconds: None))
