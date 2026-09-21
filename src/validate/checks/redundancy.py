"""Stub -- implemented in a later phase of the generic-check-library plan."""
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from src.context import Context
from src.data_store.schema import Table
from src.validate.result import CheckResult


def check_redundancy(context: Context, table: Table | str, *, config: DictConfig,
             cache: Any = None, tickers: list[str] | None = None, **kwargs: Any) -> CheckResult:
    raise NotImplementedError("redundancy")
