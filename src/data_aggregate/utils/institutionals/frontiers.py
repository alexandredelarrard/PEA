"""Completeness-frontier resolution for institutional source families."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.context import Context
from src.data_store.schema import Table


def _normalized_timestamp(value: Any) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value)).normalize()


def schedule_complete_through(
    context: Context,
    log: logging.Logger,
    table: Table,
    *,
    expected_tickers: Sequence[str],
) -> pd.Timestamp | None:
    """Trust only a complete manifest frontier for the analysis universe."""
    path = Path(context.paths["DATA_STORE"]) / Path(context.config.local.filename.extraction)
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        entry: dict[str, Any] = payload.get(table.name) or {}
    except (OSError, ValueError, TypeError):
        log.warning("%s has no readable extraction manifest frontier", table.name)
        return None
    if entry.get("coverage_complete") is not True:
        log.warning(
            "%s manifest predates completeness-sensitive subject discovery; known events remain usable, but absence cannot be emitted as zero",
            table.name,
        )
        return None
    expected = sorted(set(map(str, expected_tickers)))
    if entry.get("tickers") != expected:
        log.warning(
            "%s manifest ticker membership does not match the %s-name analysis universe; zero semantics disabled",
            table.name,
            len(expected),
        )
        return None
    try:
        return _normalized_timestamp(entry["last_run_date"])
    except (KeyError, TypeError, ValueError):
        log.warning("%s completeness manifest has no valid last_run_date", table.name)
        return None
