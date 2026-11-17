"""Shared projected source loading for cube builders."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from src.context import Context
from src.data_store.schema import Table
from src.utils.data_policy import source_read_window
from src.utils.window import Window

logger = logging.getLogger(__name__)


def load_source(
    context: Context,
    table: Table,
    universe: Sequence[str] | None = None,
    markers: bool = False,
    columns: Sequence[str] | None = None,
    window: Window | None = None,
) -> pd.DataFrame | None:
    """Stream one projected source payload in its economic scope; `markers` keeps the empty-filing marker rows, `columns` replaces the projection."""
    item = context.config.data.tables[table.name]
    if item.status == "inactive":
        return None
    if not context.store.exists(table):
        logger.warning("%s is absent -> its features are unavailable.", table.name)
        return None
    selected = universe if universe is not None else context.universe.tickers
    filters = {table.ticker_col: sorted(set(selected))} if table.ticker_col else {}
    retained = source_read_window(context.config, table, context.run.as_of)
    clock = table.resume.frontier_col if table.resume is not None else table.date_col
    if item.status == "active":
        retained = retained.intersection(window) if retained is not None and window is not None else retained
        if retained is None:
            return None
        if clock is None:
            raise ValueError(f"{table.name}: undated source rows require inherited parent keys")
        filters[clock] = retained
    chunks = list(context.store.iter_load(table, project=columns is None, columns=columns, where=filters, markers=markers))
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    if df is None or df.empty:
        logger.warning("%s is absent or empty -> its features are skipped.", table.name)
        return None
    logger.info("Loaded %s: %s rows x %s cols", table.name, len(df), len(df.columns))
    return df
