"""Shared projected source loading for cube builders."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from src.context import Context
from src.data_store.schema import Table, resolve
from src.utils.data_policy import input_read_window, source_column
from src.utils.publication import period_end
from src.utils.window import Window

logger = logging.getLogger(__name__)


def input_windows(context: Context, part: Table, write: Window) -> dict[Table, Window | None]:
    """Compile each declared payload dependency from the actual output rewrite boundary."""
    return {
        resolve(str(name)): input_read_window(context.config, part, resolve(str(name)), write, context.run.as_of)
        for name in context.config.data.tables[part.name].warmup
    }


def source_read_scope(context: Context, table: Table, window: Window | None) -> dict[str, object]:
    """Push source-clock bounds into storage; archive tags are selected as compact metadata first."""
    clock = source_column(context.config, table)
    if clock is None:
        return {}
    end = min(context.run.as_of, window.end) if window is not None else context.run.as_of
    if clock != "period" or clock in table.date_type_cols:
        return {"since": None if window is None else window.start, "until": end, "date_col": clock}

    # A period string is not a SQL date; normalize only its small inventory before the payload query.
    selected = []
    invalid = []
    for value in context.store.distinct(table, clock):
        try:
            day = pd.Timestamp(period_end(str(value)))
        except (TypeError, ValueError, OverflowError):
            invalid.append(value)
            continue
        if day <= end and (window is None or day >= window.start):
            selected.append(value)
    if invalid:
        logger.warning("%s: excluded %d invalid source period values: %s", table.name, len(invalid), invalid[:5])
    return {"where": {clock: selected}}


def scope_summary(filters: dict[str, object]) -> dict[str, object]:
    """The read scope for the log: a list-valued filter (tickers, archive periods) shows its size, never its members."""
    return {key: f"{len(value)} value(s)" if isinstance(value, Sequence) and not isinstance(value, str) else value for key, value in filters.items()}


def load_source(
    context: Context,
    table: Table,
    universe: Sequence[str] | None = None,
    markers: bool = False,
    columns: Sequence[str] | None = None,
    window: Window | None = None,
) -> pd.DataFrame | None:
    """Stream one projected source payload in its economic scope; `markers` keeps the empty-filing marker rows, `columns` replaces the projection."""
    if not context.store.exists(table):
        logger.warning("%s is absent -> its features are unavailable.", table.name)
        return None
    selected = universe if universe is not None else context.universe.tickers
    filters: dict[str, object] = {table.ticker_col: sorted(set(selected))} if table.ticker_col else {}
    scope = source_read_scope(context, table, window)
    filters.update(scope.pop("where", {}))
    logger.info("Input read %s: window=%s; scope=%s; clock bounds=%s", table.name, window or "full stored history", scope_summary(filters), scope)
    chunks = list(context.store.iter_load(table, project=columns is None, columns=columns, where=filters, markers=markers, **scope))
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    if df is None or df.empty:
        logger.warning("%s is absent or empty -> its features are skipped.", table.name)
        return None
    logger.info("Loaded %s: %s rows x %s cols", table.name, len(df), len(df.columns))
    return df
