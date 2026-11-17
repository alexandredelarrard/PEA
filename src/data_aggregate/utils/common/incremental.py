"""Resolve cube read/write scopes once and publish bounded, atomic replacements.

Source algorithms retain their own sparse histories. Warm-up expands computation reads,
while only the resolved inclusive write Window and selected keys authorize deletion.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.data_store.schema import Table
from src.data_store.store import DataStore
from src.utils.window import Offset, WindowPlan, WindowPolicy, WindowRun, resolve_window, shift_date

# Legacy effective value; the approved policy migration replaces its owner in P3/P5.
PART_REFRESH_TRADING_DAYS = 7

logger = logging.getLogger(__name__)


def window_start(trading_index: pd.DatetimeIndex, last: pd.Timestamp, n_back: int) -> pd.Timestamp:
    """The date `n_back` trading days BEFORE `last` on the (untrimmed) price calendar."""
    return shift_date(last, Offset(n_back, "trading_sessions"), trading_index)


def plan_window(store: DataStore, part: Table, policy: WindowPolicy, calendar: pd.DatetimeIndex, *, full: bool, scope: Sequence[str]) -> WindowPlan:
    """Read the selected frontier once and pass the resolved scope through to publication."""
    selected = tuple(sorted(set(scope)))
    if not selected:
        raise ValueError(f"{part.name}: cube planning requires an explicit ticker scope")
    last = store.max_date(part, where={"ticker": selected})
    run = (
        WindowRun("full", "explicit full build", policy.requested.end)
        if full
        else WindowRun("initial", "no selected output", policy.requested.end)
        if last is None
        else WindowRun("incremental", "configured overlap", policy.requested.end)
    )
    return resolve_window(policy, run, last, calendar, (("ticker", selected),))


def changed_tickers(store: DataStore, part: Table, df_rows: pd.DataFrame, window: WindowPlan) -> list[str]:
    """Compare the requested saved prefix once; return tickers with changed keys, nulls or exact values."""
    before = window.write.start
    keys = list(part.pk)
    in_scope = pd.to_datetime(df_rows["date"]).between(window.requested.start, before, inclusive="left")
    df_old = df_rows.loc[in_scope & df_rows["ticker"].isin(dict(window.scope)["ticker"])]
    index = pd.MultiIndex.from_arrays([pd.to_datetime(df_old["date"]), df_old["ticker"]])
    fields = [field for field in df_rows if field not in keys]
    values = df_old[fields].to_numpy(dtype="float64")
    seen = np.zeros(len(df_old), dtype=bool)
    changed: set[str] = set()
    stream = store.iter_load(
        part, columns=list(df_rows.columns), where=dict(window.scope), since=window.requested.start, until=pd.Timestamp(before) - pd.Timedelta(days=1)
    )
    try:
        for df_chunk in stream:
            positions = index.get_indexer(pd.MultiIndex.from_arrays([pd.to_datetime(df_chunk["date"]), df_chunk["ticker"]]))
            matched = positions >= 0
            seen[positions[matched]] = True
            old, new = df_chunk.loc[matched, fields].to_numpy(dtype="float64"), values[positions[matched]]
            differs = ~matched
            differs[matched] = ~((old == new) | (np.isnan(old) & np.isnan(new))).all(axis=1)
            changed.update(df_chunk.loc[differs, "ticker"])
    finally:
        stream.close()
    changed.update(df_old.loc[~seen, "ticker"])
    return sorted(changed)


def drop_empty_feature_rows(rows: pd.DataFrame, keys: Sequence[str], part: Table) -> pd.DataFrame:
    """Drop (date, ticker) rows where EVERY feature is NaN.

    The merge-based builders left-join onto the full universe grid, so a name with no
    coverage at all — before its IPO, outside a sector gate, or simply absent from a source
    — still gets a row. Persisting those would store the whole 1.85M-cell grid per part
    regardless of how sparse the features are, and then carry it through the assemble merge.
    """
    fcols = [c for c in rows.columns if c not in set(keys)]
    if not fcols:
        return rows.iloc[0:0]
    keep = rows[fcols].notna().any(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        logger.info("%s: dropped %s all-NaN grid rows (%.1f%% of %s)", part, dropped, 100 * dropped / len(rows), len(rows))
    return rows[keep]


def write_part(
    store: DataStore,
    part: Table,
    rows: pd.DataFrame | None,
    window: WindowPlan,
    *,
    drop_empty: bool = False,
    existing_columns: Sequence[str] | None = None,
) -> int:
    """Publish only the resolved dates and keys, including valid empty corrections."""
    if rows is None:
        logger.warning("%s unavailable -> nothing persisted.", part)
        return 0
    keys = list(part.pk)
    if existing_columns and set(existing_columns) != set(rows.columns) and not window.is_full:
        raise ValueError(f"{part.name}: column set changed; explicit retained-schema rebuild required")
    scope = dict(window.scope)
    selected = window.write.mask(rows["date"])
    for column, values in scope.items():
        selected &= rows[column].isin(values)
    tail = rows.loc[selected]
    if drop_empty:
        tail = drop_empty_feature_rows(tail, keys, part)
    effects = store.replace_window(part, tail, window.write, scope, schema_rebuild=window.is_full)
    logger.info(
        "Persisted %s (%s): %s..%s, %s deleted, %s inserted.",
        part,
        window.run.mode,
        window.write.start.date(),
        window.write.end.date(),
        effects.deleted,
        effects.inserted,
    )
    return effects.inserted
