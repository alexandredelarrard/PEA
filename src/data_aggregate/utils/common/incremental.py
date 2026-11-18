"""Resolve cube read/write scopes once and publish bounded, atomic replacements.

Source algorithms retain their own sparse histories. Warm-up expands computation reads,
while only the resolved inclusive write Window and selected keys authorize deletion.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from src.context import Context
from src.data_aggregate.utils.common.parts import PART_BY_NAME
from src.data_aggregate.utils.price.returns import MOMENTUM_SEAM_WINDOW
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.data_policy import window_policy
from src.utils.window import Offset, Window, WindowPlan, WindowRun, resolve_window, shift_date

logger = logging.getLogger(__name__)


def plan_window(
    context: Context,
    part: Table,
    calendar: pd.DatetimeIndex,
    scope: Sequence[str] | None = None,
    *,
    inherited_start: pd.Timestamp | None = None,
) -> WindowPlan | None:
    """Resolve the configured cube policy against one scoped frontier and the root run."""
    policy = window_policy(context.config, part, context.run.as_of)
    selected = tuple(sorted(set(context.universe.tickers if scope is None else scope)))
    if not selected:
        raise ValueError(f"{part.name}: cube planning requires an explicit ticker scope")
    if policy is None:
        return None
    if calendar.empty:
        raise ValueError(f"{part.name}: a retained market calendar is required")
    end = min(policy.requested.end, calendar[-1])
    if end < policy.requested.start:
        return None
    last = context.store.max_date(part, where={"ticker": selected, "date": policy.requested})
    if last is not None and pd.Timestamp(last) > calendar[-1]:
        raise RuntimeError(f"{part.name}: source calendar ends before the saved frontier; no rows may be replaced")

    # Available source dates may narrow the hard ceiling; they never replace its policy.
    policy = replace(policy, requested=Window(policy.requested.start, end))
    run = context.run
    if last is None and run.mode == "incremental":
        run = WindowRun("initial", "no selected output", run.instant)
    plan = resolve_window(policy, run, last, calendar, (("ticker", selected),), inherited_start=inherited_start)
    part_info = PART_BY_NAME.get(part.name)
    required = max((value for _, value in part_info.binding_lookbacks), default=0) if part_info is not None else 0
    if part in {Tables.cube_part_targets, Tables.cube_part_betas}:
        required = max(required, MOMENTUM_SEAM_WINDOW[0] + int(context.config.data_aggregate.betas.window))
    if part == Tables.cube_part_fundamentals:
        required = max(required, int(context.config.data_aggregate.hist.window))
    insufficient = policy.warmup is not None and policy.warmup.unit == "trading_sessions" and policy.warmup.value < required
    if insufficient or plan.read.start > shift_date(plan.write.start, Offset(required, "trading_sessions"), calendar):
        raise ValueError(f"{part.name}: configured warm-up does not cover its {required}-session calculation dependency")
    return plan


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
