"""Read exact scoped source cells without claiming whole-table or feature certification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.constants.constants_price import MACRO_ALL_SERIES
from src.context import Context
from src.data_store.schema import ALL, RESUME_ARCHIVE, Table, Tables, resolve
from src.data_store.store import KeySample
from src.utils.data_policy import publication_policy
from src.utils.publication import period_end
from src.utils.window import Window
from src.validate.result import CheckResult, Finding, jsonable


def timing_sample_tables() -> tuple[Table, ...]:
    """Extraction and cube tables in the timing audit; prediction/portfolio outputs are excluded."""
    excluded = {Tables.cube_signal, Tables.predictions, Tables.predictions_latest, Tables.trend_asset_returns, Tables.strategy}
    return tuple(table for table in ALL if table not in excluded)


@dataclass(frozen=True)
class SampleSelection:
    """A pinned projection and explicit row scope, independent of the window resolver."""

    columns: tuple[str, ...]
    where: dict[str, Any] | list[dict[str, Any]]
    since: str | None = None
    until: str | None = None
    date_col: str | None = None


def _sample_source(context: Context, table: Table, columns: list[str]) -> tuple[Table, str | None]:
    """Use the configured raw clock, or the inherited filing source for undated companions."""
    item = context.config.data.tables[table.name]
    if item.status == "snapshot":
        return table, None
    clock = item.pit.get("clock") or table.date_col
    if table.resume is not None and table.resume.mode == RESUME_ARCHIVE:
        clock = table.resume.period_col
    if clock in columns or item.pit.mode != "inherit":
        return table, clock
    return resolve(str(item.pit.source)), publication_policy(context.config, table).get("clock")


def capture_window_selection(context: Context, table: Table) -> dict[str, Any]:
    """Plan a fixed-year grouped source selection; incomplete evidence never becomes a smaller passing sample."""
    # Resolve the audit's scope before any payload or model-output access.
    if table not in timing_sample_tables():
        raise ValueError(f"{table.name}: model and portfolio outputs are excluded from timing validation")
    if context.config.data.tables[table.name].status == "inactive":
        return {"reason": "No active producer in the configured pipeline", "applicability": "inactive"}
    options = context.config.validate.window_samples
    start = pd.Timestamp(year=int(options.year), month=1, day=1)
    end = min(context.run.as_of, pd.Timestamp(year=int(options.year), month=12, day=31))
    if start > end:
        return {"reason": "The requested sample year is later than the run date"}
    window = Window(start, end)
    columns = context.store.columns(table)
    if not columns or not set(table.pk) <= set(columns):
        return {"reason": "Source table absent or missing its registered complete primary key"}
    source, clock = _sample_source(context, table, columns)
    source_columns = columns if source == table else context.store.columns(source)
    if clock is not None and clock not in source_columns:
        return {"reason": f"Missing configured source clock {source.name}.{clock}"}

    # A sample searches only scoped entities and the declared source-year clock.
    where: dict[str, Any] = {}
    if source.ticker_col is not None and source.ticker_col in source_columns:
        where[source.ticker_col] = list(MACRO_ALL_SERIES) if source == Tables.prices_macro else context.universe.tickers
    if clock is not None and source.resume is not None and clock == source.resume.period_col:
        year = int(options.year)
        tags = [
            *(f"{year}q{quarter}" for quarter in range(1, 5)),
            *(f"{year}_{month:02d}" for month in range(1, 13)),
            *(f"{year}{month:02d}{half}" for month in range(1, 13) for half in ("a", "b")),
        ]
        where[clock] = [tag for tag in tags if window.start <= pd.Timestamp(period_end(tag)) <= window.end]
    elif clock is not None:
        where[clock] = window
    group = table.ticker_col if table.ticker_col is not None and source == table and table.ticker_col in columns else table.pk[0]
    request = KeySample(group, int(options.keys_per_table), where, clock, source if source != table else None)
    df_keys = context.store.sample_keys(table, request)

    # Pin exact keys and the source clock evidence; payload reads happen in the independent checker.
    scope = {
        "year": int(options.year),
        "requested_keys": request.count,
        "groups": jsonable(df_keys["__sample_group"].tolist()),
        "source_table": source.name,
        "source_clock": clock,
        "source_dates": jsonable(df_keys["__sample_clock"].tolist()),
    }
    if len(df_keys) != request.count or df_keys[list(table.pk)].isna().any(axis=None):
        return {"reason": f"Only {len(df_keys)} of {request.count} required sample groups have complete keys", "sample_scope": scope}
    projection = tuple(column for column in columns if column != table.vector_col)
    scalar_clock = clock if source == table and (source.resume is None or clock != source.resume.period_col) else None
    selection = SampleSelection(projection, jsonable(df_keys[list(table.pk)].to_dict("records")), date_col=scalar_clock)
    return {"selection": asdict(selection), "sample_scope": scope, "excluded_vector": table.vector_col}


def check_window_sample(context: Context, table: Table, *, selection: SampleSelection) -> CheckResult:
    """Check selected persisted keys, clocks and values; missing evidence abstains."""
    # Reject excluded outputs before schema discovery or data access.
    if table not in timing_sample_tables():
        raise ValueError(f"{table.name}: model and portfolio outputs are excluded from timing validation")
    # Load only the pinned projection and scope, including known-empty markers.
    available = context.store.columns(table)
    required = set(table.pk) | set(selection.columns)
    missing = sorted(required - set(available))
    if missing:
        return CheckResult.abstained("window_sample", table.name, f"Missing sample columns: {missing}")
    if not selection.where:
        raise ValueError("A sample needs explicit key/partition scope; an unscoped table read is not a sample")
    if not set(table.pk) <= set(selection.columns):
        raise ValueError(f"{table.name}: the sample projection must include every primary-key column")
    clock = selection.date_col or table.date_col
    if (selection.since is not None or selection.until is not None) and clock not in selection.columns:
        raise ValueError(f"{table.name}: bounded samples must project their clock")
    where: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame = selection.where
    if (
        isinstance(selection.where, dict)
        and set(selection.where) == set(table.pk)
        and all(not isinstance(value, list | tuple | set) for value in selection.where.values())
    ):
        where = pd.DataFrame([selection.where])
    elif isinstance(selection.where, list) and all(set(row) == set(table.pk) for row in selection.where):
        where = pd.DataFrame(selection.where)
    df_rows = context.store.load(
        table,
        columns=selection.columns,
        where=where,
        since=selection.since,
        until=selection.until,
        date_col=clock,
        markers=True,
        optional=True,
        order_by=table.pk,
    )
    if df_rows is None or df_rows.empty:
        return CheckResult.abstained("window_sample", table.name, "No rows in the pinned sample scope")

    # Check grain and boundaries without calling a production timing resolver.
    findings: list[Finding] = []
    bad_keys = df_rows[list(table.pk)].isna().any(axis=1) | df_rows.duplicated(list(table.pk), keep=False)
    if bad_keys.any():
        findings.append(Finding.at(8, f"{int(bad_keys.sum())} invalid/duplicate sample keys", "Non-null unique primary keys"))
    if clock in df_rows.columns:
        dates = pd.to_datetime(df_rows[clock], errors="coerce")
        outside = dates.isna()
        if selection.since is not None:
            outside |= dates < pd.Timestamp(selection.since)
        if selection.until is not None:
            outside |= dates > pd.Timestamp(selection.until)
        if outside.any():
            findings.append(
                Finding.at(8, f"{int(outside.sum())} missing/out-of-scope clocks", "Every sampled row inside the declared inclusive interval")
            )

    # Preserve precision, actual nulls and source provenance for repeatable comparison.
    records = jsonable(df_rows.to_dict(orient="records"))
    identity = {
        "columns": list(df_rows.columns),
        "dtypes": [str(dtype) for dtype in df_rows.dtypes],
        "cells": [[f"{type(value).__name__}:{value!r}" for value in row] for row in df_rows.itertuples(index=False, name=None)],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    return CheckResult.measured(
        "window_sample",
        table.name,
        findings,
        scope={"rows": len(df_rows), "where": jsonable(selection.where), "since": selection.since, "until": selection.until, "clock": clock},
        metrics={"sha256": digest, "nulls": df_rows.isna().sum().to_dict(), "records": records, "identity": identity},
        reason="Scoped persisted-cell evidence only; feature economics and actual producer writes require separate public-run checks.",
    )
