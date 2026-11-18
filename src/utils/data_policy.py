"""Validate the single table-policy format and compile explicit calendar-day windows."""

from __future__ import annotations

from datetime import date

import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import ALL, BY_NAME, KIND_EXTRACT, KIND_PART, KIND_REFERENCE, Table, Tables
from src.utils.window import Offset, Window, WindowPolicy


def _date(value: object, label: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO date: {value}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must use YYYY-MM-DD")
    return pd.Timestamp(parsed)


def _keys(value: DictConfig, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, DictConfig):
        raise ValueError("Table policies must be mappings")
    unknown, missing = set(value) - required - (optional or set()), required - set(value)
    if unknown or missing:
        raise ValueError(f"Invalid policy fields: missing={sorted(missing)}; unknown={sorted(unknown)}")


def _amount(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer calendar-day count")


def _columns(table: Table) -> set[str]:
    columns = {*table.pk, *table.date_type_cols, *table.read_columns, *table.retrieval_columns}
    columns.update(column for column in (table.date_col, table.ticker_col) if column is not None)
    if table.resume is not None:
        columns.update(column for column in (table.resume.frontier_col, table.resume.period_col) if column is not None)
    return columns


def publication_policy(config: DictConfig, table: Table) -> DictConfig:
    item = config.data.tables.get(table.name)
    if item is None or "pit" not in item:
        raise ValueError(f"{table.name}: no source publication policy is declared")
    policy = item.pit
    _keys(policy, {"column", "lag_days"})
    return policy


def overlap_column(config: DictConfig, table: Table) -> str:
    item = config.data.tables.get(table.name)
    if item is None or "overlap" not in item:
        raise ValueError(f"{table.name}: no overlap reference is declared")
    return str(item.overlap.column)


def source_column(config: DictConfig, table: Table) -> str | None:
    item = config.data.tables.get(table.name)
    if item is not None and "pit" in item:
        return str(item.pit.column)
    if item is not None and "overlap" in item:
        return str(item.overlap.column)
    return table.date_col


def _validate_overlap(table: Table, overlap: DictConfig) -> None:
    method = overlap.get("method")
    if method not in {"days", "missing_docs", "missing_zips"}:
        raise ValueError(f"{table.name}: unsupported overlap method {method}")
    _keys(overlap, {"method", "column", "days"} if method == "days" else {"method", "column"})
    if overlap.column not in _columns(table):
        raise ValueError(f"{table.name}: unknown overlap column {overlap.column}")
    if method == "days":
        _amount(overlap.days, f"{table.name}.overlap.days")


def _validate_cube(config: DictConfig, table: Table, item: DictConfig) -> None:
    _keys(item, {"warmup"} if table == Tables.cube else {"warmup", "overlap"})
    if not isinstance(item.warmup, DictConfig) or not item.warmup:
        raise ValueError(f"{table.name}: an explicit input warmup map is required")
    for name, amount in item.warmup.items():
        if name not in BY_NAME or (name == table.name and table != Tables.cube_part_fundamentals):
            raise ValueError(f"{table.name}: unknown or cyclic input dependency {name}")
        if amount is not None:
            _amount(amount, f"{table.name}.warmup.{name}")
    if table != Tables.cube:
        _validate_overlap(table, item.overlap)
        if item.overlap.method != "days" or item.overlap.column != table.date_col:
            raise ValueError(f"{table.name}: cube writes require a days overlap on {table.date_col}")
    else:
        assembly = config.data_aggregate.assembly
        selected = set(assembly.compulsory_parts) | set(assembly.optional_parts)
        if set(item.warmup) != selected or any(value != 0 for value in item.warmup.values()):
            raise ValueError("cube: input map must match selected assembly parts with zero additional history")


def _validate_source(config: DictConfig, table: Table, item: DictConfig) -> None:
    _keys(item, {"start_date"}, {"overlap", "pit"})
    _date(item.start_date, f"{table.name}.start_date")
    if "overlap" in item:
        _validate_overlap(table, item.overlap)
    if "pit" in item:
        policy = publication_policy(config, table)
        if policy.column not in _columns(table):
            raise ValueError(f"{table.name}: unknown publication column {policy.column}")
        _amount(policy.lag_days, f"{table.name}.pit.lag_days")
    elif table != Tables.insider_footnotes and (table.resume is None or table.resume.mode != "snapshot"):
        raise ValueError(f"{table.name}: dated sources require an explicit publication policy")


def validate_data_policy(config: DictConfig) -> None:
    """Reject obsolete or ignored settings before source clients or stores are constructed."""
    if "data" not in config:
        raise ValueError("data.yml must define table policies")
    _keys(config.data, {"tables"})
    for setting in ("listing_end_grace_sessions", "identity_change_recheck_days"):
        value = config.data_extract[setting]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"data_extract.{setting} must be a nonnegative integer")
    expected = {
        table.name
        for table in ALL
        if table.kind == KIND_PART or table == Tables.cube or (table.kind in {KIND_EXTRACT, KIND_REFERENCE} and table.date_col is not None)
    } | {Tables.insider_footnotes.name, Tables.earning_calls_embedding.name}
    actual = set(config.data.tables)
    if actual != expected:
        raise ValueError(f"Table policy inventory differs: missing={sorted(expected - actual)}; unknown={sorted(actual - expected)}")
    for name, item in config.data.tables.items():
        table = BY_NAME[name]
        if table.kind == KIND_PART or table == Tables.cube:
            _validate_cube(config, table, item)
            continue
        _validate_source(config, table, item)


def source_read_window(config: DictConfig, table: Table, as_of: pd.Timestamp) -> Window | None:
    """The extraction work bounds; cube payload reads use their explicit input map instead."""
    item = config.data.tables.get(table.name)
    if item is None or "start_date" not in item:
        return None
    start, end = _date(item.start_date, f"{table.name}.start_date"), pd.Timestamp(as_of).normalize()
    return Window(start, end) if start <= end else None


def window_policy(config: DictConfig, table: Table, as_of: pd.Timestamp, calendar: pd.DatetimeIndex | None = None) -> WindowPolicy | None:
    """Compile extraction bounds or observed cube price dates, with one write overlap."""
    item = config.data.tables.get(table.name)
    if item is None:
        return None
    if table.kind == KIND_PART or table == Tables.cube:
        if calendar is None or calendar.empty:
            raise ValueError(f"{table.name}: observed price dates are required to plan cube outputs")
        available = calendar[calendar <= pd.Timestamp(as_of).normalize()]
        if available.empty:
            return None
        requested = Window(available[0], available[-1])
    else:
        requested = source_read_window(config, table, as_of)
        if requested is None:
            return None
    overlap = item.get("overlap")
    offset = Offset(int(overlap.days), "calendar_days") if overlap is not None and overlap.method == "days" else None
    return WindowPolicy(requested, offset, Offset(0, "calendar_days"), inherited=table == Tables.cube)


def input_read_window(config: DictConfig, cube: Table, source: Table, write: Window, as_of: pd.Timestamp) -> Window | None:
    """Additional calendar history anchored at the actual rewrite; null explicitly means full."""
    mapping = config.data.tables[cube.name].warmup
    if source.name not in mapping:
        raise ValueError(f"{cube.name}: missing warmup declaration for {source.name}")
    amount = mapping[source.name]
    if amount is None:
        return None
    _amount(amount, f"{cube.name}.warmup.{source.name}")
    return Window(write.start - pd.Timedelta(days=amount), pd.Timestamp(as_of).normalize())
