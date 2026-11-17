"""Configured table clocks and bounded read/write policies shared by pipeline stages."""

from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data_store.schema import ALL, BY_NAME, KIND_AGGREGATE, KIND_PART, Table, Tables
from src.utils.window import Offset, TimeUnit, Window, WindowPolicy


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


def _offset(value: DictConfig) -> Offset:
    _keys(value, {"value", "unit"})
    return Offset(value.value, cast(TimeUnit, value.unit))


def _keys(value: DictConfig, required: set[str], optional: set[str] | None = None) -> None:
    """Refuse missing and ignored options instead of leaving a second apparent timing owner."""
    actual = set(value) if isinstance(value, DictConfig) else set()
    unknown, missing = actual - required - (optional or set()), required - actual
    if unknown or missing:
        raise ValueError(f"Invalid policy fields: missing={sorted(missing)}; unknown={sorted(unknown)}")


def _columns(table: Table) -> set[str]:
    columns = {*table.pk, *table.date_type_cols, *table.read_columns, *table.retrieval_columns}
    if table.date_col is not None:
        columns.add(table.date_col)
    if table.resume is not None:
        columns.update(x for x in (table.resume.frontier_col, table.resume.period_col) if x is not None)
    return columns


def publication_policy(config: DictConfig, table: Table) -> DictConfig:
    """Resolve a source inheritance chain once; execution timestamps never add a lag."""
    policy = config.data.tables[table.name].pit
    clock = policy.get("clock")
    seen = {table.name}
    while policy.mode == "inherit":
        source = str(policy.source)
        if source not in BY_NAME or source in seen:
            raise ValueError(f"{table.name}: invalid or cyclic publication inheritance: {source}")
        seen.add(source)
        policy = config.data.tables[source].pit
        if clock is None:
            clock = policy.get("clock")
    result = OmegaConf.merge(policy, {"clock": clock}) if clock is not None else OmegaConf.create(OmegaConf.to_container(policy))
    if not isinstance(result, DictConfig):
        raise ValueError(f"{table.name}: publication policy must be a mapping")
    return result


def _validate_publication(config: DictConfig, table: Table, policy: DictConfig) -> None:
    fields = {
        "direct": {"mode", "clock", "lag", "roll"},
        "inherit": {"mode", "source", "clock"},
        "completed_session": {"mode", "clock", "roll"},
        "dated_snapshot": {"mode", "clock", "roll"},
        "half_month_end": {"mode", "clock", "lag", "roll"},
        "notes_release": {"mode", "clock", "filing_lag", "monthly_release_day", "quarterly_lag", "observed_from", "roll"},
        "derived": {"mode", "reason"},
        "not_applicable": {"mode", "reason"},
    }
    if not isinstance(policy, DictConfig) or policy.get("mode") not in fields:
        raise ValueError(f"{table.name}: unsupported publication mode")
    required = fields[policy.mode]
    if policy.mode == "inherit":
        required = required - {"clock"}
    _keys(policy, required, {"clock"} if policy.mode == "inherit" else {"later_clocks"} if policy.mode in {"direct", "notes_release"} else set())
    mode = policy.mode
    clock = policy.get("clock")
    if mode not in {"inherit", "derived", "not_applicable"} and clock is None:
        raise ValueError(f"{table.name}: a source publication clock is required")
    if clock is not None and clock not in _columns(table):
        raise ValueError(f"{table.name}: unknown publication clock {clock}")
    effective = publication_policy(config, table)
    if effective.mode in {"direct", "half_month_end"}:
        lag = _offset(effective.lag)
        if lag.unit != "calendar_days" or lag.value < 1:
            raise ValueError(f"{table.name}: source buffers require positive calendar days")
    if any(column not in _columns(table) for column in effective.get("later_clocks", [])):
        raise ValueError(f"{table.name}: unknown later publication clock")
    if effective.mode == "notes_release":
        filing_lag = _offset(effective.filing_lag)
        if filing_lag.unit != "calendar_days" or filing_lag.value < 1:
            raise ValueError(f"{table.name}: notes require a positive filing buffer")
        if _offset(effective.quarterly_lag).unit != "calendar_days":
            raise ValueError(f"{table.name}: quarterly notes lag must use calendar_days")
        if (
            isinstance(effective.monthly_release_day, bool)
            or not isinstance(effective.monthly_release_day, int)
            or not 1 <= effective.monthly_release_day <= 28
        ):
            raise ValueError(f"{table.name}: monthly release day must be1..28")
        _date(effective.observed_from, f"{table.name}.pit.observed_from")
    if effective.mode in {"direct", "half_month_end", "notes_release", "completed_session", "dated_snapshot"} and effective.roll != "market_session":
        raise ValueError(f"{table.name}: publication must roll to a market session")


def _validate_reprocess(table: Table, item: DictConfig) -> None:
    work = item.reprocess
    modes = {
        "tail": {"overlap"},
        "missing_documents": set(),
        "missing_units": set(),
        "source_changes": {"source"},
        "snapshot": set(),
        "candidate_window": {"lookback"},
        "inherited_max": set(),
        "stage_owned": set(),
        "not_applicable": set(),
    }
    if not isinstance(work, DictConfig) or work.get("mode") not in modes:
        raise ValueError(f"{table.name}: unsupported reprocessing mode")
    optional = {"cover_recovery"} if table == Tables.sec_13f_manager_details else {"archive_fallback"} if table == Tables.sec13f_hr else set()
    _keys(work, {"mode"} | modes[work.mode], optional)
    if "archive_fallback" in work and work.archive_fallback != "missing_units":
        raise ValueError(f"{table.name}: archive fallback must ingest missing units")
    if work.mode == "tail":
        _offset(work.overlap)
    elif "overlap" in work:
        raise ValueError(f"{table.name}: {work.mode} has no numeric overlap")
    if "cover_recovery" in work:
        _keys(work.cover_recovery, {"cached_units"})
        count = work.cover_recovery.cached_units
        if work.mode != "tail" or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{table.name}: cover recovery requires a positive cached-unit count and tail policy")
    if work.mode == "candidate_window":
        _offset(work.lookback)
    if work.mode == "source_changes" and (work.source not in BY_NAME or work.source == table.name):
        raise ValueError(f"{table.name}: unknown changed-source owner")
    if work.mode == "inherited_max" and table.kind != KIND_AGGREGATE:
        raise ValueError(f"{table.name}: inherited maximum is an assembly policy")
    if "warmup" in item:
        if table.kind != KIND_PART:
            raise ValueError(f"{table.name}: warm-up is only applicable to cube parts")
        _keys(item.warmup, {"grid"}, {"sources"})
        grid = item.warmup.grid
        _keys(grid, {"mode", "value", "unit"} if grid.get("mode") == "offset" else {"mode"})
        if grid.mode == "offset":
            _offset(OmegaConf.create({"value": grid.value, "unit": grid.unit}))
        elif grid.mode != "full_history":
            raise ValueError(f"{table.name}: unsupported grid warm-up {grid.mode}")
        for source in item.warmup.get("sources", {}).values():
            _offset(source)
        if "sources" in item.warmup:
            if table != Tables.cube_part_fundamentals:
                raise ValueError(f"{table.name}: no per-source computation context is defined")
            _keys(item.warmup.sources, {"financial_history", "earnings_surprises", "notes_num", "fundamentals_employees"})
    if "lookahead" in item:
        _keys(item.lookahead, {"mode"})
        if table != Tables.cube_part_targets or item.lookahead.mode != "target_horizons":
            raise ValueError(f"{table.name}: future outcomes are only applicable to target horizons")


def _validate_table_dates(config: DictConfig, table: Table, item: DictConfig) -> None:
    required = {"status", "reprocess", "pit"}
    if item.status != "active":
        _keys(item, required)
        mode = "snapshot" if item.status == "snapshot" else "not_applicable" if item.status == "inactive" else "stage_owned"
        if item.reprocess.mode != mode:
            raise ValueError(f"{table.name}: work mode contradicts table status")
        return
    _keys(item, required | {"start_date", "end_date", "source_start", "coverage_start", "freshness"}, {"warmup", "lookahead"})
    start = _date(item.start_date, f"{table.name}.start_date")
    end = None if item.end_date is None else _date(item.end_date, f"{table.name}.end_date")
    if end is not None and start > end:
        raise ValueError(f"{table.name}: end_date precedes start_date")
    for field in ("source_start", "coverage_start"):
        if item[field] is not None:
            _date(item[field], f"{table.name}.{field}")
    if item.freshness is not None:
        fresh = item.freshness
        _keys(fresh, {"clock", "cadence"})
        if fresh.clock not in _columns(table) or fresh.cadence not in config.data.freshness.max_age_days:
            raise ValueError(f"{table.name}: invalid freshness clock or cadence")
    if table.kind == KIND_PART and "warmup" not in item:
        raise ValueError(f"{table.name}: cube parts require explicit computation context")
    if table in {Tables.cube_part_prices, Tables.cube_part_institutionals} and item.warmup.grid.mode != "full_history":
        raise ValueError(f"{table.name}: cumulative anchors and sparse event state require full_history context")
    if table == Tables.cube_part_targets:
        overlap = _offset(item.reprocess.overlap)
        if overlap.unit != "trading_sessions" or overlap.value < max(config.data_aggregate.targets.horizons):
            raise ValueError("Target rewrite must cover the maximum configured forward horizon in trading_sessions")
    if item.reprocess.mode in {"not_applicable", "stage_owned"}:
        raise ValueError(f"{table.name}: active table requires an executable work policy")


def validate_data_policy(config: DictConfig) -> None:
    """Validate all registered table policies before constructing a store or source client."""
    if "data" not in config or "tables" not in config.data:
        raise ValueError("data.yml must define all table policies")
    _keys(config.data, {"planning", "freshness", "derived_feature_starts", "tables"})
    _keys(config.data.planning, {"ended_grace", "lineage_recheck"})
    _keys(config.data.freshness, {"max_age_days", "prediction_min_fresh_share", "insider_max_lag_days"})
    policies = config.data.tables
    if set(policies) != set(BY_NAME):
        raise ValueError(
            f"Table policy inventory differs: missing={sorted(set(BY_NAME) - set(policies))}; unknown={sorted(set(policies) - set(BY_NAME))}"
        )
    for table in ALL:
        item = policies[table.name]
        if not isinstance(item, DictConfig) or item.get("status") not in {"active", "snapshot", "inactive", "model_owned", "portfolio_owned"}:
            raise ValueError(f"{table.name}: unsupported policy status")
        _validate_table_dates(config, table, item)
        _validate_reprocess(table, item)
        _validate_publication(config, table, item.pit)
        source, seen = item, {table.name}
        while source.reprocess.mode == "source_changes":
            name = str(source.reprocess.source)
            if name in seen:
                raise ValueError(f"{table.name}: cyclic changed-source ownership")
            seen.add(name)
            source = policies[name]
    for offset in config.data.planning.values():
        _offset(offset)
    share = config.data.freshness.prediction_min_fresh_share
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 < share <= 1:
        raise ValueError("Prediction minimum fresh share must be in(0,1]")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in config.data.freshness.max_age_days.values()):
        raise ValueError("Freshness tolerances must be nonnegative integer calendar days")
    lag = config.data.freshness.insider_max_lag_days
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise ValueError("Insider freshness lag must be nonnegative integer calendar days")
    for feature, value in config.data.derived_feature_starts.items():
        _date(value, f"data.derived_feature_starts.{feature}")


def source_read_window(config: DictConfig, table: Table, as_of: pd.Timestamp) -> Window | None:
    """Retain the configured source-floor prefix needed by derived histories and identity snapshots."""
    item = config.data.tables[table.name]
    if item.status != "active":
        return None
    start = _date(item.source_start if item.source_start is not None else item.start_date, f"{table.name}.source_start")
    end = pd.Timestamp(as_of).normalize()
    if item.end_date is not None:
        end = min(end, _date(item.end_date, f"{table.name}.end_date"))
    return Window(start, end) if start <= end else None


def window_policy(config: DictConfig, table: Table, as_of: pd.Timestamp) -> WindowPolicy | None:
    """Compile explicit dates, units and dependency context for the shared Window resolver."""
    item = config.data.tables[table.name]
    if item.status != "active":
        return None
    start = _date(item.start_date, f"{table.name}.start_date")
    if item.source_start is not None:
        start = max(start, _date(item.source_start, f"{table.name}.source_start"))
    end = pd.Timestamp(as_of).normalize()
    if item.end_date is not None:
        end = min(end, _date(item.end_date, f"{table.name}.end_date"))
    if end < start:
        return None
    overlap = _offset(item.reprocess.overlap) if item.reprocess.mode == "tail" else None
    warmup: Offset | None = Offset(0, "calendar_days")
    if "warmup" in item:
        grid = item.warmup.grid
        warmup = _offset(OmegaConf.create({"value": grid.value, "unit": grid.unit})) if grid.mode == "offset" else None
    lookahead: Offset | None = None
    if "lookahead" in item:
        if item.lookahead.mode != "target_horizons":
            raise ValueError(f"{table.name}: unsupported future-outcome context")
        lookahead = Offset(max(config.data_aggregate.targets.horizons), "trading_sessions")
    lookback = _offset(item.reprocess.lookback) if item.reprocess.mode == "candidate_window" else None
    return WindowPolicy(Window(start, end), overlap, warmup, lookahead, lookback, item.reprocess.mode == "inherited_max")
