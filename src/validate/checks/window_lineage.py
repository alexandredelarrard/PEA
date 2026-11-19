"""Independent source-clock and assembled-state verification; no producer imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pandas._libs.tslibs.nattype import NaTType

from src.constants.constants import SEC_NET_PENSION_LIABILITY_TAGS
from src.context import Context
from src.data_store.schema import Table, Tables, resolve
from src.validate.checks.window_sample import timing_sample_tables
from src.validate.frame import archive_period_end
from src.validate.result import CheckResult, Finding


@dataclass
class _State:
    table: Table
    part: Table
    rows: pd.DataFrame
    field: str
    daily: pd.DataFrame
    clock: str = "as_of"
    age_days: int | None = None
    lifetime_sessions: int | None = None
    exception: str | None = None


def _day(value: Any) -> pd.Timestamp | NaTType:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT
    if pd.isna(stamp):
        return pd.NaT
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    return stamp.normalize()


def expected_publication(policy: dict[str, Any], row: pd.Series | dict[str, Any], sessions: pd.DatetimeIndex) -> pd.Timestamp | NaTType:
    """Independent policy oracle; deliberately does not call Window/publication arithmetic."""
    column, lag = str(policy["column"]), policy["lag_days"]
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise ValueError("Publication lag must be a nonnegative integer calendar-day amount")
    value = row[column]
    try:
        anchor = archive_period_end(value) if column == "period" else _day(value)
    except (ValueError, TypeError, OverflowError):
        return pd.NaT
    anchor += pd.Timedelta(days=lag)
    # Iteration is intentionally independent of production searchsorted/roll_to_session.
    if isinstance(anchor, NaTType):
        return pd.NaT
    eligible = sessions[sessions >= anchor]
    return eligible[0] if len(eligible) else pd.NaT


def _source_row(
    state: _State,
    row: pd.Series,
    sessions: pd.DatetimeIndex,
    policies: dict[Table, dict[str, Any]],
    dates: dict[tuple[Any, ...], pd.Timestamp | NaTType],
) -> tuple[dict[str, Any], pd.Timestamp | NaTType]:
    policy = policies[state.table]
    clock = str(policy["column"])
    names = {*state.table.pk, "adsh", "accession_number", "quarter", "period"}
    for item in policies.values():
        names.add(str(item["column"]))
    raw = {name: row.get(f"_source_{name}", row.get(name)) for name in names if name in row or f"_source_{name}" in row}
    ready = {}
    for source, item in policies.items():
        evidence = dict(raw)
        if source == Tables.sec_fundamentals_history and "_sec_source_as_of" in row:
            evidence["as_of"] = row["_sec_source_as_of"]
        column = str(item["column"])
        key = (source, str(evidence.get(column)))
        if key not in dates:
            dates[key] = expected_publication(item, evidence, sessions)
        ready[source] = dates[key]
        if source == state.table:
            raw = evidence
    expected = ready[state.table]
    if "_projection_parent_asof" in row:
        expected = max(expected, _day(row["_projection_parent_asof"]))
    # The mixed-history producer gates the whole joined row on both source policies.
    if state.table in {Tables.sharadar_fundamentals_history, Tables.sec_fundamentals_history}:
        clocks = [ready[Tables.sharadar_fundamentals_history]]
        if any(pd.notna(row[name]) for name in row.index if name.startswith("sec_")):
            clocks.append(ready[Tables.sec_fundamentals_history])
        expected = max(clocks) if all(pd.notna(day) for day in clocks) else pd.NaT
    identifiers = {name: raw[name] for name in state.table.pk if name in raw}
    identifiers.update({name: raw[name] for name in ("adsh", "accession_number", "quarter", clock) if name in raw})
    if "_source_units" in row:
        identifiers["complete_source_units_and_served_day_witnesses"] = row["_source_units"]
    if "_selected_accessions" in row:
        identifiers["actual_selected_accessions"] = row["_selected_accessions"]
        units = row["_source_units"]
        if set(row["_selected_accessions"]) != {unit["accession_number"] for unit in units}:
            raise ValueError("Actual selected book references do not match their raw filing ancestry")
        expected = max(expected_publication(policy, unit, sessions) for unit in units)
    return identifiers, expected


def _boundaries(config: DictConfig, tickers: list[str], state: _State, sessions: pd.DatetimeIndex) -> tuple[list[dict[str, Any]], list[Finding]]:
    records: list[dict[str, Any]] = []
    findings: list[Finding] = []
    year = int(config.validate.window_samples.year)
    tables = (
        (Tables.sharadar_fundamentals_history, Tables.sec_fundamentals_history)
        if state.table in {Tables.sharadar_fundamentals_history, Tables.sec_fundamentals_history}
        else (state.table,)
    )
    policies = {table: dict(cast(dict[str, Any], OmegaConf.to_container(config.data.tables[table.name].pit, resolve=True))) for table in tables}
    dates: dict[tuple[Any, ...], pd.Timestamp | NaTType] = {}
    for ticker in tickers:
        observed, ticker_findings = _ticker_boundaries(state, ticker, sessions, policies, dates, year)
        records.extend(observed)
        findings.extend(ticker_findings)
    return records, findings


def _ticker_boundaries(
    state: _State,
    ticker: str,
    sessions: pd.DatetimeIndex,
    policies: dict[Table, dict[str, Any]],
    dates: dict[tuple[Any, ...], pd.Timestamp | NaTType],
    year: int,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    records: list[dict[str, Any]] = []
    findings: list[Finding] = []
    keep_null = state.table in {Tables.sec_8k_votes, Tables.sec_fails_to_deliver_security}
    df_rows = state.rows.loc[state.rows.ticker.eq(ticker) & (state.rows[state.field].notna() | keep_null)].copy()
    if state.table == Tables.dividends:
        # The producer still computes its entire history, including observed
        # zero days. Only actual cash events contribute to a TTM cash value.
        df_rows = df_rows.loc[pd.to_numeric(df_rows[state.field], errors="coerce").ne(0)]
    df_rows = df_rows.sort_values("as_of", kind="stable")
    if state.table not in {Tables.insider_transactions, Tables.sec_13d, Tables.sec_13g}:
        df_rows = df_rows.drop_duplicates("as_of", keep="last")
    df_units = state.rows.loc[state.rows.ticker.eq(ticker)].sort_values("as_of", kind="stable")
    df_candidate_rows = df_units if df_rows.empty and state.table == Tables.notes_num else df_rows
    df_candidates = df_candidate_rows.loc[pd.to_datetime(df_candidate_rows.as_of).dt.year.eq(year)]
    if df_candidates.empty and state.table in {Tables.notes_num, Tables.dividends}:
        df_candidates = df_units.loc[pd.to_datetime(df_units.as_of).dt.year.eq(year)]
    if df_candidates.empty:
        return records, findings
    event = df_candidates.iloc[0]
    event_identity, event_ready = _source_row(state, event, sessions, policies, dates)
    if pd.isna(event_ready):
        return records, findings
    if pd.Timestamp(event.as_of) != event_ready and state.table != Tables.insider_transactions:
        findings.append(
            Finding.at(
                8,
                "Observed boundary unit has wrong producer usability",
                "Configured source clock and session roll",
                source=event_identity,
                actual=event.as_of,
                expected_date=event_ready,
            )
        )
    # The replay calendar is validated unique, so every retained date has one position.
    position = cast(int, sessions.get_loc(pd.Timestamp(event.as_of)))
    observed, clock_findings = _observations(state, df_rows, sessions, policies, dates)
    findings.extend(clock_findings)
    for day in sessions[max(0, position - 1) : position + 2]:
        eligible = [item for item in observed if pd.notna(item[0]) and item[0] <= day]
        actual = cast(float | np.integer | np.floating, state.daily.reindex(index=[day], columns=[ticker]).iloc[0, 0])
        expected_value, identity, ready = _expected_state(state, eligible, day, sessions)
        same = bool(pd.isna(actual) and pd.isna(expected_value)) or bool(np.isclose(actual, expected_value, rtol=1e-12, atol=0, equal_nan=True))
        if not same:
            findings.append(
                Finding.at(
                    8,
                    "Production daily state differs from independently eligible source",
                    "Latest eligible whole observation only",
                    ticker=ticker,
                    field=state.field,
                    date=day,
                    source=identity,
                    ready=ready,
                    actual=actual,
                    expected_value=expected_value,
                )
            )
        records.append(
            {
                "ticker": ticker,
                "date": day,
                "source": identity,
                "usable_on": ready,
                "value": actual,
                "event_source": event_identity,
                "event_usable_on": event_ready,
                "known_empty": df_rows.empty,
            }
        )
    return records, findings


def _expected_state(
    state: _State, eligible: list[tuple[pd.Timestamp | NaTType, pd.Series, dict[str, Any]]], day: pd.Timestamp, sessions: pd.DatetimeIndex
) -> tuple[float, Any, pd.Timestamp | NaTType | None]:
    selected = max(eligible, key=lambda item: item[0]) if eligible else None
    value, identity, ready = np.nan, None, None
    if selected is not None:
        ready, row, identity = selected
        alive = state.age_days is None or day <= ready + pd.Timedelta(days=state.age_days)
        if state.table == Tables.earnings_surprises:
            assert state.age_days is not None
            alive = day <= _day(row.earnings_date) + pd.Timedelta(days=state.age_days)
        if state.lifetime_sessions is not None:
            alive = cast(int, sessions.get_loc(day)) < cast(int, sessions.get_loc(ready)) + state.lifetime_sessions
        if alive:
            value = row[state.field]
    contributors = None
    if state.table == Tables.dividends:
        contributors = [
            item
            for item in eligible
            if _day(item[1].date) in sessions and cast(int, sessions.get_loc(day)) < cast(int, sessions.get_loc(_day(item[1].date))) + 252
        ]
    if state.table in {Tables.sec_13d, Tables.sec_13g}:
        contributors = [item for item in eligible if cast(int, sessions.get_loc(day)) < cast(int, sessions.get_loc(item[0])) + 126]
    if state.table == Tables.insider_transactions:
        contributors = [
            item
            for item in eligible
            if day < _day(item[1].anchor) + pd.Timedelta(days=20) and (pd.isna(_day(item[1].visible_until)) or day < _day(item[1].visible_until))
        ]
    if contributors is not None:
        value = sum(float(item[1][state.field]) for item in contributors)
        if state.table == Tables.insider_transactions and not eligible:
            value = np.nan
        identity = [item[2] for item in contributors if item[1][state.field]]
    return pd.to_numeric(value, errors="coerce"), identity, ready


def _observations(
    state: _State,
    df_rows: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    policies: dict[Table, dict[str, Any]],
    dates: dict[tuple[Any, ...], pd.Timestamp | NaTType],
) -> tuple[list[tuple[pd.Timestamp | NaTType, pd.Series, dict[str, Any]]], list[Finding]]:
    observed = []
    findings = []
    for _, row in df_rows.iterrows():
        identity, expected = _source_row(state, row, sessions, policies, dates)
        insider = state.table == Tables.insider_transactions
        observed.append((pd.Timestamp(row.as_of) if insider else expected, row, identity))
        if pd.isna(expected) or (pd.Timestamp(row.as_of) < expected if insider else pd.Timestamp(row.as_of) != expected):
            findings.append(
                Finding.at(
                    8,
                    "Producer usability differs from independent source clock",
                    "Exactly one configured buffer and actual market-session roll",
                    ticker=str(row.ticker),
                    field=state.field,
                    source=identity,
                    actual=row.as_of,
                    expected_date=expected,
                )
            )
    return observed, findings


def route_family(table: Table, field: str) -> str:
    """Stable semantic routes; sparse field alternatives are declared here, not inferred from coverage."""
    variants = {
        Tables.sec_fundamentals_history: {"sec_minorityInterest", "sec_operatingLeaseLiability", "sec_financeLeaseLiability"},
        Tables.notes_num: {"DefinedBenefitPlanBenefitObligation", "DefinedBenefitPlanFairValueOfPlanAssets", *SEC_NET_PENSION_LIABILITY_TAGS},
    }
    if table in variants and field not in variants[table]:
        raise ValueError(f"Undeclared sparse source-state alternative {table.name}.{field}")
    return table.name


def _required_routes() -> dict[str, tuple[Table, Table]]:
    tables = [
        Tables.sharadar_fundamentals_history,
        Tables.sec_fundamentals_history,
        Tables.earnings_surprises,
        Tables.def14a_llm,
        Tables.notes_num,
        Tables.earnings_call_sentiment,
        Tables.dividends,
        Tables.fundamentals_employees,
        Tables.sec_8k_votes,
        Tables.insider_transactions,
        Tables.sec_13d,
        Tables.sec_13g,
    ]
    result = {table.name: (table, table) for table in tables}
    for table in (Tables.sec_short_volume_security, Tables.sec_fails_to_deliver_security):
        result[table.name] = (table, Tables.cube_part_institutionals)
    return result


def _requested(manifest: dict[str, Any], table: Table) -> set[str]:
    item = manifest.get("tables", {}).get(table.name, {})
    return set(item.get("sample_scope", {}).get("groups", []))


def _entry_routes(entry: dict[str, Any], table: Table) -> list[str]:
    if table not in timing_sample_tables():
        raise ValueError("Model and portfolio outputs are excluded from lineage validation")
    if "field" in entry and entry.get("route_id", route_family(table, entry["field"])) != route_family(table, entry["field"]):
        raise ValueError("Replay field does not belong to its declared semantic route")
    names = [entry["route_id"]] if "route_id" in entry else [name for name, (source, _) in _required_routes().items() if source == table]
    for name in names:
        if name not in _required_routes() or _required_routes()[name][0] != table:
            raise ValueError(f"Undeclared semantic route {name}")
    return names


def check_window_lineage(context: Context, replay: dict[str, Any], manifest: dict[str, Any]) -> dict[str, CheckResult]:
    """Verify each required route against immutable requested groups; FAIL dominates missing evidence."""
    sessions = pd.DatetimeIndex(pd.to_datetime(replay["sessions"]))
    if sessions.hasnans or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("Replay requires a complete ordered unique market calendar")
    families: dict[str, list[CheckResult]] = {name: [] for name in _required_routes()}
    for entry in replay["routes"]:
        table = resolve(entry["table"])
        for name in _entry_routes(entry, table):
            requested = _requested(manifest, _required_routes()[name][1])
            families[name].append(_check_route(context, entry, requested, sessions))
    combined = {
        name: _combine_routes(name, rows, _requested(manifest, table), int(context.config.validate.window_samples.keys_per_table))
        for name, rows in families.items()
        for _, table in [_required_routes()[name]]
    }
    results = {}
    for table in timing_sample_tables():
        required = [result for name, result in combined.items() if _required_routes()[name][0] == table]
        results[table.name] = _combine_table(table.name, required)
    return results


def _check_route(context: Context, entry: dict[str, Any], requested: set[str], sessions: pd.DatetimeIndex) -> CheckResult:
    table = resolve(entry["table"])
    findings = []
    if "failure" in entry:
        findings.append(Finding.at(8, entry["failure"], "Production replay succeeds"))
    if "reason" in entry or findings:
        return CheckResult(
            "window_lineage",
            table.name,
            "fail" if findings else "abstain",
            entry.get("reason", "Replay failed"),
            findings=findings,
            metrics={"field": entry.get("field")},
        )
    if set(entry.get("requested_tickers", [])) != requested:
        findings.append(
            Finding.at(
                8,
                "Replay changed the immutable requested groups",
                "Manifest groups remain fixed",
                actual=entry.get("requested_tickers"),
                pinned=sorted(requested),
            )
        )
    df_rows = pd.DataFrame(entry["rows"])
    df_daily = pd.DataFrame(entry["daily"])
    df_daily["date"] = pd.to_datetime(df_daily.date)
    df_daily = df_daily.pivot(index="date", columns="ticker", values="value").apply(pd.to_numeric, errors="coerce")
    state = _State(
        table,
        resolve(entry["part"]),
        df_rows,
        entry["field"],
        df_daily,
        entry["clock"],
        entry["age_days"],
        entry["lifetime_sessions"],
        entry["exception"],
    )
    records, boundary_findings = _boundaries(context.config, sorted(requested), state, sessions)
    findings.extend(boundary_findings)
    findings.extend(_assembly_findings(entry, records))
    return CheckResult.measured(
        "window_lineage",
        table.name,
        findings,
        scope={"tickers": sorted({row["ticker"] for row in records})},
        metrics={
            "source_rows": len(df_rows),
            "dependency_scope": entry.get("dependency_scope"),
            "field": state.field,
            "boundaries": records,
            "assembly": entry["assembly"],
            "retrospective_exception": state.exception,
        },
    )


def _combine_routes(name: str, results: list[CheckResult], requested: set[str], count: int) -> CheckResult:
    findings = [finding for result in results for finding in result.findings]
    covered = {ticker for result in results for ticker in result.scope.get("tickers", [])}
    missing = any(
        result.status == "abstain"
        and (name not in {Tables.sec_fundamentals_history.name, Tables.notes_num.name} or result.metrics.get("field") is None)
        for result in results
    )
    status = (
        "fail"
        if any(result.status == "fail" for result in results)
        else "pass"
        if len(requested) == count and covered == requested and not missing
        else "abstain"
    )
    return CheckResult(
        "window_lineage",
        name,
        status,
        "Required semantic route; only predeclared sparse alternatives within this family may pool coverage",
        scope={"required": True, "requested_tickers": sorted(requested), "tickers": sorted(covered)},
        findings=findings,
        metrics={"variants": [asdict(result) for result in results]},
    )


def _combine_table(name: str, required: list[CheckResult]) -> CheckResult:
    status = (
        "fail"
        if any(result.status == "fail" for result in required)
        else "abstain"
        if not required or any(result.status == "abstain" for result in required)
        else "pass"
    )
    return CheckResult(
        "window_lineage",
        name,
        status,
        "Required routes checked independently; selected-state/packet eligibility and diagnostic assembly only, not saved economic feature or peer parity",
        findings=[finding for result in required for finding in result.findings],
        metrics={"required_routes": [asdict(result) for result in required]},
    )


def _assembly_findings(entry: dict[str, Any], records: list[dict[str, Any]]) -> list[Finding]:
    df_assembled = pd.DataFrame(entry["assembly"]["assembled"])
    df_assembled["date"] = pd.to_datetime(df_assembled.date)
    column = entry["assembly"]["column"]
    df_joined = pd.DataFrame(records).merge(df_assembled, on=["date", "ticker"], how="outer", validate="one_to_one", indicator=True)
    same = df_joined._merge.eq("both").all() and np.allclose(
        pd.to_numeric(df_joined.value).astype("float32"), pd.to_numeric(df_joined[column]), rtol=0, atol=0, equal_nan=True
    )
    findings = []
    if not same:
        findings.append(
            Finding.at(8, "Assembly changed/dropped a selected state boundary", "Exact keys, values and nulls at each contributing source boundary")
        )
    if entry["assembly"]["outside_write_rows_preserved"] <= 0:
        findings.append(
            Finding.at(8, "Finite write has no outside-scope witness", "Real neighboring rows preserved outside the authorized write interval")
        )
    return findings
