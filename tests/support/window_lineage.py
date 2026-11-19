"""Permanent real-source replay driver; tests may call both producer and verifier."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sqlalchemy import create_engine

from src.constants.constants import SEC_NET_PENSION_LIABILITY_TAGS
from src.context import Context
from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.transformers.step_cube_governance import StepCubeGovernance
from src.data_aggregate.utils.common.assembly import part_has_rows
from src.data_aggregate.utils.common.incremental import plan_window, write_part
from src.data_aggregate.utils.common.inputs import load_source, source_read_scope
from src.data_aggregate.utils.common.parts import CUBE_PARTS
from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.fundamentals.dividend_features import _ttm_dividends
from src.data_aggregate.utils.fundamentals.earnings_features import _report_state_to_daily, reported_events
from src.data_aggregate.utils.fundamentals.fundamental_features import _FN_PBO_TAG, _FN_PLAN_ASSETS_TAG, _notes_num_daily, load_notes_num_scoped
from src.data_aggregate.utils.institutionals.inputs import FTD_LINE_COLUMNS, LINEAGE_COLUMNS
from src.data_aggregate.utils.institutionals.short_flow_features import _pivot, _publish_ftd_zip_states, _publish_shortvol
from src.data_aggregate.utils.price.frames import load_trading_calendar
from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_filings, tag_company_lines
from src.data_aggregate.utils.sec_thirteen.inputs import _filing_layout, _iter_whole_quarters, _partition_packet
from src.data_aggregate.utils.sec_thirteen.institutional_books import prepare_institutional_book_history
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import prepare_superinvestor_book_history
from src.data_aggregate.utils.sec_thirteen.value_basis import nullable_value_to_daily
from src.data_aggregate.utils.text.earnings_call_features import (
    SENTIMENT_COLUMNS,
    _daily_frame,
    canonical_sentiment_rows,
    completed_sentiment_calls,
    sentiment_kpis,
    valid_sentiment_rows,
)
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore, KeySample
from src.utils.institutional_vintages import prepare_book_vintages, resolve_prepared_books
from src.utils.price.adjustments import load_bugfix
from src.utils.publication import point_in_time
from src.utils.security_series import LINE_COLUMNS, SHORT_LINE_COLUMNS, fails_series, finra_key, load_fails_basis, prepare_master, short_volume_series
from src.utils.window import Window, WindowPlan, WindowRun
from src.validate.checks.window_lineage import _State, check_window_lineage, route_family
from src.validate.checks.window_plan import check_window_plans
from src.validate.checks.window_sample import timing_sample_tables
from src.validate.result import CheckResult, jsonable
from tests.support.window_lineage_sources import employees, insiders, ownership, ownership_identity, votes
from tests.support.window_samples import WindowSamples, cube_input_windows


class _MissingEvidenceError(ValueError):
    pass


def measure_window_lineage(run: WindowSamples) -> dict[str, CheckResult]:
    """Replay only at the explicit lineage entry point, using the existing context/deadline."""
    if run.context is None:
        raise RuntimeError("The sample source context has not been constructed")
    run.lineage = {table.name: {"status": "abstain", "reason": "Not measured before interruption/deadline"} for table in timing_sample_tables()}
    run.publish()
    replay = replay_window_lineage(run.context, run.manifest, run.out / "lineage-replay.jsonl")
    (run.out / "lineage-replay.json").write_text(json.dumps(replay, indent=2), encoding="utf-8")
    results = check_window_lineage(run.context, replay, run.manifest)
    run.planning = {name: result.to_json() for name, result in check_window_plans(run.context, replay).items()}
    run.lineage = {name: result.to_json() for name, result in results.items()}
    run.publish()
    return results


@dataclass
class _Journal:
    report: dict[str, Any]
    output: Path
    prepared: dict[Any, Any] = dataclass_field(default_factory=dict)


def _context(context: Context, tickers: list[str], store: DataStore | None = None) -> Context:
    return cast(
        Context,
        SimpleNamespace(
            store=context.store if store is None else store,
            config=context.config,
            config_dir=context.config_dir,
            paths=context.paths,
            run=context.run,
            universe=SimpleNamespace(
                tickers=tickers,
                source_tickers=getattr(context.universe, "source_tickers", context.universe.tickers),
                attach_gics=context.universe.attach_gics,
                gics_codes=context.universe.gics_codes,
            ),
        ),
    )


def _tickers(manifest: dict[str, Any], table: Table, count: int) -> list[str]:
    groups = manifest.get("tables", {}).get(table.name, {}).get("sample_scope", {}).get("groups", [])
    if table.ticker_col != "ticker" or len(groups) != count or any(not isinstance(value, str) for value in groups):
        raise _MissingEvidenceError(f"{table.name}: manifest lacks all {count} pinned ticker groups")
    return groups


def _financial(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    windows = cube_input_windows(context, Tables.cube_part_fundamentals, sessions)
    rows = StepCubeFundamentals(context, context.config)._load_fundamentals(tuple(context.universe.tickers), sessions, windows)
    states = []
    for field in ["netIncome", *[name for name in rows if name.startswith("sec_") and rows[name].notna().any()]]:
        if field not in rows:
            continue
        source = Tables.sec_fundamentals_history if field.startswith("sec_") else Tables.sharadar_fundamentals_history
        daily = fundamentals_to_daily(rows, field, sessions, max_age_days=185)
        states.append(_State(source, Tables.cube_part_fundamentals, rows, field, daily, age_days=185))
    return states


def _governance(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    table = Tables.def14a_llm
    windows = cube_input_windows(context, Tables.cube_part_governance, sessions)
    raw = StepCubeGovernance(context, context.config)._load_def14a(windows[table])
    if raw is None:
        raise _MissingEvidenceError("No qualifying proxy parent observations")
    rows = point_in_time(raw, context.config, table, sessions)
    field = "board_size"
    if field not in rows:
        raise _MissingEvidenceError("Proxy source lacks board_size")
    return [
        _State(
            table,
            Tables.cube_part_governance,
            rows,
            field,
            fundamentals_to_daily(rows, field, sessions),
            exception="Later-filing CEO/person attribute fills and gender consensus remain accepted retrospective exceptions; board_size destination timing only.",
        )
    ]


def _earnings(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    table = Tables.earnings_surprises
    windows = cube_input_windows(context, Tables.cube_part_fundamentals, sessions)
    raw = load_source(context, table, columns=["ticker", "earnings_date", "eps_actual", "eps_estimate", "surprise_pct"], window=windows[table])
    if raw is None:
        raise _MissingEvidenceError("No earnings history")
    ready = point_in_time(raw, context.config, table, sessions)
    rows = reported_events(ready, sessions)
    field = "eps_actual"
    daily = _report_state_to_daily(rows, [field], sessions)[field]
    return [_State(table, Tables.cube_part_fundamentals, rows, field, daily, age_days=185)]


def _notes(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    table = Tables.notes_num
    windows = cube_input_windows(context, Tables.cube_part_fundamentals, sessions)
    prepared = load_notes_num_scoped(context, sessions, tickers=tuple(context.universe.tickers), window=windows[table])
    if prepared is None:
        raise _MissingEvidenceError("No actual notes filing units for the pinned tickers")
    states = []
    for tag in (_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG, *SEC_NET_PENSION_LIABILITY_TAGS):
        rows = prepared.copy()
        rows = rows.sort_values(["ticker", "as_of", "ddate", "filed", "adsh"])
        rows[str(tag)] = rows["value"].where(rows.tag.eq(tag) & pd.to_numeric(rows.qtrs, errors="coerce").fillna(0).eq(0))
        daily = _notes_num_daily(prepared, str(tag), sessions)
        states.append(_State(table, Tables.cube_part_fundamentals, rows, str(tag), daily, clock="available_at", age_days=460))
    return states


def _calls(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    table = Tables.earnings_call_sentiment
    windows = cube_input_windows(context, Tables.cube_part_text, sessions)
    raw = load_source(context, table, columns=SENTIMENT_COLUMNS, window=windows[table])
    source = load_source(
        context, Tables.earnings_call_sections, columns=["ticker", "quarter", "paragraph", "as_of"], window=windows[Tables.earnings_call_sections]
    )
    if raw is None or source is None:
        raise _MissingEvidenceError("Missing saved sentiment or its source call dates; inference is prohibited")
    source = source[["ticker", "quarter", "as_of"]].drop_duplicates()
    source["as_of"] = pd.to_datetime(source["as_of"])
    model = f"{context.config.data_aggregate.earnings_calls.sentiment_model}:speaker-clean-v3"
    valid = canonical_sentiment_rows(valid_sentiment_rows(raw, model))
    complete = completed_sentiment_calls(valid)
    valid = valid.loc[pd.MultiIndex.from_frame(valid[["ticker", "quarter"]]).isin(complete)]
    prepared = point_in_time(valid, context.config, table, sessions)
    calls = point_in_time(source, context.config, Tables.earnings_call_sections, sessions)
    clocks = prepared.groupby(["ticker", "quarter"], as_index=False)["_source_as_of"].first()
    rows = sentiment_kpis(prepared).merge(clocks, on=["ticker", "quarter"], validate="one_to_one")
    calls = calls.rename(columns={"_source_as_of": "_source_call_as_of", "as_of": "_projection_parent_asof"})
    rows = rows.merge(calls[["ticker", "quarter", "_source_call_as_of", "_projection_parent_asof"]], on=["ticker", "quarter"], validate="one_to_one")
    rows = rows.loc[pd.to_datetime(rows._source_as_of).eq(pd.to_datetime(rows._source_call_as_of))].copy()
    rows["as_of"] = rows[["as_of", "_projection_parent_asof"]].max(axis=1)
    units = {key: group[[*table.pk, "as_of", "model"]].to_dict("records") for key, group in valid.groupby(["ticker", "quarter", "as_of"])}
    rows["_source_units"] = [units[(row["ticker"], row["quarter"], row["_source_as_of"])] for row in rows.to_dict("records")]
    lifetime = int(context.config.data_aggregate.earnings_calls.signal_sessions)
    field = "ec_prep_tone"
    daily = _daily_frame(rows, field, sessions, lifetime)
    return [_State(table, Tables.cube_part_text, rows, field, daily, lifetime_sessions=lifetime)]


def _security_rows(
    context: Context, table: Table, columns: tuple[str, ...], master: pd.DataFrame, lineage: pd.DataFrame, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    """Complete relevant security lines plus genuine source rows witnessing served days."""
    # Keep the complete configured scoped history, including the prior ZIP/state
    # carried into the first asserted2026 boundary.
    interval = cube_input_windows(context, Tables.cube_part_institutionals, sessions)[table]
    payload_scope = source_read_scope(context, table, interval)
    target = master.loc[master.canonical_company.isin(context.universe.tickers)]
    if target.empty:
        raise _MissingEvidenceError("Missing complete dated security identity units for pinned output tickers")
    if table == Tables.sec_fails_to_deliver_security:
        selected = {"cusip": sorted(target.cusip.dropna().unique()), **payload_scope.get("where", {})}
    else:
        identity_symbols = lineage.loc[lineage.canonical_ticker.isin(context.universe.tickers), "symbol"]
        wanted = set(target.source_symbol.dropna().map(finra_key)) | set(identity_symbols.dropna().map(finra_key))
        symbols = context.store.distinct(table, "source_symbol", where={} if interval is None else {"date": interval})
        selected = {"source_symbol": [symbol for symbol in symbols if finra_key(symbol) in wanted]}
    metadata_interval = _short_metadata_window(context, sessions, interval) if table == Tables.sec_short_volume_security else interval
    metadata_scope: dict[str, Any] = {"date": metadata_interval} if table == Tables.sec_short_volume_security else {}
    if table == Tables.sec_fails_to_deliver_security:
        metadata_scope["period"] = sorted(_ftd_metadata_periods(context, sessions, interval))
    global_days = set(context.store.distinct(table, "date", where=metadata_scope))
    own_metadata_scope = {**selected, **metadata_scope}
    own_days = set(context.store.distinct(table, "date", where=own_metadata_scope))
    missing = sorted(global_days - own_days)
    # Never request calendar-day-count groups: nonserved weekends would make the
    # streaming sampler read the entire global payload seeking nonexistent groups.
    witnesses = (
        context.store.sample_keys(
            table,
            KeySample(
                "date", len(missing), {"date": missing, **({"period": metadata_scope["period"]} if "period" in metadata_scope else {})}, "date"
            ),
        )
        if missing
        else pd.DataFrame(columns=list(table.pk))
    )
    keys = witnesses[list(table.pk)].to_dict("records")
    payload_scope.pop("where", None)
    chunks = list(context.store.iter_load(table, columns=columns, where=[selected, *keys], **payload_scope))
    if not chunks:
        raise _MissingEvidenceError("No security-source payload in the bounded complete-unit scope")
    result = pd.concat(chunks, ignore_index=True).drop_duplicates(list(table.pk))
    result.attrs["lineage_dependency_scope"] = {
        "economic_history": {"start": None if interval is None else interval.start, "end": context.run.as_of},
        "global_served_date_metadata": {"start": metadata_interval.start, "end": metadata_interval.end, "periods": metadata_scope.get("period")},
        "global_served_dates": len(global_days),
        "witness_rows": len(witnesses),
        "economic_and_witness_rows": len(result),
    }
    return result


def _short_metadata_window(context: Context, sessions: pd.DatetimeIndex, retained: Window | None) -> Window:
    # Own volume lasts exactly one publication session, with no rolling dependency.
    # Keep all scoped economic history; only unasserted older global zero-day
    # witnesses are unnecessary. Derive the first raw day through the actual
    # production publication selector, retaining the pre-year boundary when needed.
    ready = point_in_time(pd.DataFrame({"date": sessions}, index=sessions), context.config, Tables.sec_short_volume_security, sessions)[
        "as_of"
    ].reindex(sessions)
    anchors = sessions[pd.to_datetime(ready).dt.year.eq(int(context.config.validate.window_samples.year))]
    if anchors.empty:
        raise _MissingEvidenceError("No source session publishes into the asserted sample year")
    return Window(anchors[0] if retained is None else max(retained.start, anchors[0]), context.run.as_of if retained is None else retained.end)


def _ftd_metadata_periods(context: Context, sessions: pd.DatetimeIndex, retained: Window | None) -> set[str]:
    """Actual published ZIPs in the asserted interval plus their genuine prior vector seed."""
    table = Tables.sec_fails_to_deliver_security
    stats = context.store.key_stats(table, "period", "date", where=source_read_scope(context, table, retained).get("where", {}))
    if stats.empty:
        raise _MissingEvidenceError("No persisted FTD ZIP-period metadata")
    calendar = sessions[sessions.year == int(context.config.validate.window_samples.year)]
    if calendar.empty:
        raise _MissingEvidenceError("No actual market sessions in the asserted sample year")
    start = sessions[max(0, sessions.get_loc(calendar[0]) - 1)]
    end = sessions[min(len(sessions) - 1, sessions.get_loc(calendar[-1]) + 1)]
    coverage = stats.rename(columns={"key": "period", "last": "date"})[["period", "date"]]
    ready = point_in_time(coverage, context.config, table, sessions).set_index("period")["as_of"]
    events = pd.DataFrame({"period": ready.index, "published": ready.to_numpy()}).dropna().sort_values(["published", "period"])
    # Follow the publisher's forward-ZIP ordering; the seed is chosen from real
    # publication metadata, never by searching economic values for non-null data.
    forward = []
    current = ""
    for event in events.itertuples(index=False):
        if event.period > current:
            forward.append(event)
            current = event.period
    selected = {event.period for event in forward if start <= event.published <= end}
    prior = [event for event in forward if event.published < start]
    if prior:
        selected.add(prior[-1].period)
    return selected


def _security_identity(context: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = context.store.load(Tables.universe_cusip, columns=FTD_LINE_COLUMNS, where={"source": "ftd"}, optional=True)
    lineage = context.store.load(Tables.universe_cik_symbol, columns=LINEAGE_COLUMNS, optional=True)
    if master is None or lineage is None:
        raise _MissingEvidenceError("Missing security master or dated issuer lineage")
    return master, lineage


def _short(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    table = Tables.sec_short_volume_security
    master, lineage = _security_identity(context)
    raw = _security_rows(context, table, SHORT_LINE_COLUMNS, master, lineage, sessions)
    prepared = point_in_time(raw, context.config, table, sessions)
    series, _ = short_volume_series(prepared, master, lineage, context.universe.tickers)
    source = _pivot(series, "short_volume", sessions)
    clock = prepared[["date", "as_of", "_source_date"]].drop_duplicates().assign(date=lambda frame: pd.to_datetime(frame.date))
    dates = clock.set_index("date")["as_of"].reindex(sessions)
    daily = _publish_shortvol(source, dates)
    rows = series.merge(clock, on="date", validate="many_to_one")
    ids = {pd.Timestamp(date): group[list(table.pk)].to_dict("records") for date, group in raw.groupby("date")}
    rows["_source_units"] = pd.to_datetime(rows["date"]).map(ids)
    rows.attrs.update(raw.attrs)
    return [
        _State(
            table,
            Tables.cube_part_institutionals,
            rows,
            "short_volume",
            daily,
            lifetime_sessions=1,
            exception="Own-ticker one-session published volume state after production security mapping. All scoped economic history retained; global served-date witnesses cover raw sessions actually publishing into the asserted year. Historical/rolling ratios, peer ranks and the full institutional feature panel remain unmeasured.",
        )
    ]


def _fails(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    table = Tables.sec_fails_to_deliver_security
    master, lineage = _security_identity(context)
    raw = _security_rows(context, table, LINE_COLUMNS, master, lineage, sessions)
    prepared = point_in_time(raw, context.config, table, sessions)
    # Keep colliding identities for the selected CUSIPs; unrelated witness lines
    # supply actual served dates but cannot acquire a sampled company's identity.
    wanted = set(master.loc[master.canonical_company.isin(context.universe.tickers), "cusip"])
    selected = prepare_master(master.loc[master.cusip.isin(wanted)])
    windows = cube_input_windows(context, Tables.cube_part_institutionals, sessions)
    basis = load_fails_basis(context, load_bugfix(context.config_dir), selected, windows=windows)
    series, coverage = fails_series(prepared, selected, basis)
    series = series.loc[series.ticker.isin(context.universe.tickers)]
    state = _pivot(series, "fails_quantity", sessions)
    clock = prepared[["period", "as_of", "_source_period"]].drop_duplicates()
    available = clock.set_index("period")["as_of"].to_dict()
    daily = _publish_ftd_zip_states(state, coverage, sessions, available)
    rows = _ftd_selected_rows(series, coverage, sessions, context.universe.tickers)
    rows = rows.merge(clock, on="period", validate="many_to_one")
    ids = {period: group[list(table.pk)].to_dict("records") for period, group in raw.groupby("period")}
    rows["_source_units"] = rows.period.map(ids)
    rows.attrs.update(raw.attrs)
    return [
        _State(
            table,
            Tables.cube_part_institutionals,
            rows,
            "fails_quantity",
            daily,
            exception="Complete sampled security units plus genuine global served-date witnesses; production settlement conversion and ZIP-state publication, before rolling economic ratios.",
        )
    ]


def _ftd_selected_rows(series: pd.DataFrame, coverage: pd.DataFrame, sessions: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """The publisher selects the latest actual served session's whole vector, including NULL."""
    latest = coverage.loc[pd.to_datetime(coverage.date).isin(sessions), ["period", "date"]].drop_duplicates()
    latest = latest.sort_values("date").drop_duplicates("period", keep="last")
    grid = latest.merge(pd.DataFrame({"ticker": tickers}), how="cross")
    return grid.merge(series, on=["ticker", "date", "period"], how="left", validate="one_to_one")


def _dividends(context: Context, sessions: pd.DatetimeIndex) -> list[_State]:
    table = Tables.dividends
    windows = cube_input_windows(context, Tables.cube_part_fundamentals, sessions)
    raw = load_source(context, table, columns=["ticker", "date", "dividends"], window=windows[table])
    if raw is None:
        raise _MissingEvidenceError("No real dividend cash events")
    rows = point_in_time(raw, context.config, table, sessions)
    rows["available_on"] = rows["as_of"]
    daily = _ttm_dividends(rows, sessions, context.universe.tickers)
    rows["as_of"] = rows.available_on
    return [
        _State(
            table,
            Tables.cube_part_fundamentals,
            rows,
            "dividends",
            daily,
            exception="Actual252-session dividend cash state with original ex-date expiry; per-share basis and peer transforms are outside this state probe.",
        )
    ]


def _packet_batches(context: Context, pins: list[dict[str, Any]], window: Window | None) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Read every version/position of each pinned manager-quarter, without ticker filtering."""
    table = Tables.sec13f_hr
    query = _filing_layout(context.store, window, sorted({row["cik"] for row in pins}), None, ())
    lines = context.store.load(Tables.universe_cusip, columns=FTD_LINE_COLUMNS, optional=True)
    if query is None or lines is None:
        raise _MissingEvidenceError("Missing complete-accession13F layout or dated security identity")
    read_scope = source_read_scope(context, table, window)
    source_filters = read_scope.pop("where", {})
    query.update(read_scope)
    scopes = {(row["cik"], row["period"]) for row in pins}
    query["where"] = [
        {**source_filters, "cik": cik, "period": pd.Timestamp(period), "parse_status": ["complete", "position"]} for cik, period in scopes
    ]
    stream = context.store.iter_load(table, chunksize=100_000, order_by=["cik", "period", "accession_number"], **query)
    batches = []
    for packet in _iter_whole_quarters(stream):
        filings, holdings = clean_filings(*_partition_packet(packet), context.config.data_aggregate.bond_lines, context.config)
        # Book eligibility is independent of the accepted pooled value/unit donor
        # repair; values from this diagnostic are never economic aggregate evidence.
        batches.append((filings, tag_company_lines(holdings, lines, None)))
    return batches


def _packet_rows(states: pd.DataFrame, manifests: pd.DataFrame, pins: list[dict[str, Any]], field: str) -> pd.DataFrame:
    rows = []
    for pin in pins:
        selected = states.loc[states.cik.eq(pin["cik"]) & states.period.eq(pd.Timestamp(pin["period"]))]
        origins = selected.loc[selected.book_known.astype(bool), "_day"]
        origin = origins.min() if len(origins) else pd.NaT
        for _, state in selected.iterrows():
            references = state["_selected_accessions"]
            evidence = manifests.loc[
                manifests.cik.eq(pin["cik"]) & manifests.period.eq(pd.Timestamp(pin["period"])) & manifests.accession_number.isin(references)
            ]
            if evidence.empty:
                raise _MissingEvidenceError("Actual selected book references have no raw complete filing-clock ancestry")
            if set(evidence.accession_number) != set(references):
                raise _MissingEvidenceError("An actual selected accession is missing from the complete raw manifest")
            source = evidence.sort_values(["public_at", "filing_date", "accession_number"]).iloc[-1].to_dict()
            source.update(
                ticker=pin["ticker"],
                as_of=state._day,
                _source_origin=origin,
                _source_units=evidence[["cik", "period", "accession_number", "filing_date", "accepted_at"]].to_dict("records"),
                _selected_accessions=references,
            )
            source[field] = float(bool(state.book_known))
            rows.append(source)
    return pd.DataFrame(rows)


def _thirteen_f(context: Context, sessions: pd.DatetimeIndex, pins: list[dict[str, Any]]) -> list[_State]:
    windows = cube_input_windows(context, Tables.cube_part_sec_thirteen, sessions)
    batches = _packet_batches(context, pins, windows[Tables.sec13f_hr])
    if not batches:
        raise _MissingEvidenceError("No complete pinned13F manager-quarter packets")
    manifests = pd.concat([prepare_book_vintages(filings, holdings, sessions, context.config) for filings, holdings in batches], ignore_index=True)
    observed = {}

    def observe(manifest: pd.DataFrame, positions: pd.DataFrame, day: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        books, states = resolve_prepared_books(manifest, positions, day)
        for state in states.to_dict("records"):
            actual = books.loc[books.cik.eq(state["cik"]) & books.period.eq(state["period"])]
            observed[(state["cik"], state["period"], state["public_at"])] = sorted({state["accession_number"], *actual.accession_number.dropna()})
        return books, states

    # Observe the existing resolver boundary without changing arguments, values,
    # selection or application interfaces; compact output intentionally loses IDs.
    with patch("src.data_aggregate.utils.sec_thirteen.institutional_books.resolve_prepared_books", side_effect=observe):
        broad = prepare_institutional_book_history(iter(batches), sessions, config=context.config)
    broad_observed = observed
    age = int(context.config.data_aggregate.sec_thirteen.superinvestor.max_age_calendar_days)
    observed = {}
    with patch("src.data_aggregate.utils.sec_thirteen.superinvestor_books.resolve_prepared_books", side_effect=observe):
        elite = prepare_superinvestor_book_history(iter(batches), sessions, age, context.config)
    if broad is None or elite is None:
        raise _MissingEvidenceError("Pinned packets did not reach both production book-family selectors")
    selected = broad[1].copy()
    selected["_selected_accessions"] = [
        broad_observed.get((row.cik, row.period, row.public_at), [row.accession_number]) for row in selected.itertuples()
    ]
    broad_rows = _packet_rows(selected, manifests, pins, "packet_known_broad")
    broad_daily = nullable_value_to_daily(broad_rows, "packet_known_broad", sessions)
    elite_events = []
    elite_selected = []
    for snapshot in elite:
        for pin in pins:
            book = snapshot.level_books.get(pin["cik"])
            known = book is not None and book.period == pd.Timestamp(pin["period"]) and book.support.book_known
            elite_events.append({"ticker": pin["ticker"], "as_of": snapshot.day, "packet_known_elite": 1.0 if known else np.nan})
            if known and not any(row["cik"] == book.cik and row["_day"] == book.public_at for row in elite_selected):
                elite_selected.append(
                    {
                        "cik": book.cik,
                        "period": book.period,
                        "_day": book.public_at,
                        "book_known": True,
                        "_selected_accessions": observed.get(
                            (book.cik, book.period, book.public_at), sorted(set(book.df_positions.accession_number.dropna()))
                        ),
                    }
                )
    elite_rows = _packet_rows(pd.DataFrame(elite_selected), manifests, pins, "packet_known_elite")
    elite_daily = nullable_value_to_daily(pd.DataFrame(elite_events), "packet_known_elite", sessions)
    limitation = "Sampled complete manager-quarter packet eligibility through each actual book-family selector; neither a broad/elite economic aggregate nor elite-cohort membership. Pooled later-donor value/unit repair remains an accepted retrospective exception and is not rerun or certified."
    return [
        _State(Tables.sec13f_hr, Tables.cube_part_sec_thirteen, broad_rows, "packet_known_broad", broad_daily, exception=limitation),
        _State(Tables.sec13f_hr, Tables.cube_part_sec_thirteen, elite_rows, "packet_known_elite", elite_daily, age_days=age, exception=limitation),
    ]


def _assemble_fields(context: Context, states: list[_State], probes: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Persist only in disposable SQLite, then use the real projection/merge/write path."""
    state = states[0]
    wanted = pd.concat([pd.DataFrame(records)[["date", "ticker"]] for records in probes.values()]).drop_duplicates()
    write = Window(pd.Timestamp(wanted.date.min()), pd.Timestamp(wanted.date.max()))
    root, presence = _assembly_inputs(context, states, write)
    engine = create_engine("sqlite://")
    try:
        store = DataStore(engine)
        panel = pd.concat(
            {
                f"_lineage_state_{item.field}": item.daily.reindex(columns=context.universe.tickers)
                .rename_axis(index="date", columns="ticker")
                .stack(future_stack=True)
                for item in states
            },
            axis=1,
        ).reset_index()
        columns = [f"_lineage_state_{item.field}" for item in states]
        # Keep genuine neighboring daily values as witnesses outside the finite rewrite.
        days = pd.DatetimeIndex(state.daily.index).sort_values()
        neighbors = [*days[days < write.start][-1:], *days[days > write.end][:1]]
        panel = panel.loc[write.mask(panel.date) | panel.date.isin(neighbors)].copy()
        if root is None or root.empty:
            raise _MissingEvidenceError("No observed production price grid at selected eligibility boundaries")
        if presence is not None:
            panel = panel.merge(presence, on=["date", "ticker"], how="left", validate="one_to_one")
        store.replace(Tables.cube_part_prices, root)
        store.replace(state.part, panel)
        replay = _context(context, context.universe.tickers, store)
        config = deepcopy(context.config)
        config.data_aggregate.assembly.compulsory_parts = [Tables.cube_part_prices.name, state.part.name]
        config.data_aggregate.assembly.optional_parts = []
        config.data_aggregate.assembly.columns = {state.part.name: columns}
        config.data.tables.cube.warmup = {name: 0 for name in config.data_aggregate.assembly.compulsory_parts}
        replay.config = config
        step = StepAssembleCube(replay, config)
        cube, _ = step._assemble(write)
        checked = {item.field: _assembled_rows(cube, panel, item.field, probes[item.field]) for item in states}
        store.replace(Tables.cube, panel[["date", "ticker", *columns]])
        before = store.load(Tables.cube)
        run = WindowRun("incremental", "read-only source lineage replay", context.run.instant)
        plan = WindowPlan(write, write, write, run, None, (("ticker", tuple(context.universe.tickers)),))
        write_part(store, Tables.cube, cube[["date", "ticker", *columns]], plan)
        after = store.load(Tables.cube)
        outside = ~(write.mask(before.date) & before.ticker.isin(context.universe.tickers))
        keys = ["date", "ticker"]
        pd.testing.assert_frame_equal(
            before.loc[outside].sort_values(keys).reset_index(drop=True),
            after.loc[~(write.mask(after.date) & after.ticker.isin(context.universe.tickers))].sort_values(keys).reset_index(drop=True),
            check_dtype=False,
        )
        common = {
            "write": {"start": write.start, "end": write.end, "tickers": context.universe.tickers},
            "write_kind": "Probe min/max finite-write protection only; configured overlap is measured separately by actual production planners",
            "outside_write_rows_preserved": int(outside.sum()),
            "store": "disposable in-memory SQLite",
            "contextual_presence": {
                "table": state.part.name,
                "columns": [] if presence is None else [name for name in presence if name not in {"date", "ticker"}],
                "role": "Genuine stored part values, excluded by production projection; structural presence only, not note ancestry or economic evidence",
            },
        }
        return {
            item.field: {**common, "assembled": checked[item.field], "column": f"_lineage_state_{item.field}", "projected_family_columns": columns}
            for item in states
        }
    finally:
        engine.dispose()


def _assembled_rows(cube: pd.DataFrame, panel: pd.DataFrame, field: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    column = f"_lineage_state_{field}"
    wanted = pd.DataFrame(records)[["date", "ticker"]].drop_duplicates()
    check = wanted.merge(cube[["date", "ticker", column]], on=["date", "ticker"], how="left", validate="one_to_one", indicator=True)
    expected = wanted.merge(panel, on=["date", "ticker"], how="left", validate="one_to_one")[column].astype("float32")
    if not check._merge.eq("both").all() or not np.allclose(pd.to_numeric(check[column], errors="coerce"), expected, rtol=0, atol=0, equal_nan=True):
        raise ValueError("Assembly dropped a boundary key or changed its selected source state")
    return jsonable(check.drop(columns="_merge").to_dict("records"))


def _assembly_inputs(context: Context, states: list[_State], write: Window) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    state = states[0]
    root = context.store.load(
        Tables.cube_part_prices,
        columns=["date", "ticker", "close_split"],
        where={"ticker": context.universe.tickers},
        since=write.start,
        until=write.end,
        optional=True,
    )
    presence = None
    observed = pd.concat([item.daily.notna() for item in states]).groupby(level=0).any()
    if not observed.reindex(columns=context.universe.tickers, fill_value=False).any(axis=0).all():
        actual = context.store.columns(state.part)
        fields = list(dict.fromkeys([field for field in ["f_days_since_earnings", *actual] if field in actual and field not in {"date", "ticker"}]))[
            :3
        ]
        days = pd.DatetimeIndex(state.daily.index).sort_values()
        before = days[days < write.start][-1:]
        after = days[days > write.end][:1]
        presence = context.store.load(
            state.part,
            columns=["date", "ticker", *fields],
            where={"ticker": context.universe.tickers},
            since=before[0] if len(before) else write.start,
            until=after[0] if len(after) else write.end,
            optional=True,
        )
        if presence is None or not fields or not presence[fields].notna().any(axis=None):
            raise _MissingEvidenceError("Known-empty state has no genuine observed wide-part context to establish assembly presence")
        presence = presence.rename(columns={field: f"_lineage_context_{field}" for field in fields})
    return root, presence


def replay_window_lineage(context: Context, manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    """Retain full required histories; probe real adjacent sessions in the pinned year."""
    sessions = load_trading_calendar(context)
    report: dict[str, Any] = {"sessions": jsonable(list(sessions)), "routes": []}
    output.write_text(json.dumps({"sessions": report["sessions"]}) + "\n", encoding="utf-8")
    report["planning"] = _planning_trace(context, sessions, manifest)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"planning": jsonable(report["planning"])}) + "\n")
    groups = [
        [
            (Tables.sharadar_fundamentals_history, _financial),
            (Tables.sec_fundamentals_history, _financial),
            (Tables.earnings_surprises, _earnings),
            (Tables.notes_num, _notes),
            (Tables.dividends, _dividends),
            (Tables.fundamentals_employees, employees),
        ],
        [(Tables.def14a_llm, _governance), (Tables.sec_8k_votes, votes)],
        [(Tables.earnings_call_sentiment, _calls)],
        [
            (Tables.sec_short_volume_security, _short),
            (Tables.sec_fails_to_deliver_security, _fails),
            (Tables.insider_transactions, insiders),
            (Tables.sec_13d, ownership),
            (Tables.sec_13g, ownership),
        ],
    ]
    journal = _Journal(report, output)
    for routes in groups:
        _replay_group(context, manifest, routes, sessions, journal)
    count = int(context.config.validate.window_samples.keys_per_table)
    pins = manifest.get("tables", {}).get(Tables.sec13f_hr.name, {}).get("selection", {}).get("where", [])
    try:
        if not isinstance(pins, list) or len(pins) != count:
            raise _MissingEvidenceError("Manifest lacks the complete five pinned13F source keys")
        tickers = sorted({pin["ticker"] for pin in pins})
        scoped = _context(context, tickers)
        pending = _stage_family(scoped, _thirteen_f(scoped, sessions, pins), sessions, journal)
        _finish_group(context, pending, journal)
    except (ValueError, RuntimeError, KeyError, AssertionError) as error:
        entry = _error_entry(Tables.sec13f_hr, error)
        report["routes"].append(entry)
        _checkpoint_entries([entry], output)
    return report


def _replay_group(
    context: Context, manifest: dict[str, Any], routes: list[tuple[Table, Callable[..., list[_State]]]], sessions: pd.DatetimeIndex, journal: _Journal
) -> None:
    pending = []
    for table, build in routes:
        try:
            scoped, states = _build_route(context, manifest, (table, build), sessions, journal.prepared)
            pending.extend(_stage_family(scoped, states, sessions, journal))
        except (ValueError, RuntimeError, KeyError, AssertionError) as error:
            entry = _error_entry(table, error)
            journal.report["routes"].append(entry)
            _checkpoint_entries([entry], journal.output)
    _finish_group(context, pending, journal)


def _build_route(
    context: Context, manifest: dict[str, Any], route: tuple[Table, Callable[..., list[_State]]], sessions: pd.DatetimeIndex, prepared: dict[Any, Any]
) -> tuple[Context, list[_State]]:
    table, build = route
    count = int(context.config.validate.window_samples.keys_per_table)
    sample_table = Tables.cube_part_institutionals if table in {Tables.sec_short_volume_security, Tables.sec_fails_to_deliver_security} else table
    tickers = _tickers(manifest, sample_table, count)
    calculation_tickers = (
        sorted(set(_tickers(manifest, Tables.sec_13d, count)) | set(_tickers(manifest, Tables.sec_13g, count))) if build is ownership else tickers
    )
    key = (build, table if build is ownership else None, tuple(calculation_tickers))
    if key not in prepared:
        scoped = _context(context, calculation_tickers)
        if build is ownership:
            if "ownership_identity" not in prepared:
                prepared["ownership_identity"] = ownership_identity(context)
            prepared[key] = ownership(scoped, sessions, table, prepared["ownership_identity"])
        else:
            prepared[key] = build(scoped, sessions)
    states = [state for state in prepared[key] if state.table == table] if build in {_financial, ownership} else prepared[key]
    if not states:
        raise _MissingEvidenceError("No qualifying source state reached a production selector")
    return _context(context, tickers), states


def _error_entry(table: Table, error: Exception) -> dict[str, Any]:
    if isinstance(error, _MissingEvidenceError):
        return {
            "table": table.name,
            "reason": str(error),
            "remedy": "Supply the missing complete source/key/calendar dependency; do not substitute synthetic economics.",
        }
    return {"table": table.name, "failure": f"{type(error).__name__}: {error}"}


def _checkpoint_entries(entries: list[dict[str, Any]], output: Path) -> None:
    # Completed JSONL lines survive a later source failure or the common deadline.
    with output.open("a", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry) + "\n")


def _planning_trace(context: Context, sessions: pd.DatetimeIndex, manifest: dict[str, Any]) -> dict[str, Any]:
    tickers = _tickers(manifest, Tables.cube, int(context.config.validate.window_samples.keys_per_table))
    scoped = copy(context)
    scoped._universe = SimpleNamespace(tickers=tickers)
    parts = {}
    written = {}
    for part in CUBE_PARTS:
        plan = plan_window(scoped, part.table, sessions, scope=tickers)
        if plan is None:
            parts[part.name] = {"reason": "Actual production planner returned no applicable window"}
            continue
        written[part.table] = plan.write
        parts[part.name] = _plan_record(plan)
    assembly = scoped.config.data_aggregate.assembly
    selected = [*assembly.compulsory_parts, *assembly.optional_parts]
    used = [
        name
        for name in selected
        if name in assembly.compulsory_parts
        or part_has_rows(scoped.store, next(part.table for part in CUBE_PARTS if part.name == name), {"ticker": tickers})
    ]
    assembled = StepAssembleCube(scoped, scoped.config)._plan_window(written)
    sources = {
        table.name: {
            "resume_mode": None if table.resume is None else table.resume.mode,
            "overlap": (
                OmegaConf.to_container(scoped.config.data.tables[table.name].overlap, resolve=True)
                if "overlap" in scoped.config.data.tables.get(table.name, {})
                else None
            ),
            "note": "Configured source-work/Resume mapping only; provider and cache work is covered by separate propagation tests",
        }
        for table in timing_sample_tables()
    }
    return jsonable(
        {
            "parts": parts,
            "assembly": None if assembled is None else _plan_record(assembled),
            "selected_components": used,
            "sources": sources,
            "tickers": tickers,
            "as_of": scoped.run.as_of,
        }
    )


def _plan_record(plan: WindowPlan) -> dict[str, Any]:
    return jsonable(
        {
            "requested": {"start": plan.requested.start, "end": plan.requested.end},
            "read": {"start": plan.read.start, "end": plan.read.end},
            "write": {"start": plan.write.start, "end": plan.write.end},
            "frontier": plan.frontier,
            "mode": plan.run.mode,
            "keys": dict(plan.scope),
        }
    )


def _stage_family(
    context: Context, states: list[_State], sessions: pd.DatetimeIndex, journal: _Journal
) -> list[tuple[_State, dict[str, Any], list[dict[str, Any]]]]:
    pending = []
    for state in states:
        probes = _probe_rows(context, state, sessions)
        entry = _capture_record(context, state, probes, {"status": "pending"})
        entry["reason"] = "Actual target-part assembly pending" if probes else f"{state.field}: no eligible2026 source coverage for the pinned keys"
        journal.report["routes"].append(entry)
        pending.append((state, entry, probes))
    _checkpoint_entries([entry for _, entry, _ in pending], journal.output)
    return pending


def _finish_group(context: Context, pending: list[tuple[_State, dict[str, Any], list[dict[str, Any]]]], journal: _Journal) -> None:
    eligible = [(state, entry, probes) for state, entry, probes in pending if probes]
    if not eligible:
        return
    if len({state.part for state, _, _ in eligible}) != 1:
        raise ValueError("A projected group must belong to exactly one actual production part")
    # Namespace only diagnostic column names; original raw fields, selected IDs,
    # requested pins and per-state expected-value verification remain unchanged.
    states = [replace(state, field=f"{state.table.name}__{state.field}") for state, _, _ in eligible]
    probes = {state.field: original[2] for state, original in zip(states, eligible, strict=True)}
    covered = sorted({row["ticker"] for records in probes.values() for row in records})
    try:
        assembled = _assemble_fields(_context(context, covered), states, probes)
        for state, (_, entry, _) in zip(states, eligible, strict=True):
            entry.pop("reason", None)
            entry["assembly"] = jsonable(assembled[state.field])
    except (ValueError, RuntimeError, KeyError, AssertionError) as error:
        for state, entry, _ in eligible:
            entry.pop("reason", None)
            entry.update(_error_entry(state.table, error))
    _checkpoint_entries([entry for _, entry, _ in eligible], journal.output)


def _capture_record(context: Context, state: _State, records: list[dict[str, Any]], assembly: dict[str, Any]) -> dict[str, Any]:
    return {
        "table": state.table.name,
        "route_id": route_family(state.table, state.field),
        "required": True,
        "part": state.part.name,
        "tickers": sorted({row["ticker"] for row in records}),
        "calculation_tickers": list(state.daily.columns),
        "requested_tickers": context.universe.tickers,
        "field": state.field,
        "clock": state.clock,
        "age_days": state.age_days,
        "lifetime_sessions": state.lifetime_sessions,
        "exception": state.exception,
        "rows": jsonable(state.rows.loc[state.rows.ticker.isin(context.universe.tickers)].to_dict("records")),
        "dependency_scope": jsonable(state.rows.attrs.get("lineage_dependency_scope")),
        "daily": jsonable(records),
        "assembly": jsonable(assembly),
    }


def _probe_rows(context: Context, state: _State, sessions: pd.DatetimeIndex) -> list[dict[str, Any]]:
    records = []
    year = int(context.config.validate.window_samples.year)
    for ticker in context.universe.tickers:
        keep_null = state.table in {Tables.sec_8k_votes, Tables.sec_fails_to_deliver_security}
        rows = state.rows.loc[state.rows.ticker.eq(ticker) & (state.rows[state.field].notna() | keep_null)].sort_values("as_of", kind="stable")
        if state.table == Tables.dividends:
            rows = rows.loc[pd.to_numeric(rows[state.field], errors="coerce").ne(0)]
        if rows.empty and state.table == Tables.notes_num:
            rows = state.rows.loc[state.rows.ticker.eq(ticker)].sort_values("as_of", kind="stable")
        candidates = rows.loc[pd.to_datetime(rows.as_of).dt.year.eq(year)]
        if candidates.empty and state.table in {Tables.notes_num, Tables.dividends}:
            units = state.rows.loc[state.rows.ticker.eq(ticker)].sort_values("as_of", kind="stable")
            candidates = units.loc[pd.to_datetime(units.as_of).dt.year.eq(year)]
        if candidates.empty:
            continue
        position = sessions.get_loc(pd.Timestamp(candidates.iloc[0].as_of))
        for day in sessions[max(0, position - 1) : position + 2]:
            value = state.daily.reindex(index=[day], columns=[ticker]).iloc[0, 0]
            records.append({"ticker": ticker, "date": day, "value": value})
    return records
