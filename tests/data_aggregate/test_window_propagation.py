"""Configured boundaries exercised through actual cube producers and shared planning."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.transformers.step_cube_governance import StepCubeGovernance
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.utils.common.assembly import read_receipts, write_receipt
from src.data_aggregate.utils.common.incremental import plan_window
from src.data_aggregate.utils.common.parts import CUBE_PARTS
from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_aggregate.utils.price.frames import load_price_frames, load_trading_calendar
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import prepare_superinvestor_book_history
from src.data_store.schema import Table, Tables
from src.utils.data_policy import source_read_window
from src.utils.window import Window, WindowPlan, WindowRun
from tests.canonical_schema import create_canonical
from tests.data_aggregate.test_price_pit_contract import seed_prices
from tests.data_aggregate.test_superinvestor_books import _on, _packet
from tests.support.config import fixture_config
from tests.support.universe import universe_of


def test_actual_price_build_obeys_configured_dates(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    context.config = fixture_config()
    item = context.config.data.tables[Tables.cube_part_prices.name]
    item.start_date, item.end_date = "2022-10-03", "2022-10-21"
    context.run = WindowRun("full", "explicit bounded price rebuild", pd.Timestamp("2022-11-16"))

    StepCubePrices(context, context.config).run()

    df_saved = sqlite_store.load(Tables.cube_part_prices)
    assert not df_saved.empty
    assert pd.to_datetime(df_saved.date).between("2022-10-03", "2022-10-21").all(), (df_saved.date.min(), df_saved.date.max())
    assert set(df_saved.ticker) == {"AAA", "BBB"}
    print("SANITY: the actual price producer publishes only its configured finite dates and scoped tickers, including full mode.")


@pytest.mark.parametrize("table", [part.table for part in CUBE_PARTS], ids=lambda table: table.name)
def test_every_part_uses_configured_overlap_and_ignores_future_frontier(sqlite_store: Any, table: Table) -> None:
    config = fixture_config()
    item = config.data.tables[table.name]
    item.start_date, item.end_date = "2026-02-01", "2026-11-10"
    sessions = pd.bdate_range("2010-01-01", "2026-12-31")
    overlap = 120 if table == Tables.cube_part_targets else 9
    item.reprocess.overlap.value = overlap
    item.reprocess.overlap.unit = "trading_sessions" if table == Tables.cube_part_targets else "calendar_days"
    sqlite_store.replace(table, pd.DataFrame({"ticker": ["AAPL", "AAPL"], "date": pd.to_datetime(["2026-10-30", "2026-12-31"]), "value": 1.0}))
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL", "AMZN", "JPM", "MSFT", "XOM"]),
            run=WindowRun("incremental", "finite cube policy check", pd.Timestamp("2026-11-16")),
        ),
    )

    plan = plan_window(context, table, sessions)

    assert plan is not None and plan.frontier == pd.Timestamp("2026-10-30")
    expected = sessions[sessions.get_loc("2026-10-30") - overlap] if table == Tables.cube_part_targets else pd.Timestamp("2026-10-21")
    assert plan.write.start == expected and plan.write.end == pd.Timestamp("2026-11-10")
    assert plan.requested.start == pd.Timestamp("2026-02-01")
    assert plan.run is context.run
    print(
        f"SANITY: {table.name} uses its changed configured overlap, finite bounds and unchanged root anchor; a later stored suffix cannot move the frontier."
    )


def test_assembly_inherits_mixed_units_and_actual_upstream_recovery(sqlite_store: Any) -> None:
    config = fixture_config()
    config.data_aggregate.assembly.compulsory_parts = [Tables.cube_part_prices.name, Tables.cube_part_targets.name, Tables.cube_part_governance.name]
    config.data_aggregate.assembly.optional_parts = []
    sessions = pd.bdate_range("2010-01-01", "2026-11-16")
    sqlite_store.replace(Tables.cube_part_prices, pd.DataFrame({"ticker": "AAPL", "date": sessions, "close_split": 100.0}))
    for table in (Tables.cube_part_targets, Tables.cube_part_governance, Tables.cube):
        sqlite_store.replace(table, pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2026-11-13"]), "value": 1.0}))
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            config_dir=Path("configs"),
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "component-window inheritance", pd.Timestamp("2026-11-16")),
        ),
    )
    step = StepAssembleCube(context, config)

    ordinary = step._plan_window(None)
    assert ordinary is not None and ordinary.write.start == pd.Timestamp("2026-11-13") - pd.Timedelta(days=380)
    config.data.tables.cube_part_governance.reprocess.overlap.value = 4
    shorter_governance = step._plan_window(None)
    assert shorter_governance is not None and shorter_governance.write.start == sessions[sessions.get_loc("2026-11-13") - 90]
    written = {
        Tables.cube_part_prices: Window(pd.Timestamp("2019-01-02"), pd.Timestamp("2026-11-16")),
        Tables.cube_part_targets: Window(pd.Timestamp("2026-07-01"), pd.Timestamp("2026-11-16")),
        Tables.cube_part_governance: Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-16")),
    }
    actual = step._plan_window(written)
    assert actual is not None and actual.write.start == pd.Timestamp("2019-01-02")
    del written[Tables.cube_part_governance]
    with pytest.raises(ValueError, match="successful upstream write receipt"):
        step._plan_window(written)
    print(
        "SANITY: assembly compares actual380CD/90-session boundaries, changes with selected policies, covers an older actual price repair and refuses missing upstream evidence."
    )


def test_scoped_price_values_keep_the_complete_market_clock(sqlite_store: Any) -> None:
    config = fixture_config()
    sqlite_store.replace(
        Tables.cube_part_prices,
        pd.DataFrame(
            {
                "ticker": ["OTHER", "AAPL"],
                "date": pd.to_datetime(["2010-01-04", "2026-11-16"]),
                "close_split": [10.0, 100.0],
            }
        ),
    )
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "retained publication clock", pd.Timestamp("2026-11-16")),
        ),
    )
    calendar = load_trading_calendar(context)
    assert calendar.tolist() == list(pd.to_datetime(["2010-01-04", "2026-11-16"]))
    frames = load_price_frames(context, peers={}, fields=("close_split",), calendar=calendar)
    assert frames.trading_index.equals(calendar)
    assert frames.universe == ("AAPL",) and frames.close_split.loc[calendar[0]].isna().all()
    print(
        "SANITY: equity payloads remain ticker-scoped while the full retained market clock survives; a recent listing cannot reset older filing eligibility."
    )


def test_governance_source_read_obeys_its_configured_clock_and_scope(sqlite_store: Any) -> None:
    config = fixture_config()
    item = config.data.tables[Tables.def14a_llm.name]
    item.start_date, item.source_start, item.end_date = "2026-01-01", "2026-01-01", "2026-06-30"
    create_canonical(sqlite_store.engine, [Tables.def14a_llm])
    sqlite_store.save(
        Tables.def14a_llm,
        pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "OTHER"],
                "accession_number": ["old", "inside", "future", "foreign-scope"],
                "as_of": pd.to_datetime(["2025-05-01", "2026-05-01", "2026-12-01", "2026-05-01"]),
                "board_size": 10.0,
            }
        ),
    )
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            config_dir=Path("configs"),
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "configured governance source clock", pd.Timestamp("2026-11-16")),
        ),
    )

    df_source = StepCubeGovernance(context, config)._load_def14a(source_read_window(config, Tables.def14a_llm, context.run.as_of))

    assert df_source is not None and df_source.accession_number.tolist() == ["inside"]
    print("SANITY: governance reads only the configured source prefix/ceiling and ticker scope; a future filing cannot enter cleaning or history.")


def test_proxy_companion_scope_preserves_pay_inputs(sqlite_store: Any) -> None:
    config = fixture_config()
    table = Tables.def14a_executive_comp
    item = config.data.tables[table.name]
    item.start_date, item.source_start, item.end_date = "2026-01-01", "2026-01-01", "2026-06-30"
    create_canonical(sqlite_store.engine, [table])
    parents = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 3,
            "accession_number": ["old", "inside", "future"],
            "as_of": pd.to_datetime(["2025-05-01", "2026-05-01", "2026-12-01"]),
        }
    )
    sqlite_store.save(table, parents.assign(name="Known CEO", fiscal_year=2025, total=100.0, salary=50.0))
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            config_dir=Path("configs"),
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "inherited proxy companion dates", pd.Timestamp("2026-11-16")),
        ),
    )

    df_child = StepCubeGovernance(context, config)._load_proxy_child(table)

    assert df_child is not None and {"accession_number", "as_of", "total", "salary"} <= set(df_child.columns)
    assert df_child.accession_number.tolist() == ["inside"] and df_child.total.iloc[0] == 100.0
    print("SANITY: proxy companion reads retain their finite source bounds, Universe scope and the pay inputs required by governance.")


def test_upstream_receipts_refuse_changed_pipeline_configuration(sqlite_store: Any) -> None:
    config = fixture_config()
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "same-run write receipt", pd.Timestamp("2026-11-16")),
        ),
    )
    sqlite_store.replace(Tables.cube_part_prices, pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2026-11-13"]), "close_split": 100.0}))
    plan = plan_window(context, Tables.cube_part_prices, pd.bdate_range("2026-01-01", "2026-11-16"))
    assert plan is not None
    payload = json.dumps([write_receipt(context, {Tables.cube_part_prices: plan})])
    assert read_receipts(context, payload) == {Tables.cube_part_prices: plan.write}

    config.data.tables.cube_part_prices.reprocess.overlap.value += 1
    with pytest.raises(ValueError, match="configuration"):
        read_receipts(context, payload)
    print(
        "SANITY: upstream write footprints use the same run and data pipeline configuration; a mid-run policy change cannot mix silently into assembly."
    )


@pytest.fixture
def receipt_context_and_plan() -> tuple[Context, WindowPlan]:
    context = cast(
        Context,
        SimpleNamespace(
            config=fixture_config(),
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "exact receipt cutoff", pd.Timestamp("2026-11-10T15:00:00Z")),
        ),
    )
    interval = Window(pd.Timestamp("2026-11-03"), pd.Timestamp("2026-11-10"))
    plan = WindowPlan(interval, interval, interval, context.run, None, (("ticker", ("AAPL",)),))
    return context, plan


@pytest.mark.parametrize("operation", ["read", "write"])
def test_receipts_reject_a_different_same_day_instant(receipt_context_and_plan: tuple[Context, WindowPlan], operation: str) -> None:
    context, plan = receipt_context_and_plan
    payload = json.dumps([write_receipt(context, {Tables.cube_part_prices: plan})])
    context.run = WindowRun("incremental", "same day after market close", pd.Timestamp("2026-11-10T22:00:00Z"))
    assert context.run.as_of == plan.run.as_of and context.run.instant != plan.run.instant

    with pytest.raises(ValueError, match="run"):
        if operation == "read":
            read_receipts(context, payload)
        else:
            write_receipt(context, {Tables.cube_part_prices: plan})
    print(f"SANITY: receipt {operation} rejects a different original execution instant on the same New York market day.")


@pytest.mark.parametrize("as_of", ["2026-11-10T15:00:00Z", "2026-11-10T10:00:00-05:00"])
def test_receipts_round_trip_equivalent_aware_instants(receipt_context_and_plan: tuple[Context, WindowPlan], as_of: str) -> None:
    context, plan = receipt_context_and_plan
    record = json.loads(write_receipt(context, {Tables.cube_part_prices: plan}))
    context.run = WindowRun("incremental", "equivalent root cutoff", pd.Timestamp(as_of))

    assert pd.Timestamp(record["instant"]) == plan.run.instant
    assert read_receipts(context, json.dumps([record])) == {Tables.cube_part_prices: plan.write}
    assert pd.Timestamp(json.loads(write_receipt(context, {Tables.cube_part_prices: plan}))["instant"]) == plan.run.instant
    print("SANITY: matching and timezone-equivalent original cutoffs retain their full instant through receipt production and consumption.")


def test_receipts_reject_legacy_day_only_identity(receipt_context_and_plan: tuple[Context, WindowPlan]) -> None:
    context, plan = receipt_context_and_plan
    record = json.loads(write_receipt(context, {Tables.cube_part_prices: plan}))
    record.pop("instant", None)

    with pytest.raises(ValueError, match="run"):
        read_receipts(context, json.dumps([record]))
    print("SANITY: a legacy day-only receipt cannot certify an exact original execution cutoff.")


def test_receipts_allow_cold_component_promotion(sqlite_store: Any) -> None:
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=fixture_config(),
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "cold root component", pd.Timestamp("2026-11-10T15:00:00Z")),
        ),
    )
    plan = plan_window(context, Tables.cube_part_prices, pd.bdate_range("2010-01-01", "2026-11-10"))

    assert plan is not None and plan.run.mode == "initial" and context.run.mode == "incremental"
    assert plan.run.instant == context.run.instant
    record = json.loads(write_receipt(context, {Tables.cube_part_prices: plan}))
    assert record["mode"] == "incremental"
    assert read_receipts(context, json.dumps([record])) == {Tables.cube_part_prices: plan.write}
    print("SANITY: a cold component may promote to initial mode while preserving its incremental root's exact cutoff and receipt mode.")


@pytest.mark.parametrize("mutation", ["start", "end", "part", "duplicate", "day"])
def test_receipts_preserve_component_guards(receipt_context_and_plan: tuple[Context, WindowPlan], mutation: str) -> None:
    context, plan = receipt_context_and_plan
    record = json.loads(write_receipt(context, {Tables.cube_part_prices: plan}))
    item = record["parts"][Tables.cube_part_prices.name]
    records = [record]
    if mutation == "start":
        item["start"] = "1900-01-01"
    elif mutation == "end":
        item["end"] = "2026-11-11"
    elif mutation == "part":
        record["parts"] = {"unknown_part": item}
    elif mutation == "duplicate":
        records.append(record)
    else:
        record["as_of"] = "2026-11-09"

    with pytest.raises(ValueError):
        read_receipts(context, json.dumps(records))
    print(f"SANITY: exact-cutoff receipts retain the existing {mutation} guard.")


def test_13f_actual_book_path_uses_two_day_lag_and_ninety_day_origin() -> None:
    config = fixture_config()
    age = int(config.data_aggregate.sec_thirteen.superinvestor.max_age_calendar_days)
    assert age == 90
    sessions = pd.bdate_range("2026-01-02", "2026-09-30").difference(
        pd.to_datetime(["2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07"])
    )
    packets = [
        _packet("0000000001", "2026-03-31", "2026-05-12", {"111111111": 100.0}),
        _packet("0000000001", "2026-03-31", "2026-08-20", {"111111111": 120.0}, "restatement"),
    ]
    # The production reader emits complete CIK-quarter units, including every amendment.
    unit = tuple(pd.concat([packet[index] for packet in packets], ignore_index=True) for index in (0, 1))
    history = prepare_superinvestor_book_history(iter([unit]), sessions, age, config)
    assert history is not None
    assert history[0].day == pd.Timestamp("2026-05-14")
    assert "0000000001" in _on(history, "2026-08-12").level_books
    assert not _on(history, "2026-08-13").level_books
    amended = _on(history, "2026-08-24")
    assert not amended.level_books
    assert amended.archive[("0000000001", pd.Timestamp("2026-03-31"))].first_public_at == pd.Timestamp("2026-05-14")
    print("SANITY: actual13F book preparation uses filing+2CD, keeps day90, expires on day91 and does not restart age for a later amendment.")


@pytest.mark.parametrize("lag", [1, 4])
def test_price_counts_respect_filing_buffer(sqlite_store: Any, tmp_path: Path, lag: int) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    context.config = fixture_config()
    context.config.data.tables.sharadar_fundamentals_facts.pit.lag.value = lag
    context.run = WindowRun("full", "share-count eligibility check", pd.Timestamp("2022-12-01"))
    StepCubePrices(context, context.config).run()
    df_saved = sqlite_store.load(Tables.cube_part_prices)
    df_a = df_saved.loc[df_saved.ticker.eq("AAA")].sort_values("date")
    eligible = pd.Timestamp("2020-01-02") + pd.Timedelta(days=lag)
    assert df_a.loc[pd.to_datetime(df_a.date) < eligible, "shares_outstanding_pit"].isna().all()
    assert df_a.loc[pd.to_datetime(df_a.date) >= eligible, "shares_outstanding_pit"].notna().all()
    assert df_a.loc[pd.to_datetime(df_a.date) >= eligible, "mcap_pit"].eq(10_000.0).all()
    print(
        "SANITY: counts and capitalization start only after the configured filing buffer; split-basis arithmetic still uses the original observation date."
    )


def test_assembly_respects_each_component_configured_dates(sqlite_store: Any) -> None:
    config = fixture_config()
    config.data.tables.cube_part_momentum.start_date = "2026-05-05"
    config.data.tables.cube_part_momentum.end_date = "2026-05-06"
    dates = pd.to_datetime(["2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07"])
    sqlite_store.replace(Tables.cube_part_momentum, pd.DataFrame({"ticker": "AAPL", "date": dates, "f_momentum": 1.0}))
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            config_dir=Path("configs"),
            universe=universe_of(["AAPL"]),
            run=WindowRun("full", "finite component read", pd.Timestamp("2026-11-16")),
        ),
    )
    panel = StepAssembleCube(context, config)._read_part(Tables.cube_part_momentum, None, Window(dates[0], dates[-1]), optional=False)
    assert panel is not None and pd.to_datetime(panel.date).tolist() == list(dates[1:3])
    print("SANITY: assembly cannot revive component values stored outside that component's configured start and end dates.")


@pytest.mark.parametrize("tickers", [[], ["AAPL"]])
def test_receipts_reject_partial_scope(sqlite_store: Any, tickers: list[str]) -> None:
    config = fixture_config()
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL", "MSFT"]),
            run=WindowRun("incremental", "complete receipt scope", pd.Timestamp("2026-11-16")),
        ),
    )
    sqlite_store.replace(
        Tables.cube_part_prices, pd.DataFrame({"ticker": ["AAPL", "MSFT"], "date": pd.to_datetime(["2026-11-13"] * 2), "close_split": 100.0})
    )
    plan = plan_window(context, Tables.cube_part_prices, pd.bdate_range("2026-01-01", "2026-11-16"))
    assert plan is not None
    record = json.loads(write_receipt(context, {Tables.cube_part_prices: plan}))
    record["parts"][Tables.cube_part_prices.name]["tickers"] = tickers
    with pytest.raises(ValueError, match="scope|ticker"):
        read_receipts(context, json.dumps([record]))
    print("SANITY: empty or partial ticker receipts cannot certify assembly for a larger requested universe.")


def test_receipts_reject_a_different_root_mode(sqlite_store: Any) -> None:
    config = fixture_config()
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "incremental receipt", pd.Timestamp("2026-11-16")),
        ),
    )
    sqlite_store.replace(Tables.cube_part_prices, pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2026-11-13"]), "close_split": 100.0}))
    plan = plan_window(context, Tables.cube_part_prices, pd.bdate_range("2026-01-01", "2026-11-16"))
    assert plan is not None
    payload = json.dumps([write_receipt(context, {Tables.cube_part_prices: plan})])
    context.run = WindowRun("full", "different root mode", context.run.instant)
    with pytest.raises(ValueError, match="mode|run"):
        read_receipts(context, payload)
    print("SANITY: assembly refuses an upstream result produced with a different root execution mode.")


def test_incremental_price_refuses_undeclared_historical_recovery(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    context.run = WindowRun("full", "initial known-truth price history", context.run.instant)
    StepCubePrices(context, context.config).run()
    before = sqlite_store.load(Tables.cube_part_prices)
    source = sqlite_store.load(Tables.prices)
    days = pd.DatetimeIndex(source.date.unique()).sort_values()
    selected = source.ticker.eq("BBB") & pd.to_datetime(source.date).eq(days[10])
    source.loc[selected, ["close_split", "close_total"]] = 11.0
    sqlite_store.replace(Tables.prices, source)
    context.run = WindowRun("incremental", "ordinary configured correction tail", context.run.instant)
    with pytest.raises(ValueError, match="historical|rebuild"):
        StepCubePrices(context, context.config).run()
    pd.testing.assert_frame_equal(before, sqlite_store.load(Tables.cube_part_prices))
    context.run = WindowRun("full", "explicit historical cube rebuild", context.run.instant)
    StepCubePrices(context, context.config).run()
    actual = sqlite_store.load(Tables.cube_part_prices)
    assert actual.loc[actual.ticker.eq("BBB") & pd.to_datetime(actual.date).eq(days[10]), "close_split"].iloc[0] == 11.0
    print(
        "SANITY: an older price correction is refused before mutation in normal mode and succeeds in explicit full mode, so downstream historical features cannot silently remain stale."
    )


def test_schedule_frontier_ignores_rows_after_the_root_source_end(sqlite_store: Any) -> None:
    config = fixture_config()
    sqlite_store.save(
        Tables.sec_13g,
        pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "accession_number": ["early", "future"],
                "rp_seq": [0, 0],
                "filing_date": pd.to_datetime(["2026-01-02", "2026-11-02"]),
            }
        ),
    )
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "bounded completeness frontier", pd.Timestamp("2026-05-11")),
        ),
    )
    latest = schedule_complete_through(context, Tables.sec_13g, context.run.as_of)
    assert latest is not None and latest <= context.run.as_of
    print("SANITY: a future stored filing cannot advance a historical source completeness frontier beyond the configured root cutoff.")


def test_truncated_source_calendar_cannot_hide_a_later_saved_frontier(sqlite_store: Any) -> None:
    config = fixture_config()
    sqlite_store.save(Tables.cube_part_governance, pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2026-11-13"]), "value": [1.0]}))
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "incomplete source calendar", pd.Timestamp("2026-11-16")),
        ),
    )
    with pytest.raises(RuntimeError, match="calendar|frontier"):
        plan_window(context, Tables.cube_part_governance, pd.bdate_range("2026-01-01", "2026-11-10"))
    print(
        "SANITY: an incomplete source calendar cannot hide a later in-scope saved frontier and silently reset incremental work to an initial build."
    )


@pytest.mark.parametrize("table", [Tables.cube_part_governance, Tables.cube_part_fundamentals, Tables.cube_part_momentum, Tables.cube_part_betas])
def test_undersized_configured_warmup_is_rejected_even_on_short_history(sqlite_store: Any, table: Table) -> None:
    config = fixture_config()
    config.data.tables[table.name].warmup.grid.value = 1
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("full", "short actual history", pd.Timestamp("2026-11-16")),
        ),
    )
    with pytest.raises(ValueError, match="warm-up"):
        plan_window(context, table, pd.DatetimeIndex(["2026-11-16"]))
    print(f"SANITY: {table.name} rejects an insufficient configured warm-up even when the market clock is naturally short.")


def test_financial_warmup_covers_the_configured_history_calculation(sqlite_store: Any) -> None:
    config = fixture_config()
    config.data_aggregate.hist.window = 2520
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("full", "configured long financial history", pd.Timestamp("2026-11-16")),
        ),
    )
    with pytest.raises(ValueError, match="warm-up"):
        plan_window(context, Tables.cube_part_fundamentals, pd.bdate_range("2010-01-01", "2026-11-16"))
    print("SANITY: a longer configured financial history cannot silently exceed the configured computation warm-up.")
