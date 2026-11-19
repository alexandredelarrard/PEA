"""Configured boundaries exercised through actual cube producers and shared planning."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.transformers.step_cube_governance import StepCubeGovernance
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.utils.common.incremental import plan_window
from src.data_aggregate.utils.common.parts import CUBE_PARTS
from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_aggregate.utils.price.frames import load_price_frames, load_trading_calendar
from src.data_store.schema import Table, Tables
from src.utils.window import Window, WindowRun
from tests.canonical_schema import create_canonical
from tests.data_aggregate.test_price_pit_contract import seed_prices
from tests.support.config import fixture_config
from tests.support.universe import universe_of


def test_actual_price_build_obeys_configured_dates(sqlite_store: Any, tmp_path: Path) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    context.config = fixture_config()
    context.run = WindowRun("full", "observed price-date rebuild", pd.Timestamp("2022-10-21"))

    StepCubePrices(context, context.config).run()

    df_saved = sqlite_store.load(Tables.cube_part_prices)
    assert not df_saved.empty
    assert pd.to_datetime(df_saved.date).between("2020-01-02", "2022-10-21").all(), (df_saved.date.min(), df_saved.date.max())
    assert pd.to_datetime(df_saved.date).min() == pd.Timestamp("2020-01-02")
    assert set(df_saved.ticker) == {"AAA", "BBB"}
    print("SANITY: the actual price producer publishes only its configured finite dates and scoped tickers, including full mode.")


@pytest.mark.parametrize("table", [part.table for part in CUBE_PARTS], ids=lambda table: table.name)
def test_every_part_uses_configured_overlap_and_ignores_future_frontier(sqlite_store: Any, table: Table) -> None:
    config = fixture_config()
    item = config.data.tables[table.name]
    sessions = pd.bdate_range("2010-01-01", "2026-12-31")
    overlap = 150 if table == Tables.cube_part_targets else 9
    item.overlap.days = overlap
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
    expected = pd.Timestamp("2026-10-30") - pd.Timedelta(days=overlap)
    assert plan.write.start == expected and plan.write.end == pd.Timestamp("2026-11-16")
    assert plan.requested.start == sessions[0]
    assert plan.run is context.run
    print(
        f"SANITY: {table.name} uses its changed configured overlap, finite bounds and unchanged root anchor; a later stored suffix cannot move the frontier."
    )


def test_assembly_inherits_mixed_units_and_actual_upstream_recovery(sqlite_store: Any) -> None:
    config = fixture_config()
    config.data_aggregate.assembly.compulsory_parts = [Tables.cube_part_prices.name, Tables.cube_part_targets.name, Tables.cube_part_governance.name]
    config.data_aggregate.assembly.optional_parts = []
    config.data.tables.cube.warmup = {name: 0 for name in config.data_aggregate.assembly.compulsory_parts}
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
    config.data.tables.cube_part_governance.overlap.days = 4
    shorter_governance = step._plan_window(None)
    assert shorter_governance is not None and shorter_governance.write.start == pd.Timestamp("2026-11-13") - pd.Timedelta(days=150)
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

    df_source = StepCubeGovernance(context, config)._load_def14a(Window(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30")))

    assert df_source is not None and df_source.accession_number.tolist() == ["inside"]
    print("SANITY: governance reads only the configured source prefix/ceiling and ticker scope; a future filing cannot enter cleaning or history.")


def test_proxy_companion_scope_preserves_pay_inputs(sqlite_store: Any) -> None:
    config = fixture_config()
    table = Tables.def14a_executive_comp
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

    df_child = StepCubeGovernance(context, config)._load_proxy_child(table, Window(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30")))

    assert df_child is not None and {"accession_number", "as_of", "total", "salary"} <= set(df_child.columns)
    assert df_child.accession_number.tolist() == ["inside"] and df_child.total.iloc[0] == 100.0
    print("SANITY: proxy companion reads retain their finite source bounds, Universe scope and the pay inputs required by governance.")


@pytest.mark.parametrize("lag", [1, 4])
def test_price_counts_respect_filing_buffer(sqlite_store: Any, tmp_path: Path, lag: int) -> None:
    context = seed_prices(sqlite_store, tmp_path)
    context.config = fixture_config()
    context.config.data.tables.sharadar_fundamentals_history.pit.lag_days = lag
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
    panel = StepAssembleCube(context, config)._read_part(Tables.cube_part_momentum, None, Window(dates[1], dates[2]), optional=False)
    assert panel is not None and pd.to_datetime(panel.date).tolist() == list(dates[1:3])
    print("SANITY: assembly cannot revive component values stored outside that component's configured start and end dates.")


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
    config.data.tables[table.name].warmup.cube_part_prices = 1
    context = cast(
        Context,
        SimpleNamespace(
            store=sqlite_store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("full", "short actual history", pd.Timestamp("2026-11-16")),
        ),
    )
    with pytest.raises(ValueError, match="warm.?up"):
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
    with pytest.raises(ValueError, match="warm.?up"):
        plan_window(context, Tables.cube_part_fundamentals, pd.bdate_range("2010-01-01", "2026-11-16"))
    print("SANITY: a longer configured financial history cannot silently exceed the configured computation warm-up.")
