"""Actual projected reads honor per-input calendar budgets and explicit full history."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.utils.common.incremental import plan_window
from src.data_aggregate.utils.common.inputs import input_windows, load_source
from src.data_aggregate.utils.institutionals.ownership_features import ScheduleSource, SubjectIdentity, build_ownership_feature_panel
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.window import Window, WindowRun
from tests.canonical_schema import create_canonical
from tests.conftest import make_frames
from tests.support.config import fixture_config
from tests.support.universe import universe_of


@pytest.mark.parametrize("amount", [None, 0, 7])
def test_actual_input_read_uses_rewrite_start_not_source_max(sqlite_store: DataStore, amount: int | None, monkeypatch: pytest.MonkeyPatch) -> None:
    config = fixture_config()
    config.data.tables.cube_part_targets.warmup.prices_macro = amount
    dates = pd.to_datetime(["1990-01-02", "2026-09-24", "2026-09-30", "2026-10-01", "2026-11-16", "2026-11-17"])
    sqlite_store.save(Tables.prices_macro, pd.DataFrame({"ticker": "market", "date": dates, "close": 100.0}))
    context = SimpleNamespace(
        config=config,
        store=sqlite_store,
        universe=universe_of(["AAPL"]),
        run=WindowRun("incremental", "lagging cube/current source", pd.Timestamp("2026-11-16")),
    )
    write = Window(pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-16"))
    windows = input_windows(context, Tables.cube_part_targets, write)
    calls = []
    original = sqlite_store.iter_load

    def traced(table: Table, **kwargs: Any) -> Iterator[pd.DataFrame]:
        calls.append(kwargs)
        return original(table, **kwargs)

    monkeypatch.setattr(sqlite_store, "iter_load", traced)
    rows = load_source(context, Tables.prices_macro, ["market"], columns=["ticker", "date", "close"], window=windows[Tables.prices_macro])
    expected = dates[dates <= context.run.as_of]
    if amount is not None:
        expected = expected[expected >= write.start - pd.Timedelta(days=amount)]
    assert rows is not None and pd.to_datetime(rows.date).tolist() == list(expected)
    assert len(calls) == 1 and calls[0]["until"] == context.run.as_of
    assert calls[0]["since"] == (None if amount is None else write.start - pd.Timedelta(days=amount))
    assert calls[0]["where"] == {"ticker": ["market"]}
    print(f"SANITY: input budget {amount!r} reaches one projected source-clock read anchored at actual rewrite start; future rows stay excluded.")


def test_archive_period_inventory_filters_payload_before_loading(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    create_canonical(sqlite_store.engine, [Tables.sec_fails_to_deliver_security])
    sqlite_store.save(
        Tables.sec_fails_to_deliver_security,
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02"] * 4),
                "cusip": ["A", "B", "C", "D"],
                "source_symbol": "AAPL",
                "period": ["202609a", "202609b", "202610a", "invalid"],
                "fails_quantity": 1.0,
                "price": 1.0,
                "fails_value": 1.0,
            }
        ),
    )
    context = SimpleNamespace(
        config=fixture_config(),
        store=sqlite_store,
        universe=universe_of(["AAPL"]),
        run=WindowRun("incremental", "source-period pushdown", pd.Timestamp("2026-10-16")),
    )
    calls = []
    original = sqlite_store.iter_load

    def traced(table: Table, **kwargs: Any) -> Iterator[pd.DataFrame]:
        calls.append(kwargs)
        return original(table, **kwargs)

    monkeypatch.setattr(sqlite_store, "iter_load", traced)
    rows = load_source(
        context,
        Tables.sec_fails_to_deliver_security,
        columns=["date", "cusip", "period", "fails_quantity"],
        window=Window(pd.Timestamp("2026-09-30"), context.run.as_of),
    )
    assert rows is not None and set(rows.period) == {"202609b", "202610a"}
    assert len(calls) == 1 and set(calls[0]["where"]["period"]) == {"202609b", "202610a"}
    assert "since" not in calls[0] and "until" not in calls[0]
    print(
        "SANITY: string archive periods are normalized as compact metadata and selected before payload I/O; settlement date is not substituted for the configured period clock."
    )


def test_target_calendar_overlap_refuses_an_insufficient_real_session_span(sqlite_store: DataStore) -> None:
    config = fixture_config()
    sessions = pd.bdate_range("2026-01-02", "2026-11-16").difference(
        pd.to_datetime(["2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07"])
    )
    sqlite_store.replace(Tables.cube_part_targets, pd.DataFrame({"ticker": "AAPL", "date": [sessions[-1]], "target": 1.0}))
    context = SimpleNamespace(
        config=config, store=sqlite_store, universe=universe_of(["AAPL"]), run=WindowRun("incremental", "target maturity coverage", sessions[-1])
    )
    config.data.tables.cube_part_targets.overlap.days = 110
    with pytest.raises(ValueError, match="110.*90-session"):
        plan_window(context, Tables.cube_part_targets, sessions)
    config.data.tables.cube_part_targets.overlap.days = 150
    plan = plan_window(context, Tables.cube_part_targets, sessions)
    assert plan is not None and plan.write.start <= sessions[-91]
    print(
        "SANITY: a 110-calendar-day rewrite cannot cover 90 actual retained market sessions across holidays; 150 days covers the target horizon without widening the configured write."
    )


@pytest.fixture(params=[Tables.sec_13d, Tables.sec_13g], ids=["13d", "13g"])
def ownership_history(sqlite_store: DataStore, request: pytest.FixtureRequest) -> SimpleNamespace:
    config = fixture_config()
    assert config.data.tables.cube_part_institutionals.warmup.sec_13d is None
    assert config.data.tables.cube_part_institutionals.warmup.sec_13g is None
    tickers = ["AAPL", "AMZN", "JPM", "MSFT", "XOM"]
    sessions = pd.bdate_range("2026-05-01", "2026-12-10").difference(
        pd.to_datetime(["2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26"])
    )
    records = [("AAPL", f"0000999999-22-{number:06d}", -1, "2022-01-03", "") for number in range(1, 101)]
    records.extend(
        [
            ("AAPL", "0000999999-26-000001", -1, "2026-11-09", ""),
            ("AAPL", "0000888888-26-000001", 0, "2026-11-09", "AAPL00001"),
            ("AMZN", "0000777777-26-000001", 0, "2026-11-09", "AMZN00001"),
            ("JPM", "0000666666-26-000001", 0, "2026-11-13", "JPM000001"),
        ]
    )
    df_source = pd.DataFrame(records, columns=["ticker", "accession_number", "rp_seq", "filing_date", "cusip"])
    df_source = df_source.assign(filing_date=pd.to_datetime(df_source.filing_date), reporting_person_cik="")
    sqlite_store.save(request.param, df_source)
    # A saved frontier yields the ordinary 30-calendar-day incremental rewrite.
    sqlite_store.save(Tables.cube_part_institutionals, pd.DataFrame({"ticker": "AAPL", "date": [sessions[-1]], "value": 1.0}))
    context = cast(
        Context,
        SimpleNamespace(
            config=config,
            config_dir=Path(__file__).resolve().parents[2] / "configs",
            store=sqlite_store,
            universe=universe_of(tickers),
            run=WindowRun("incremental", "full-history subject identity", sessions[-1]),
        ),
    )
    plan = plan_window(context, Tables.cube_part_institutionals, sessions)
    assert plan is not None and plan.write.start == pd.Timestamp("2026-11-10")
    frames = make_frames(sessions, peers={ticker: {other: 1.0 for other in tickers if other != ticker} for ticker in tickers})
    df_lineage = pd.DataFrame({"canonical_ticker": tickers, "cik": [f"{number:010d}" for number in range(1, 6)], "role": "cik_window"})
    df_lines = pd.DataFrame(
        {
            "canonical_company": tickers,
            "cusip": ["AAPL00001", "AMZN00001", "JPM000001", "MSFT00001", "XOM000001"],
            "lineage_role": "canonical_current",
        }
    )
    return SimpleNamespace(
        context=context, table=request.param, df_source=df_source, plan=plan, frames=frames, df_lineage=df_lineage, df_lines=df_lines
    )


@pytest.mark.parametrize("amount", [0, 750])
def test_ownership_finite_history_refuses_before_payload_reads(
    ownership_history: SimpleNamespace, amount: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = ownership_history
    case.context.config.data.tables.cube_part_institutionals.warmup[case.table.name] = amount
    windows = input_windows(case.context, Tables.cube_part_institutionals, case.plan.write)
    reads = []
    original = case.context.store.iter_load

    def traced(table: Table, **kwargs: Any) -> Iterator[pd.DataFrame]:
        reads.append(table)
        return original(table, **kwargs)

    monkeypatch.setattr(case.context.store, "iter_load", traced)
    with pytest.raises(ValueError, match=rf"{case.table.name}.*full.*null warmup"):
        StepCubeInstitutionals(case.context, case.context.config)._ownership_panel(case.frames, case.df_lines, case.df_lineage, windows)
    assert not reads
    print(f"SANITY: {case.table.name} refuses finite {amount}CD before either ownership payload read; no history is silently widened.")


@pytest.mark.parametrize("old_markers", [True, False], ids=["old-markers", "naturally-short"])
def test_ownership_null_history_preserves_full_subject_reference(ownership_history: SimpleNamespace, old_markers: bool) -> None:
    case = ownership_history
    if not old_markers:
        case.df_source = case.df_source.loc[case.df_source.filing_date.ge("2026-01-01")]
        case.context.store.replace(case.table, case.df_source)
    windows = input_windows(case.context, Tables.cube_part_institutionals, case.plan.write)
    assert windows[Tables.sec_13d] is None and windows[Tables.sec_13g] is None
    step = StepCubeInstitutionals(case.context, case.context.config)
    df_actual = step._ownership_panel(case.frames, case.df_lines, case.df_lineage, windows)
    df_reference = build_ownership_feature_panel(
        case.frames,
        [ScheduleSource(case.table, case.df_source, case.df_source.filing_date.max())],
        case.context.config,
        availability=step._availability,
        identity=SubjectIdentity.from_tables(case.df_lineage, case.df_lines),
    )
    assert df_actual is not None
    pd.testing.assert_frame_equal(df_actual, df_reference, check_exact=True)
    field = "f_ic_act_13d_count_126d_z" if case.table is Tables.sec_13d else "f_ic_bo_13g_count_126d_z"
    first_active = df_actual.loc[df_actual.date.eq(case.plan.write.start)].set_index("ticker")[field]
    expected = [1.0954451561, 1.0954451561] if old_markers else [1.5652475357, 0.4472135901]
    assert first_active.reindex(["AAPL", "AMZN"]).tolist() == pytest.approx(expected)
    assert case.plan.write.start == case.context.run.as_of - pd.Timedelta(days=30) and case.plan.write.end == case.context.run.as_of
    assert len(case.df_source) == (104 if old_markers else 4)
    print(
        f"SANITY: {case.table.name} null input matches full subject filtering with {len(case.df_source)} genuine stored rows; "
        "old identity markers change classification, naturally short history is valid, and the planned write stays30CD."
    )


def test_input_read_log_gives_scope_sizes_not_members(sqlite_store: DataStore, caplog: pytest.LogCaptureFixture) -> None:
    sqlite_store.save(Tables.prices_macro, pd.DataFrame({"ticker": "market", "date": pd.to_datetime(["2026-11-16"]), "close": 100.0}))
    context = SimpleNamespace(
        config=fixture_config(), store=sqlite_store, universe=universe_of(["AAPL"]), run=WindowRun("incremental", "test", pd.Timestamp("2026-11-16"))
    )
    with caplog.at_level("INFO"):
        load_source(context, Tables.prices_macro, ["market", "AAPL", "MSFT"], columns=["ticker", "date", "close"])  # type: ignore[arg-type]
    line = next(record.getMessage() for record in caplog.records if record.getMessage().startswith("Input read"))
    assert "3 value(s)" in line and "MSFT" not in line
    print(f"SANITY: the read log shows the ticker scope as a count, no member names: {line}")
