"""Actual projected reads honor per-input calendar budgets and explicit full history."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_aggregate.utils.common.incremental import plan_window
from src.data_aggregate.utils.common.inputs import input_windows, load_source
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.window import Window, WindowRun
from tests.canonical_schema import create_canonical
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
