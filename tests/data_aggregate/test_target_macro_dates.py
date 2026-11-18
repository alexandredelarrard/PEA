"""Known-truth SQL DATE/calendar boundary for the target macro loader."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.constants.constants_price import DAILY_MACRO_LEVELS, MACRO_CUBE_FACTORS, MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.macro import load_macro_wide
from src.utils.window import Window, WindowRun
from tests.support.config import fixture_config
from tests.support.universe import universe_of


def _context(store: DataStore) -> Context:
    config = fixture_config()
    policy = config.data.tables[Tables.prices_macro.name]
    policy.start_date = "2026-01-06"
    policy.source_start = "2026-01-02"
    policy.end_date = "2026-01-08"
    return cast(
        Context,
        SimpleNamespace(
            store=store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "macro date contract", pd.Timestamp("2026-01-09")),
        ),
    )


def _source_rows() -> pd.DataFrame:
    days = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"])
    df_vix = pd.DataFrame({"date": days, "ticker": "vix", "close": [900.0, 18.0, 20.0, 24.0, None, 25.5, 999.0]})
    df_market = pd.DataFrame({"date": days, "ticker": MACRO_MARKET_SERIES, "close": [900.0, 100.0, 101.0, 102.0, 103.0, 104.0, 999.0]})
    df_outside = pd.DataFrame({"date": [days[3]], "ticker": ["OUTSIDE"], "close": [123.0]})
    return pd.concat([df_vix, df_market, df_outside], ignore_index=True)


@pytest.mark.parametrize("timestamp_payload", [False, True], ids=["native-sql-date", "timestamp-boundary"])
def test_macro_loader_normalizes_dates_without_changing_scope_or_factors(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, timestamp_payload: bool
) -> None:
    """Preserve the old adapter's dates and exact sparse-calendar change values."""
    # Use the existing canonical DATE schema and the actual store for every source row.
    context = _context(sqlite_store)
    context.store.create_declared(Tables.prices_macro)
    context.store.save(Tables.prices_macro, _source_rows())
    calendar = pd.bdate_range("2026-01-02", "2026-01-08", name="date")
    source_window = Window(pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-08"))
    series = sorted({MACRO_MARKET_SERIES, *MACRO_CUBE_FACTORS.values(), *DAILY_MACRO_LEVELS})
    calls: list[tuple[Table, dict[str, Any]]] = []
    native_types: set[type] = set()
    iter_load = context.store.iter_load

    # Observe the real DATE read; separately exercise the Timestamp parser boundary.
    def observe(table: Table, **kwargs: Any) -> Iterator[pd.DataFrame]:
        calls.append((table, kwargs))
        for df_chunk in iter_load(table, **kwargs):
            native_types.update(type(value) for value in df_chunk["date"])
            yield df_chunk.assign(date=pd.to_datetime(df_chunk["date"])) if timestamp_payload else df_chunk

    monkeypatch.setattr(context.store, "iter_load", observe)
    step = object.__new__(StepCubeTarget)
    step._context = context
    df_actual = step._load_macro()
    print(
        f"BOUNDARY: SQL types={sorted(kind.__name__ for kind in native_types)}; {timestamp_payload=}; index={type(df_actual.index).__name__}/{df_actual.index.dtype}; element={type(df_actual.index[0]).__name__}; unique={df_actual.index.is_unique}"
    )
    assert native_types == {date}
    df_changes = step._macro_changes(df_actual, calendar)

    # Compare the previous adapter on the same retained source prefix and exact known differences.
    df_prior = load_macro_wide(context.store, series=series, since=source_window.start)
    assert df_prior is not None
    df_prior = df_prior.set_index("date").loc[: source_window.end].rename_axis(columns="ticker")
    assert isinstance(df_actual.index, pd.DatetimeIndex)
    pd.testing.assert_frame_equal(df_actual, df_prior, check_exact=True)
    pd.testing.assert_frame_equal(df_changes, step._macro_changes(df_prior, calendar), check_exact=True)
    df_expected = pd.DataFrame({DAILY_MACRO_LEVELS["vix"]: [float("nan"), 2.0, 4.0, 0.0, 1.5]}, index=calendar)
    pd.testing.assert_frame_equal(df_changes, df_expected, check_exact=True)
    assert calls == [
        (
            Tables.prices_macro,
            {"project": False, "columns": ["date", "ticker", "close"], "where": {"ticker": series, "date": source_window}, "markers": False},
        )
    ]
    print(
        "SANITY: native DATE and Timestamp payloads retain one scoped projected read, causal prefix, dates/nulls/precision and exact prior-loader macro changes."
    )
