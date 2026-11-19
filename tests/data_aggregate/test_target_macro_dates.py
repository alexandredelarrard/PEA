"""Known-truth SQL DATE/calendar boundary for the target macro loader."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.constants.constants_price import DAILY_MACRO_LEVELS, MACRO_CUBE_FACTORS, MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.transformers import step_cube_target as target_step
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.macro import load_macro_wide
from src.utils.publication import point_in_time
from src.utils.window import Window, WindowRun
from tests.support.config import fixture_config
from tests.support.universe import universe_of


def _context(store: DataStore, as_of: str = "2026-01-09") -> Context:
    config = fixture_config()
    return cast(
        Context,
        SimpleNamespace(
            store=store,
            config=config,
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "macro date contract", pd.Timestamp(as_of)),
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
    """Preserve source scope and sparse-calendar changes while publishing on sessions."""
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
    step._config = context.config
    df_actual = step._load_macro(calendar, source_window)
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
    df_expected_macro = df_prior.rename(index={pd.Timestamp("2026-01-03"): pd.Timestamp("2026-01-05")})
    pd.testing.assert_frame_equal(df_actual, df_expected_macro, check_exact=True)
    pd.testing.assert_frame_equal(df_changes, step._macro_changes(df_prior, calendar), check_exact=True)
    df_expected = pd.DataFrame({DAILY_MACRO_LEVELS["vix"]: [float("nan"), 2.0, 4.0, 0.0, 1.5]}, index=calendar)
    pd.testing.assert_frame_equal(df_changes, df_expected, check_exact=True)
    assert calls == [
        (
            Tables.prices_macro,
            {
                "project": False,
                "columns": ["date", "ticker", "close"],
                "where": {"ticker": series},
                "markers": False,
                "since": source_window.start,
                "until": source_window.end,
                "date_col": "date",
            },
        )
    ]
    print(
        "SANITY: native DATE and Timestamp payloads retain one scoped projected read, causal prefix, dates/nulls/precision and exact prior-loader macro changes."
    )


@pytest.mark.parametrize(
    ("lag_days", "eligible_dates", "november_10_return"),
    [
        (0, ["2026-11-06", "2026-11-09", "2026-11-10"], 1.0),
        (1, ["2026-11-09", "2026-11-10", "2026-11-11"], 1.0),
        (3, ["2026-11-09", "2026-11-12", "2026-11-13"], 0.0),
    ],
    ids=["lag-0", "lag-1", "lag-3"],
)
def test_macro_loader_consumes_publication_lag(sqlite_store: DataStore, lag_days: int, eligible_dates: list[str], november_10_return: float) -> None:
    """A configured macro publication lag changes actual target factor operands."""
    context = _context(sqlite_store, "2026-11-13")
    context.config.data.tables[Tables.prices_macro.name].pit.lag_days = lag_days
    context.store.create_declared(Tables.prices_macro)
    context.store.save(
        Tables.prices_macro,
        pd.DataFrame(
            {"date": pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10"]), "ticker": MACRO_MARKET_SERIES, "close": [100.0, 200.0, 400.0]}
        ),
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]), name="date")
    source_window = Window(calendar[0], calendar[-1])
    step = object.__new__(StepCubeTarget)
    step._context = context
    step._config = context.config

    df_macro = step._load_macro(calendar, source_window)
    actual_return = step._market_return(df_macro, calendar).loc["2026-11-10"]
    print(f"BOUNDARY: {lag_days=}; eligible={df_macro.index.tolist()}; November 10 market return={actual_return}; expected={november_10_return}")
    assert actual_return == november_10_return
    pd.testing.assert_index_equal(df_macro.index, pd.DatetimeIndex(pd.to_datetime(eligible_dates), name="date"))
    assert df_macro[MACRO_MARKET_SERIES].tolist() == [100.0, 200.0, 400.0]
    print("SANITY: target macro inputs use source date plus the configured calendar lag and the next retained market session exactly once.")


def test_macro_publication_preserves_latest_quotes_and_mixed_operands(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Holiday/prefix collisions choose the newest quote, with invalid/future rows excluded."""
    context = _context(sqlite_store, "2026-07-10")
    context.config.data.tables[Tables.prices_macro.name].pit.lag_days = 1
    days = pd.to_datetime(
        ["2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06", "2026-07-07", "2026-07-10", "2026-07-11"]
    )
    closes = pd.Series([80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 999.0, 150.0, 160.0])
    df_source = pd.concat(
        [
            pd.DataFrame({"date": days, "ticker": series, "close": closes * multiplier})
            for series, multiplier in [(MACRO_MARKET_SERIES, 1), ("vix", 0.1), ("oil", 2), ("fx_usdeur", 0.01)]
        ],
        ignore_index=True,
    )
    context.store.create_declared(Tables.prices_macro)
    context.store.save(Tables.prices_macro, df_source)
    calendar = pd.DatetimeIndex(["2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10"], name="date")
    payloads: list[pd.DataFrame] = []
    prepared: list[pd.DataFrame] = []
    reads: list[dict[str, Any]] = []
    iter_load = context.store.iter_load

    # Malformed adapter values are tested after a genuine bounded SQL DATE read.
    def observe(table: Table, **kwargs: Any) -> Iterator[pd.DataFrame]:
        reads.append(kwargs)
        for df_chunk in iter_load(table, **kwargs):
            df_chunk = df_chunk.iloc[::-1].copy()
            df_chunk["date"] = df_chunk["date"].astype(object)
            df_chunk.loc[df_chunk["date"] == date(2026, 7, 7), "date"] = "invalid"
            payloads.append(df_chunk.copy(deep=True))
            yield df_chunk

    def observe_publication(*args: Any, **kwargs: Any) -> pd.DataFrame:
        df_ready = point_in_time(*args, **kwargs)
        prepared.append(df_ready.copy(deep=True))
        return df_ready

    monkeypatch.setattr(context.store, "iter_load", observe)
    monkeypatch.setattr(target_step, "point_in_time", observe_publication)
    step = object.__new__(StepCubeTarget)
    step._context = context
    step._config = context.config
    step._log = logging.getLogger(__name__)

    df_macro = step._load_macro(calendar, None)

    assert len(reads) == len(payloads) == len(prepared) == 1
    assert reads[0]["since"] is None and reads[0]["until"] == context.run.as_of
    assert "4 invalid/missing" in caplog.text
    pd.testing.assert_series_equal(prepared[0]["date"], prepared[0]["_source_date"], check_names=False)
    assert date(2026, 7, 10) not in prepared[0]["date"].tolist()
    pd.testing.assert_index_equal(df_macro.index, calendar[:3])
    assert df_macro[MACRO_MARKET_SERIES].tolist() == [90.0, 130.0, 140.0]
    assert df_macro["vix"].tolist() == [9.0, 13.0, 14.0]
    expected_returns = pd.Series([float("nan"), 130.0 / 90.0 - 1, 140.0 / 130.0 - 1, 0.0, 0.0, 0.0], index=calendar)
    pd.testing.assert_series_equal(step._market_return(df_macro, calendar), expected_returns, check_names=False)
    commodity, currency = step._asset_factors(df_macro, calendar)
    pd.testing.assert_series_equal(commodity["oil"], expected_returns, check_names=False)
    pd.testing.assert_series_equal(currency["USD/EUR"], expected_returns, check_names=False)
    assert step._macro_changes(df_macro, calendar)["d_vix"].iloc[1:].tolist() == [4.0, 1.0, 0.0, 0.0, 0.0]
    print(
        "SANITY: one scoped payload and one PIT retain raw clocks, exclude invalid/future eligibility, and choose latest holiday/prefix quotes for every factor family."
    )


@pytest.mark.parametrize(("lag_days", "expected"), [(0, "2026-11-06"), (1, "2026-11-09"), (3, "2026-11-09")])
def test_target_financial_history_remains_a_pit_positive_control(sqlite_store: DataStore, lag_days: int, expected: str) -> None:
    context = _context(sqlite_store, "2026-11-13")
    context.config.data.tables[Tables.sharadar_fundamentals_history.name].pit.lag_days = lag_days
    context.store.save(
        Tables.sharadar_fundamentals_history,
        pd.DataFrame(
            {"ticker": ["AAPL"], "as_of": [pd.Timestamp("2026-11-06")], "netIncome": [7.0], "freeCashflow": [8.0], "stockholdersEquity": [9.0]}
        ),
    )
    calendar = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"])
    step = object.__new__(StepCubeTarget)
    step._context = context
    step._config = context.config

    df_actual = step._load_fundamentals(calendar, Window(calendar[0], calendar[-1]))

    assert df_actual is not None
    assert df_actual["as_of"].tolist() == [pd.Timestamp(expected)]
    assert df_actual["_source_as_of"].tolist() == [date(2026, 11, 6)]
    assert df_actual[["netIncome", "freeCashflow", "stockholdersEquity"]].iloc[0].tolist() == [7.0, 8.0, 9.0]
    print(f"SANITY: target financial history still applies lag{lag_days} once ({expected}) and retains original source date and operands.")
