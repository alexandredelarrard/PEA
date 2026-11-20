"""Known-truth SQL DATE/calendar boundary for the target macro loader."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

from src.constants.constants_price import DAILY_MACRO_LEVELS, MACRO_CUBE_FACTORS, MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.transformers import step_cube_target as target_step
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_aggregate.utils.price.frames import load_trading_calendar
from src.data_aggregate.utils.target.betas import estimate_all_betas
from src.data_aggregate.utils.target.factors import assemble_factor_panel
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.db import get_engine
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
    context.config.data.tables[Tables.prices_macro.name].pit.lag_days = 0
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
    ("lag_days", "eligible_dates", "november_10_change"),
    [
        (0, ["2026-11-06", "2026-11-09", "2026-11-10"], 200.0),
        (1, ["2026-11-09", "2026-11-10", "2026-11-11"], 100.0),
        (2, ["2026-11-10", "2026-11-11", "2026-11-12"], float("nan")),
        (3, ["2026-11-11", "2026-11-12", "2026-11-13"], float("nan")),
    ],
    ids=["lag-0", "lag-1", "lag-2", "lag-3"],
)
def test_macro_loader_consumes_publication_lag(sqlite_store: DataStore, lag_days: int, eligible_dates: list[str], november_10_change: float) -> None:
    """Only SPY/gold/oil bypass the configured lag in the shared beta/target inputs."""
    context = _context(sqlite_store, "2026-11-13")
    context.config.data.tables[Tables.prices_macro.name].pit.lag_days = lag_days
    original_config = OmegaConf.to_container(context.config, resolve=True)
    series = sorted({MACRO_MARKET_SERIES, *MACRO_CUBE_FACTORS.values(), *DAILY_MACRO_LEVELS})
    source_dates = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10"], name="date")
    context.store.create_declared(Tables.prices_macro)
    context.store.save(
        Tables.prices_macro,
        pd.concat([pd.DataFrame({"date": source_dates, "ticker": ticker, "close": [100.0, 200.0, 400.0]}) for ticker in series], ignore_index=True),
    )
    calendar = pd.DatetimeIndex(pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]), name="date")
    source_window = Window(calendar[0], calendar[-1])
    step = object.__new__(StepCubeTarget)
    step._context = context
    step._config = context.config

    df_macro = step._load_macro(calendar, source_window)
    for ticker in series:
        expected_dates = source_dates if ticker in (MACRO_MARKET_SERIES, "gold", "oil") else pd.DatetimeIndex(eligible_dates, name="date")
        actual = df_macro[ticker].dropna()
        pd.testing.assert_index_equal(actual.index, expected_dates)
        assert actual.tolist() == [100.0, 200.0, 400.0]
    assert step._market_return(df_macro, calendar).loc["2026-11-10"] == 1.0
    commodity, _ = step._asset_factors(df_macro, calendar)
    assert commodity["gold"].loc["2026-11-10"] == 1.0
    assert commodity["oil"].loc["2026-11-10"] == 1.0
    actual_change = step._macro_changes(df_macro, calendar).loc["2026-11-10", "d_vix"]
    assert pd.isna(actual_change) if pd.isna(november_10_change) else actual_change == november_10_change
    assert OmegaConf.to_container(context.config, resolve=True) == original_config
    print(
        f"SANITY: lag={lag_days}; SPY/gold/oil stay on source sessions; all other consumed series follow config; fills and caller config preserved."
    )


def test_macro_session_lag_preserves_changes_across_weekend_and_holiday(sqlite_store: DataStore) -> None:
    """Thanksgiving and the weekend consume no lag sessions or daily changes."""
    context = _context(sqlite_store, "2026-12-02")
    calendar = pd.DatetimeIndex(["2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30", "2026-12-01", "2026-12-02"], name="date")
    context.store.create_declared(Tables.prices_macro)
    context.store.save(
        Tables.prices_macro,
        pd.DataFrame({"date": calendar[:4], "ticker": "vix", "close": [10.0, 11.0, 13.0, 16.0]}),
    )
    step = object.__new__(StepCubeTarget)
    step._context, step._config = context, context.config

    actual = step._load_macro(calendar, Window(calendar[0], calendar[-1]))

    pd.testing.assert_index_equal(actual.index, calendar[2:])
    assert actual["vix"].tolist() == [10.0, 11.0, 13.0, 16.0]
    assert step._macro_changes(actual, calendar)["d_vix"].dropna().tolist() == [1.0, 2.0, 3.0]
    print("SANITY: two retained sessions skip Thanksgiving/weekend; every daily change survives exactly, with no collision or artificial zero.")


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

    assert len(reads) == len(payloads) == 1
    assert reads[0]["since"] is None and reads[0]["until"] == context.run.as_of
    assert "invalid/missing" in caplog.text
    df_prepared = pd.concat(prepared, ignore_index=True)
    pd.testing.assert_series_equal(df_prepared["date"], df_prepared["_source_date"], check_names=False)
    assert not df_prepared.duplicated(["date", "ticker"]).any()
    assert set(df_prepared.loc[df_prepared["date"] == date(2026, 7, 10), "ticker"]) == {MACRO_MARKET_SERIES, "oil", "vix", "fx_usdeur"}
    pd.testing.assert_index_equal(df_macro.index, calendar[[0, 1, 2, 5]])
    assert df_macro[MACRO_MARKET_SERIES].dropna().tolist() == [100.0, 140.0, 150.0]
    assert df_macro["vix"].dropna().tolist() == [10.0, 14.0]
    assert pd.isna(df_macro.loc["2026-07-10", "vix"])
    expected_returns = pd.Series([float("nan"), 140.0 / 100.0 - 1, 0.0, 0.0, 0.0, 150.0 / 140.0 - 1], index=calendar)
    pd.testing.assert_series_equal(step._market_return(df_macro, calendar), expected_returns, check_names=False)
    commodity, currency = step._asset_factors(df_macro, calendar)
    pd.testing.assert_series_equal(commodity["oil"], expected_returns, check_names=False)
    expected_fx = pd.Series([float("nan"), float("nan"), 140.0 / 100.0 - 1, 0.0, 0.0, 0.0], index=calendar)
    pd.testing.assert_series_equal(currency["USD/EUR"], expected_fx, check_names=False)
    assert step._macro_changes(df_macro, calendar)["d_vix"].iloc[2:].tolist() == [4.0, 0.0, 0.0, 0.0]
    print(
        "SANITY: one scoped payload, PIT once per row, raw clocks retained, invalid/future eligibility excluded, latest collisions and mixed fills correct."
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


@pytest.mark.parametrize(("ticker", "eligible"), [("gold", "2026-11-17"), ("vix", "2026-11-19")])
def test_macro_loader_handles_one_publication_group(sqlite_store: DataStore, ticker: str, eligible: str) -> None:
    context = _context(sqlite_store, "2026-11-19")
    context.config.data.tables[Tables.prices_macro.name].pit.lag_days = 2
    context.store.create_declared(Tables.prices_macro)
    context.store.save(Tables.prices_macro, pd.DataFrame({"date": [pd.Timestamp("2026-11-17")], "ticker": [ticker], "close": [100.0]}))
    calendar = pd.bdate_range("2026-11-16", "2026-11-19", name="date")
    step = object.__new__(StepCubeTarget)
    step._context, step._config = context, context.config
    actual = step._load_macro(calendar, Window(calendar[0], calendar[-1]))
    assert actual.index.tolist() == [pd.Timestamp(eligible)]
    assert actual[ticker].tolist() == [100.0]
    print(f"SANITY: {ticker} alone publishes on {eligible}; an empty other group is harmless.")


def test_macro_publication_default_is_two_days() -> None:
    assert fixture_config().data.tables[Tables.prices_macro.name].pit.lag_days == 2
    assert fixture_config().data_aggregate.betas.ffill_limit == 21
    print("SANITY: production macro lag is 2; estimated-beta forward-fill stays 21 rows.")


@pytest.mark.parametrize(("since", "until"), [("1995-11-13", "2026-11-19"), ("2012-12-01", "2013-01-31"), ("2026-10-01", "2026-11-19")])
def test_macro_loader_real_sources_follow_selective_lag(sqlite_store: DataStore, since: str, until: str) -> None:
    """Replay a bounded read-only real snapshot through candidate and zero-lag controls."""
    # Load the projected live payload once; all replay writes use disposable SQLite.
    load_dotenv(find_dotenv(usecwd=True))
    source = DataStore(get_engine().execution_options(postgresql_readonly=True))
    calendar = load_trading_calendar(_context(source, until))
    calendar = calendar[calendar >= pd.Timestamp(since)]
    series = sorted({MACRO_MARKET_SERIES, *MACRO_CUBE_FACTORS.values(), *DAILY_MACRO_LEVELS})
    df_source = source.load(Tables.prices_macro, columns=["date", "ticker", "close"], where={"ticker": series}, since=since, until=until)
    assert df_source is not None and set(df_source["ticker"]) == set(series), "required real-source witnesses are missing"
    frozen = df_source.copy(deep=True)
    assert not calendar.empty, "required production trading calendar is missing"
    context = _context(sqlite_store, until)
    context.store.create_declared(Tables.prices_macro)
    context.store.save(Tables.prices_macro, df_source)
    step = object.__new__(StepCubeTarget)
    step._context, step._config = context, context.config
    step._log = logging.getLogger(__name__)
    df_actual = step._load_macro(calendar, Window(pd.Timestamp(since), pd.Timestamp(until)))

    # Independent date mapping checks every retained value, collision and unavailable tail.
    for ticker in series:
        df_quotes = df_source.loc[df_source["ticker"] == ticker].sort_values("date")
        lag = 0 if ticker in (MACRO_MARKET_SERIES, "gold", "oil") else 2
        positions = calendar.searchsorted(pd.to_datetime(df_quotes["date"])) + lag
        valid = positions < len(calendar)
        expected = pd.Series(df_quotes.loc[valid, "close"].to_numpy(), index=calendar.take(positions[valid])).groupby(level=0).last().dropna()
        pd.testing.assert_series_equal(df_actual[ticker].dropna(), expected, check_names=False, check_exact=True)

    # Exact zero-lag factor controls include missing sessions and existing forward-fill.
    step._config = context.config.copy()
    step._config.data.tables[Tables.prices_macro.name].pit.lag_days = 0
    df_control = step._load_macro(calendar, Window(pd.Timestamp(since), pd.Timestamp(until)))
    pd.testing.assert_series_equal(step._market_return(df_actual, calendar), step._market_return(df_control, calendar), check_exact=True)
    actual_commodities, _ = step._asset_factors(df_actual, calendar)
    control_commodities, _ = step._asset_factors(df_control, calendar)
    pd.testing.assert_frame_equal(actual_commodities, control_commodities, check_exact=True)
    actual_changes = step._macro_changes(df_actual, calendar)
    control_changes = step._macro_changes(df_control, calendar)
    pd.testing.assert_frame_equal(actual_changes, control_changes.shift(2), check_exact=True)
    if since == "1995-11-13":
        _, currency = step._asset_factors(df_actual, calendar)
        market = step._market_return(df_actual, calendar)
        panel, macro_cols = assemble_factor_panel(market, pd.DataFrame(index=calendar), actual_commodities, currency, actual_changes)
        assert set(macro_cols) == set(DAILY_MACRO_LEVELS.values())
        betas = estimate_all_betas(market.to_frame("SPY"), panel, window=63, min_obs=40, step=21)
        expected_betas = [f"beta_{name}" for name in macro_cols]
        assert betas["SPY"][expected_betas].notna().all(axis=1).any()
        for name in macro_cols:
            zero_frac = actual_changes[name].dropna().abs().lt(1e-12).mean()
            assert zero_frac <= 0.30
            print(f"SANITY: {name}: {zero_frac:.2%} unchanged sessions; retained in final panel and fitted betas.")
    pd.testing.assert_frame_equal(df_source, frozen, check_exact=True)
    assert context.config.data.tables[Tables.prices_macro.name].pit.lag_days == 2
    print(
        f"SANITY: {since}..{until}: {len(df_source)} real observations/{len(calendar)} production sessions/{len(series)} series; exact selective dates/values and zero-lag market/gold/oil factors; source unchanged."
    )
