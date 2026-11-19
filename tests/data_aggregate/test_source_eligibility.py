"""Known-truth source availability controls at actual consumer boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.utils.common.inputs import input_windows
from src.data_aggregate.utils.fundamentals.dividend_features import _ttm_dividends
from src.data_aggregate.utils.fundamentals.earnings_features import _report_state_to_daily, reported_events
from src.data_aggregate.utils.sec_thirteen.value_basis import _row_consensus, consensus_prices
from src.data_store.schema import Tables
from src.utils.publication import point_in_time
from src.utils.window import Window, WindowRun
from tests.support.config import fixture_config
from tests.support.universe import universe_of


def test_financial_history_rolls_buffer_to_actual_session(sqlite_store: Any) -> None:
    config = fixture_config()
    sqlite_store.save(
        Tables.sharadar_fundamentals_history,
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "as_of": pd.to_datetime(["2026-11-06"]),
                "fiscal_end": pd.to_datetime(["2026-09-30"]),
                "netIncome": [10.0],
            }
        ),
    )
    context = SimpleNamespace(
        store=sqlite_store,
        config=config,
        config_dir=Path("configs"),
        universe=universe_of(["AAA"]),
        run=WindowRun("full", "financial filing clock", pd.Timestamp("2026-11-16")),
    )
    step = StepCubeFundamentals(context, config)
    sessions = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-16"])
    windows = input_windows(context, Tables.cube_part_fundamentals, Window(sessions[0], sessions[-1]))
    rows = step._load_fundamentals(("AAA",), sessions, windows)
    assert pd.to_datetime(rows.as_of).tolist() == [pd.Timestamp("2026-11-09")]
    assert pd.to_datetime(rows._source_as_of).tolist() == [pd.Timestamp("2026-11-06")]
    print("SANITY: financial rows retain raw filing clocks and become eligible after the buffer on a real session.")


def test_earnings_state_uses_prepared_eligibility_without_relagging() -> None:
    sessions = pd.DatetimeIndex(["2026-11-05", "2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"])
    hist = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "earnings_date": pd.to_datetime(["2026-11-05"]),
            "as_of": pd.to_datetime(["2026-11-10"]),
            "eps_actual": [2.0],
            "eps_estimate": [1.5],
            "surprise_pct": [33.3],
        }
    )
    reports = reported_events(hist, sessions)
    assert reports.as_of.tolist() == [pd.Timestamp("2026-11-10")]
    state = _report_state_to_daily(reports, ["eps_actual"], sessions)["eps_actual"]["AAA"]
    assert state.loc[:"2026-11-09"].isna().all() and state.loc["2026-11-10"] == 2.0
    print("SANITY: earnings state uses its configured eligibility exactly once and keeps the actual publication date for age.")


def test_mixed_financial_roots_keep_raw_grain_and_wait_for_both_operands(sqlite_store: Any) -> None:
    config = fixture_config()
    config.data.tables.sharadar_fundamentals_history.pit.lag_days = 1
    config.data.tables.sec_fundamentals_history.pit.lag_days = 4
    source_day = pd.Timestamp("2026-11-06")
    sqlite_store.save(
        Tables.sharadar_fundamentals_history,
        pd.DataFrame({"ticker": ["AAA"], "as_of": [source_day], "fiscal_end": pd.to_datetime(["2026-09-30"]), "netIncome": [10.0]}),
    )
    sqlite_store.save(
        Tables.sec_fundamentals_history,
        pd.DataFrame(
            {"ticker": ["AAA"], "as_of": [source_day], "minorityInterest": [7.0], "operatingLeaseLiability": [2.0], "financeLeaseLiability": [1.0]}
        ),
    )
    sessions = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10"])
    context = SimpleNamespace(
        store=sqlite_store,
        config=config,
        config_dir=Path("configs"),
        universe=universe_of(["AAA"]),
        run=WindowRun("full", "different independently prepared source clocks", sessions[-1]),
    )
    windows = input_windows(context, Tables.cube_part_fundamentals, Window(sessions[0], sessions[-1]))
    rows = StepCubeFundamentals(context, config)._load_fundamentals(("AAA",), sessions, windows)
    assert len(rows) == 1
    assert pd.Timestamp(rows._source_as_of.iloc[0]) == pd.Timestamp(rows._sec_source_as_of.iloc[0]) == source_day
    assert rows._sharadar_as_of.iloc[0] == pd.Timestamp("2026-11-09")
    assert rows._sec_as_of.iloc[0] == rows.as_of.iloc[0] == pd.Timestamp("2026-11-10")
    assert rows.sec_minorityInterest.iloc[0] == 7.0 and rows.netIncome.iloc[0] == 10.0
    print(
        "SANITY: independently prepared roots join on their exact original key; the carried SEC value appears on the later operand clock, with both raw clocks retained."
    )


def test_dividend_eligibility_preserves_original_ex_date_expiry() -> None:
    sessions = pd.bdate_range("2025-01-02", "2026-02-01")
    hist = pd.DataFrame(
        {"ticker": ["AAA"], "date": pd.to_datetime(["2025-01-02"]), "available_on": pd.to_datetime(["2025-01-06"]), "dividends": [1.0]}
    )
    daily = _ttm_dividends(hist, sessions, ["AAA"], "365D")["AAA"]
    assert daily.loc[:"2025-01-03"].eq(0.0).all() and daily.loc["2025-01-06"] == 1.0
    assert daily.loc["2026-01-02"] == 0.0
    assert hist.date.iloc[0] == pd.Timestamp("2025-01-02")
    print("SANITY: dividend cash starts at eligibility and expires from its original ex date; the buffer never shifts the economic event.")


def test_insider_publication_never_falls_back_to_a_non_session() -> None:
    sessions = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10"])
    dates = pd.Series(pd.to_datetime(["2026-11-06", "2026-11-10"]))
    ready = point_in_time(pd.DataFrame({"filing_date": dates}), fixture_config(), Tables.insider_transactions, sessions)
    assert ready.as_of.tolist() == [pd.Timestamp("2026-11-09")]
    assert ready._source_filing_date.tolist() == [pd.Timestamp("2026-11-06")]
    print("SANITY: insider eligibility shares the market-session resolver; an unavailable future session stays unknown.")


def test_13f_repair_consensus_preserves_existing_pooled_evidence() -> None:
    donors = pd.DataFrame(
        {
            "period": pd.to_datetime(["2026-03-31"] * 5),
            "cusip": ["111111111"] * 5,
            "cik": ["1", "2", "3", "4", "5"],
            "filing_date": pd.to_datetime(["2026-05-05", "2026-05-06", "2026-05-20", "2026-05-20", "2026-05-20"]),
            "position_type": ["common"] * 5,
            "shares": [100.0] * 5,
            "value_usd": [1000.0, 1000.0, 100000.0, 100000.0, 100000.0],
        }
    )
    quotes = consensus_prices(donors, 2)
    receiver = pd.DataFrame(
        {
            "period": pd.to_datetime(["2026-03-31"] * 3),
            "cusip": ["111111111"] * 3,
            "filing_date": pd.to_datetime(["2026-05-05", "2026-05-06", "2026-05-21"]),
        }
    )
    prices = _row_consensus(receiver, quotes)
    assert prices.eq(1000.0).all()
    print("SANITY: 13F unit/value repair retains the existing pooled donor calculation, independent of the filing-based publication buffer.")


def test_dividend_buffer_does_not_revive_events_outside_the_calculation_grid() -> None:
    sessions = pd.bdate_range("2026-01-05", "2026-01-09")
    rows = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2025-01-02", "2026-01-05"]),
            "available_on": pd.to_datetime(["2025-01-03", "2026-01-06"]),
            "dividends": [100.0, 1.0],
        }
    )
    result = _ttm_dividends(rows, sessions, ["AAA"])["AAA"]
    assert result.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    print(
        "SANITY: delaying cash eligibility retains original rolling-grid membership; an old pre-grid dividend cannot become a new252-session payment."
    )
