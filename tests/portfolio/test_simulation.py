"""Known-truth accounting, with no live services or fitted financial claims."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.portfolio.transformers.simulation import simulate
from src.portfolio.utils.contracts import MarketInputs, Settings


def settings(**changes: object) -> Settings:
    base = Settings(pd.Timestamp("2022-01-03"), None, (0.5, 0.25, 0.25), "weekly", 0.033, 2, 2, 0, 0, 0)
    return replace(base, **changes)


def market() -> MarketInputs:
    dates = pd.bdate_range("2021-12-20", "2022-02-15")
    quotes = pd.DataFrame(100.0, index=dates, columns=["A", "B", "C", "D"])
    returns = pd.DataFrame(np.outer(np.where(np.arange(len(dates)) % 2, 1, -1), [0.01, -0.01, 0.02, -0.02]), index=dates, columns=quotes.columns)
    scores = pd.DataFrame(np.tile([4, 1, 3, 2], (len(dates), 1)), index=dates, columns=quotes.columns)
    return MarketInputs(
        scores,
        quotes,
        returns,
        quotes * 0 + 1,
        pd.Series(100.0, index=dates),
        pd.Series({"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}),
        {},
    )


def test_cash_uses_calendar_interest() -> None:
    data = market()
    result = simulate(data, settings(allocation=(0, 0, 1)), 10, "weekly")
    days = (result.daily.index[-1] - result.daily.index[0]).days
    assert result.status == "complete"
    assert result.daily.iloc[0].nav == 100
    assert result.daily.iloc[-1].nav == pytest.approx(100 * 1.033 ** (days / 365))
    assert result.orders.empty
    print("SANITY: only free cash compounds at 3.3%, including weekends; no LS capacity required.")


def test_signal_waits_until_next_close_and_no_fake_trades() -> None:
    data = market()
    data.quotes.loc[:, :] = 100
    data.returns.loc[:, :] = 0
    data.market.loc["2022-01-04":] = 200
    data.market.loc["2022-01-05":] = 220
    result = simulate(data, settings(allocation=(0, 1, 0), allocation_frequency="monthly"), 10, "monthly")
    assert result.daily.loc["2022-01-03", "nav"] == 100
    assert result.daily.loc["2022-01-04", "nav"] == 100
    assert result.daily.loc["2022-01-05", "nav"] == pytest.approx(110)
    assert len(result.orders) == 1
    print("SANITY: J->J+1 jump belongs to previous holdings; later drift produces no fictitious order.")


@pytest.mark.parametrize("levels", [[100, 200, 220], [100, 100, 110]])
def test_missing_stock_session_does_not_delay_sp_execution(levels: list[int]) -> None:
    data = market()
    sessions = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
    data.market.loc[sessions] = levels
    data = replace(
        data,
        quotes=data.quotes.drop(sessions[1]),
        returns=data.returns.drop(sessions[1]),
        share_basis=data.share_basis.drop(sessions[1]),
    )
    result = simulate(data, settings(allocation=(0, 1, 0), allocation_frequency="monthly", end=sessions[-1]), 10, "monthly")
    assert result.status == "complete", result.reason
    assert result.daily.index.equals(sessions)
    assert result.orders.iloc[0].date == sessions[1]
    assert result.daily.nav.to_list() == pytest.approx([100, 100, 110])
    print("SANITY: a session witnessed by scores retains J+1 execution even when every stock price row is absent and the market is flat.")


@pytest.mark.parametrize("missing, field", [("2022-01-04", "execution quote"), ("2022-01-05", "held quote")])
def test_missing_entire_stock_session_interrupts_selected_or_held_book(missing: str, field: str) -> None:
    data = market()
    date = pd.Timestamp(missing)
    data = replace(data, quotes=data.quotes.drop(date), returns=data.returns.drop(date), share_basis=data.share_basis.drop(date))
    result = simulate(data, settings(end=pd.Timestamp("2022-01-06")), 2, "monthly")
    assert result.status == "incomplete"
    assert f"{missing}: missing/invalid {field}" in str(result.reason)
    assert all(ticker in str(result.reason) for ticker in data.quotes.columns)
    assert result.daily.index.max() == date - pd.Timedelta(days=1)
    assert result.orders.empty or result.orders.date.max() < date
    print(f"SANITY: the missing {missing} stock session interrupts the {field} with named cells instead of delaying or filling it.")


def test_macro_only_weekend_and_holiday_carries_are_not_sessions() -> None:
    data = market()
    sessions = pd.to_datetime(["2022-01-14", "2022-01-18", "2022-01-19"])
    data = replace(
        data,
        scores=data.scores.loc[sessions],
        quotes=data.quotes.loc[sessions[[0, 2]]],
        returns=data.returns.loc[sessions[[0, 2]]],
        share_basis=data.share_basis.loc[sessions[[0, 2]]],
        market=pd.Series([100, 100, 100, 150, 200, 220], index=pd.date_range(sessions[0], sessions[-1])),
    )
    result = simulate(data, settings(start=sessions[0], end=sessions[-1], allocation=(0, 1, 0)), 10, "monthly")
    assert result.status == "complete", result.reason
    assert result.daily.index.equals(sessions)
    assert result.orders.iloc[0].signal_date == sessions[0]
    assert result.orders.iloc[0].date == sessions[1]
    assert result.daily.nav.to_list() == pytest.approx([100, 100, 110])
    print("SANITY: macro-only weekend/holiday carries add no sessions; the first witnessed session after the closure remains J+1.")


def test_fee_financed_targets_and_both_sides() -> None:
    data = market()
    data.returns.loc[:, :] = 0
    result = simulate(data, settings(allocation=(0, 1, 0), spread_bps=100, allocation_frequency="monthly"), 1, "monthly")
    assert result.daily.iloc[1].nav == pytest.approx(100 / 1.01)
    assert result.daily.iloc[1].cash >= -1e-9
    assert result.daily.iloc[1].sp500 == pytest.approx(result.daily.iloc[1].nav)
    assert result.orders.iloc[0].fee == pytest.approx(100 - 100 / 1.01)
    print("SANITY: entry costs reduce investible capital; no debt or rebasing away setup fees.")


def test_ls_neutrality_and_no_daily_weight_reset() -> None:
    data = market()
    result = simulate(data, settings(cash_rate=0), 2, "monthly")
    assert result.status == "complete", result.reason
    entry = result.positions.loc[result.positions.date == pd.Timestamp("2022-01-04")]
    assert entry.groupby("sector").value.sum().abs().max() < 1e-8
    assert entry.value.abs().sum() == pytest.approx(50)
    assert len(result.orders.loc[result.orders.date == pd.Timestamp("2022-01-05")]) == 0
    print("SANITY: 50 LS capital funds 25 longs/25 shorts; quantities persist between schedules.")


def test_missing_execution_price_interrupts_without_reselection() -> None:
    data = market()
    data.quotes.loc["2022-01-04", "A"] = np.nan
    result = simulate(data, settings(), 2, "daily")
    assert result.status == "incomplete" and "A" in str(result.reason)
    assert result.daily.index.max() == pd.Timestamp("2022-01-03")
    print("SANITY: missing J+1 execution quote interrupts the selected book, without hindsight replacement.")


@pytest.mark.parametrize("ticker", ["A", "B"])
def test_split_and_dividend_preserve_nav_without_fake_trades(ticker: str) -> None:
    data = market()
    data.returns.loc["2022-01-04":] = 0
    data.share_basis.loc[:"2022-01-07", ticker] = 2
    data.quotes.loc["2022-01-10":, ticker] = 50
    result = simulate(data, settings(cash_rate=0, allocation_frequency="monthly", end=pd.Timestamp("2022-01-14")), 2, "monthly")
    assert result.status == "complete", result.reason
    assert np.allclose(result.daily.nav, 100)
    before = result.positions.loc[(result.positions.ticker == ticker) & (result.positions.date == pd.Timestamp("2022-01-07")), "quantity"].iloc[0]
    after = result.positions.loc[(result.positions.ticker == ticker) & (result.positions.date == pd.Timestamp("2022-01-10")), "quantity"].iloc[0]
    assert after == pytest.approx(2 * before)
    assert len(result.orders) == 5  # four stocks plus initial S&P establishment
    data.quotes.loc["2022-01-11":, ticker] = 49
    dividend = simulate(data, settings(cash_rate=0, allocation_frequency="monthly", end=pd.Timestamp("2022-01-14")), 2, "monthly")
    assert dividend.status == "complete", dividend.reason
    assert np.allclose(dividend.daily.nav, 100)
    assert dividend.daily.loc["2022-01-11", "ls_reserved"] - dividend.daily.loc["2022-01-10", "ls_reserved"] == pytest.approx(after)
    print(f"SANITY: {ticker} split doubles signed quantity without a trade; implicit distribution credits longs/debits shorts exactly once.")


def test_outer_allocation_uses_current_capital_and_actual_fees() -> None:
    data = market()
    data.returns.loc["2022-01-04":] = 0
    data.market.loc["2022-01-10":] = 140
    result = simulate(data, settings(cash_rate=0, spread_bps=5, end=pd.Timestamp("2022-01-28")), 2, "monthly")
    assert result.status == "complete", result.reason
    row = result.daily.loc["2022-01-10"]
    assert row.long_short == pytest.approx(0.5 * row.nav)
    assert row.sp500 == pytest.approx(0.25 * row.nav)
    assert row.cash == pytest.approx(0.25 * row.nav)
    assert row.fee > 0
    assert set(result.orders.loc[result.orders.date == row.name, "cause"]) == {"allocation"}
    print("SANITY: weekly 50/25/25 applies to current after-fee NAV, scaling the monthly-selected LS book and charging actual deltas.")
