# ruff: noqa: N806
"""Public dividend and shareholder yields with separate matched price/count bases.

Adjusted dividend/quote pairs remain matched; cash fallback uses saved issuer cap and buybacks use genuine comparison units."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.frames import sanitize
from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.fundamentals.feature_views import build_fundamental_views
from src.data_aggregate.utils.fundamentals.inputs import FundamentalHistory
from src.data_aggregate.utils.price.frames import PriceFrames, comparison_share_history

_YOY = 252  # ~1 trading year: the level TTM and the buyback comparison
_CALENDAR_YEAR_DAYS = 365  # the growth horizon, in calendar days
_QUARTERLY_MAX_AGE_DAYS = 185
DIVIDEND_TRANSFORM_SEMANTICS = {"dividend_payer": "binary"}


def _calendar_prior(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    """Each day's value on the last session at or before `days` calendar days earlier.

    NaN while that date precedes the first session (first-year edge: growth needs a full
    365 days of history)."""
    index = frame.index
    pos = index.searchsorted(index - pd.Timedelta(days=days), side="right") - 1
    prior = frame.iloc[np.maximum(pos, 0)].set_axis(index, axis=0)
    return prior.where(pd.Series(pos >= 0, index=index), axis=0)


def _cagr(now: pd.DataFrame, years: int) -> pd.DataFrame:
    """Annualized growth of a calendar-TTM series over `years` x 365 calendar days: (now/then)^(1/years)
    - 1, defined only where the base `then` is a positive dividend."""
    then = _calendar_prior(now, years * _CALENDAR_YEAR_DAYS)
    return (((now / then.where(then > 0)) ** (1.0 / years)) - 1.0).replace([np.inf, -np.inf], np.nan)


def _ttm_dividends(dividends_hist: pd.DataFrame, idx: pd.DatetimeIndex, universe: list[str], window: int | str = _YOY) -> pd.DataFrame:
    """Daily trailing-12m cash dividend per share (date x ticker), 0 for names
    that never paid, aligned to the trading calendar. Point-in-time: the rolling
    sum at t only includes ex-dates <= t. `window` is 252 sessions (levels) or "365D"
    (growth: a payment exactly 365 days old has dropped out)."""
    # Begin cash at eligibility, but expire it from the original ex date.
    dates = pd.to_datetime(dividends_hist["date"]).dt.normalize()
    available = pd.to_datetime(dividends_hist["available_on"]).dt.normalize()
    starts = idx.searchsorted(available, side="left")
    ends = idx.searchsorted(dates + pd.Timedelta(window), side="left") if isinstance(window, str) else idx.searchsorted(dates) + window
    valid = dates.notna() & dates.isin(idx) & available.notna() & (starts < len(idx)) & (starts < ends)
    df_active = dividends_hist.loc[valid, ["ticker", "dividends"]].copy()
    begins = df_active.assign(_position=starts[valid], _cash=df_active["dividends"])
    ends = ends[valid]
    stops = df_active.loc[ends < len(idx)].assign(_position=ends[ends < len(idx)], _cash=-df_active.loc[ends < len(idx), "dividends"])
    events = pd.concat([begins, stops])
    daily = events.pivot_table(index="_position", columns="ticker", values="_cash", aggfunc="sum")
    return daily.reindex(index=range(len(idx)), columns=universe, fill_value=0.0).fillna(0.0).cumsum().set_axis(idx)


def _dividend_fields(dividends_hist: pd.DataFrame, prices: PriceFrames, history: FundamentalHistory | None = None) -> dict[str, pd.DataFrame]:
    """Keep raw dividend cash units distinct from normalized share comparisons."""
    fundamentals = None if history is None else history.raw_history
    share_basis = None if history is None else history.share_basis
    comparison_history = None if history is None else history.normalized_share_history
    prices.require("close_split")
    close = prices.close_split
    assert close is not None
    idx = pd.DatetimeIndex(close.index)
    universe = list(close.columns)
    ttm_ps = _ttm_dividends(dividends_hist, idx, universe)  # per-share TTM (source A)
    close_pos = close.where(close > 0)
    listed = close.notna()

    def _fd(field: str) -> pd.DataFrame:
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame()
        return fundamentals_to_daily(
            fundamentals,
            field,
            idx,
            max_age_days=_QUARTERLY_MAX_AGE_DAYS,
        ).reindex(columns=universe)

    div_paid = _fd("dividendsPaid")  # total cash dividends, TTM (source B)
    shares = _fd("sharesOutstanding")
    net_income = _fd("netIncome")
    fcf = _fd("freeCashflow")

    mcap = pd.DataFrame(index=idx) if prices.mcap_pit is None else prices.mcap_pit.reindex_like(close)
    mcap = mcap.where(shares.reindex_like(mcap).gt(0) & np.isfinite(mcap) & mcap.gt(0))
    total_a = (ttm_ps * shares).where(ttm_ps > 0) if not shares.empty else pd.DataFrame()
    if not total_a.empty:
        total = total_a.combine_first(div_paid) if not div_paid.empty else total_a
    else:
        total = div_paid

    F: dict[str, pd.DataFrame] = {}

    yield_a = sanitize(ttm_ps.where(ttm_ps > 0) / close_pos)
    payer = ttm_ps.gt(0)
    if not div_paid.empty:
        payer = payer | div_paid.gt(0)
    fallback = sanitize(total / mcap) if not mcap.empty and not total.empty else pd.DataFrame(index=idx)
    F["dividend_yield"] = yield_a.combine_first(fallback).where(payer, 0.0).where(listed)

    # growth reads calendar windows (365 days); the level features above keep the 252-session TTM
    ttm_growth = _ttm_dividends(dividends_hist, idx, universe, f"{_CALENDAR_YEAR_DAYS}D")
    g1 = sanitize(ttm_growth / _calendar_prior(ttm_growth, _CALENDAR_YEAR_DAYS).where(lambda x: x > 0)) - 1.0
    g5 = _cagr(ttm_growth, 5)
    if not div_paid.empty:
        g1 = g1.combine_first(sanitize(div_paid / _calendar_prior(div_paid, _CALENDAR_YEAR_DAYS).where(lambda x: x > 0)) - 1.0)
        g5 = g5.combine_first(_cagr(div_paid, 5))
    F["dividend_growth"] = g1.replace([np.inf, -np.inf], np.nan)
    F["dividend_growth_5y"] = g5

    F["dividend_payer"] = payer.astype("float64").where(listed)

    if not total.empty and not net_income.empty:
        F["dividend_payout_ratio"] = sanitize(total / net_income.where(net_income > 0)).clip(lower=0.0, upper=3.0)
    if not total.empty and not fcf.empty:
        F["dividend_coverage"] = sanitize(fcf / total.where(total > 0))

    if fundamentals is not None:
        df_comparison = comparison_share_history(
            fundamentals if comparison_history is None else comparison_history, share_basis, pd.Timestamp(idx.max())
        )
        df_comparison_shares = fundamentals_to_daily(df_comparison, "sharesOutstanding", idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS)
    else:
        df_comparison_shares = pd.DataFrame(index=idx)
    if not df_comparison_shares.empty and df_comparison_shares.notna().any().any():
        buyback_yield = -(df_comparison_shares / df_comparison_shares.shift(_YOY) - 1.0)
        F["shareholder_yield"] = F["dividend_yield"].add(buyback_yield, fill_value=0.0).replace([np.inf, -np.inf], np.nan).where(listed)
    return F


def build_dividend_feature_panel(
    dividends_history: pd.DataFrame | None,
    prices: PriceFrames | None,
    history: FundamentalHistory | None = None,
    history_fields: dict[str, pd.DataFrame] | None = None,
    output_since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build dividend views with raw cash and normalized count comparisons."""
    if (
        dividends_history is None
        or dividends_history.empty
        or "dividends" not in dividends_history.columns
        or prices is None
        or prices.close_split is None
    ):
        return pd.DataFrame(columns=["date", "ticker"])

    # Restrict prices to the declared calculation grid before evaluating cash yields.
    prices = replace(
        prices,
        close_split=prices.close_split.reindex(prices.trading_index),
        mcap_pit=None if prices.mcap_pit is None else prices.mcap_pit.reindex(prices.trading_index),
    )
    fields = _dividend_fields(dividends_history, prices, history)
    semantics = {name: mode for name, mode in DIVIDEND_TRANSFORM_SEMANTICS.items() if name in fields}
    return build_fundamental_views(fields, prices.peers, semantics=semantics, history_fields=history_fields, output_since=output_since)
