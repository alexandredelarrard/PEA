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

_YOY = 252  # ~1 trading year
_FIVE_Y = 5 * 252  # ~5 trading years
_QUARTERLY_MAX_AGE_DAYS = 185
DIVIDEND_TRANSFORM_SEMANTICS = {"dividend_payer": "binary"}


def _cagr(now: pd.DataFrame, span: int, years: float) -> pd.DataFrame:
    """Annualized growth of a TTM series over `span` trading days: (now/then)^(1/years)
    - 1, defined only where the base `then` is a positive dividend."""
    then = now.shift(span)
    return (((now / then.where(then > 0)) ** (1.0 / years)) - 1.0).replace([np.inf, -np.inf], np.nan)


def _ttm_dividends(dividends_hist: pd.DataFrame, idx: pd.DatetimeIndex, universe: list[str]) -> pd.DataFrame:
    """Daily trailing-12m cash dividend per share (date x ticker), 0 for names
    that never paid, aligned to the trading calendar. Point-in-time: the rolling
    sum at t only includes ex-dates <= t."""
    piv = dividends_hist.pivot_table(index="date", columns="ticker", values="dividends", aggfunc="sum")
    piv.index = pd.to_datetime(piv.index).normalize()
    # full universe so non-payers are a real 0, aligned to trading days
    piv = piv.reindex(index=idx, columns=universe).fillna(0.0)
    return piv.rolling(_YOY, min_periods=1).sum()


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

    g1 = sanitize(ttm_ps / ttm_ps.shift(_YOY).where(lambda x: x > 0)) - 1.0
    g5 = _cagr(ttm_ps, _FIVE_Y, 5.0)
    if not div_paid.empty:
        g1 = g1.combine_first(sanitize(div_paid / div_paid.shift(_YOY).where(lambda x: x > 0)) - 1.0)
        g5 = g5.combine_first(_cagr(div_paid, _FIVE_Y, 5.0))
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
