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
from src.data_aggregate.utils.price.frames import PriceFrames, comparison_share_history
from src.utils.price.adjustments import ShareBasis

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


def _dividend_fields(
    dividends_hist: pd.DataFrame, prices: PriceFrames, fundamentals: pd.DataFrame | None, share_basis: ShareBasis | None = None
) -> dict:
    """Keep adjusted dividend/quote pairs; use saved cap for cash-only fallback yields.

    Raw vendor shares stay with cash reconciliation; count growth uses its own comparison-unit history."""
    prices.require("close_split")
    close = prices.close_split
    assert close is not None
    idx = pd.DatetimeIndex(close.index)
    universe = list(close.columns)
    ttm_ps = _ttm_dividends(dividends_hist, idx, universe)  # per-share TTM (source A)
    close_pos = close.where(close > 0)
    # THE LISTED WINDOW. Three features below fill a real 0 for non-payers, which is right
    # in the cross-section and wrong on the pre-listing pad -- see the module docstring.
    # `close` is the only frame here that knows when the company was quotable.
    listed = close.notna()

    # ---- source B (SEC cash-flow statement): reconciliation + payout / coverage ----
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

    # RECONCILED total cash dividends: per-share x shares where the name actually paid
    # (source A, ttm_ps>0), else the SEC `dividendsPaid` total (source B) fills the gap.
    mcap = pd.DataFrame(index=idx) if prices.mcap_pit is None else prices.mcap_pit.reindex_like(close)
    mcap = mcap.where(shares.reindex_like(mcap).gt(0) & np.isfinite(mcap) & mcap.gt(0))
    total_a = (ttm_ps * shares).where(ttm_ps > 0) if not shares.empty else pd.DataFrame()
    if not total_a.empty:
        total = total_a.combine_first(div_paid) if not div_paid.empty else total_a
    else:
        total = div_paid

    F: dict[str, pd.DataFrame] = {}

    # ---- dividend yield (reconciled): precise per-share/price where paid, source-B
    # fill for names the ex-date history misses, real 0 for true non-payers ----
    yield_a = sanitize(ttm_ps.where(ttm_ps > 0) / close_pos)
    payer = ttm_ps.gt(0)
    if not div_paid.empty:
        payer = payer | div_paid.gt(0)
    fallback = sanitize(total / mcap) if not mcap.empty and not total.empty else pd.DataFrame(index=idx)
    F["dividend_yield"] = yield_a.combine_first(fallback).where(payer, 0.0).where(listed)

    # ---- growth (1y + 5y CAGR), per-share source A primary, source-B total fills ----
    g1 = sanitize(ttm_ps / ttm_ps.shift(_YOY).where(lambda x: x > 0)) - 1.0
    g5 = _cagr(ttm_ps, _FIVE_Y, 5.0)
    if not div_paid.empty:
        g1 = g1.combine_first(sanitize(div_paid / div_paid.shift(_YOY).where(lambda x: x > 0)) - 1.0)
        g5 = g5.combine_first(_cagr(div_paid, _FIVE_Y, 5.0))
    F["dividend_growth"] = g1.replace([np.inf, -np.inf], np.nan)
    F["dividend_growth_5y"] = g5

    # ---- payer flag: paid in the trailing year per EITHER source ----
    # A comparison can never be NaN, so this flag is 0/1 on EVERY cell of the grid unless
    # it is scoped: unmasked it asserted "did not pay a dividend" for companies that had
    # not listed yet, which is not a fact about them.
    F["dividend_payer"] = payer.astype("float64").where(listed)

    # ---- payout ratio + FCF coverage (dividend safety) off the reconciled total ----
    if not total.empty and not net_income.empty:
        F["dividend_payout_ratio"] = sanitize(total / net_income.where(net_income > 0)).clip(lower=0.0, upper=3.0)
    if not total.empty and not fcf.empty:
        # FCF / dividends: >1 => free cash flow covers the payout (safe); <1 or <0 => not.
        F["dividend_coverage"] = sanitize(fcf / total.where(total > 0))

    # ---- shareholder yield = dividend yield + buyback yield (- share issuance) ----
    if fundamentals is not None:
        df_comparison = comparison_share_history(fundamentals, share_basis, pd.Timestamp(idx.max()))
        df_comparison_shares = fundamentals_to_daily(df_comparison, "sharesOutstanding", idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS)
    else:
        df_comparison_shares = pd.DataFrame(index=idx)
    if not df_comparison_shares.empty and df_comparison_shares.notna().any().any():
        buyback_yield = -(df_comparison_shares / df_comparison_shares.shift(_YOY) - 1.0)
        # `fill_value=0.0` treats a missing leg as zero, so this inherits whatever scope
        # `dividend_yield` has; `.where(listed)` is re-applied rather than assumed.
        F["shareholder_yield"] = F["dividend_yield"].add(buyback_yield, fill_value=0.0).replace([np.inf, -np.inf], np.nan).where(listed)
    return F


def build_dividend_feature_panel(
    dividends_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    prices: PriceFrames | None,
    fundamentals_history: pd.DataFrame | None = None,
    history_fields: dict[str, pd.DataFrame] | None = None,
    output_since: pd.Timestamp | None = None,
    share_basis: ShareBasis | None = None,
) -> pd.DataFrame:
    """Long-format dividend panel using the approved raw/self-history view contract.
    Empty if no dividend history is available."""
    if (
        dividends_history is None
        or dividends_history.empty
        or "dividends" not in dividends_history.columns
        or prices is None
        or prices.close_split is None
    ):
        return pd.DataFrame(columns=["date", "ticker"])
    prices = replace(
        prices,
        trading_index=trading_index,
        close_split=prices.close_split.reindex(trading_index),
        mcap_pit=None if prices.mcap_pit is None else prices.mcap_pit.reindex(trading_index),
    )
    fields = _dividend_fields(dividends_history, prices, fundamentals_history, share_basis)
    semantics = {name: mode for name, mode in DIVIDEND_TRANSFORM_SEMANTICS.items() if name in fields}
    return build_fundamental_views(
        fields,
        peer_dict,
        semantics=semantics,
        history_fields=history_fields,
        output_since=output_since,
    )
