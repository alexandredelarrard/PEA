"""
dividend_features.py  (src/data_aggregate/utils/dividend_features.py)
---------------------------------------------------------------------
Peer-relative DIVIDEND / SHAREHOLDER-YIELD features. All point-in-time (a dividend
enters TTM only on/after its ex-date, and growth compares past-vs-past):

    dividend_yield        trailing-12m cash dividends / price   (income / value tilt)
    dividend_growth       TTM dividends vs TTM one year ago      (payout trajectory)
    dividend_growth_5y    5-year CAGR of TTM dividends           (durable dividend grower)
    dividend_payer        1 if the firm paid a dividend in the trailing year
    dividend_payout_ratio cash dividends / net income            (how much earnings is paid out)
    dividend_coverage     free cash flow / cash dividends        (dividend SAFETY; >1 = covered)
    shareholder_yield     dividend_yield + buyback yield
                          (buyback yield = -YoY change in shares outstanding; net
                           issuance/dilution lowers it, buybacks raise it)

RECONCILED across the two dividend sources (they measure the same cash two ways):
  * source A = the per-share EX-DATE history (`dividends` table, `dividends` column,
    from the price download) -> precise, per-share, full history; PRIMARY for
    yield/growth.
  * source B = the SEC cash-flow `dividendsPaid` total (from fundamentals) -> the
    dollar figure that pairs with net income / FCF; powers payout & coverage and
    GAP-FILLS names the ex-date history misses.
The reconciled TTM total = per-share x shares where the name paid (source A), else
`dividendsPaid` (source B); one consistent number feeds every ratio here.

Non-payers get a real 0 dividend yield (not NaN) so they rank correctly in the
cross-section; shareholder_yield still captures their buybacks/dilution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.common.frames import sanitize
from src.data_aggregate.utils.common.panel import build_peer_relative_panel

_YOY = 252       # ~1 trading year
_FIVE_Y = 5 * 252  # ~5 trading years


def _cagr(now: pd.DataFrame, span: int, years: float) -> pd.DataFrame:
    """Annualized growth of a TTM series over `span` trading days: (now/then)^(1/years)
    - 1, defined only where the base `then` is a positive dividend."""
    then = now.shift(span)
    return (((now / then.where(then > 0)) ** (1.0 / years)) - 1.0
            ).replace([np.inf, -np.inf], np.nan)


def _ttm_dividends(dividends_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                   universe: list[str]) -> pd.DataFrame:
    """Daily trailing-12m cash dividend per share (date x ticker), 0 for names
    that never paid, aligned to the trading calendar. Point-in-time: the rolling
    sum at t only includes ex-dates <= t."""
    piv = dividends_hist.pivot_table(index="date", columns="ticker",
                                     values="dividends", aggfunc="sum")
    piv.index = pd.to_datetime(piv.index).normalize()
    # full universe so non-payers are a real 0, aligned to trading days
    piv = piv.reindex(index=idx, columns=universe).fillna(0.0)
    return piv.rolling(_YOY, min_periods=1).sum()


def _dividend_fields(dividends_hist: pd.DataFrame, close: pd.DataFrame,
                     fundamentals: pd.DataFrame | None) -> dict:
    idx = close.index
    universe = list(close.columns)
    ttm_ps = _ttm_dividends(dividends_hist, idx, universe)   # per-share TTM (source A)
    close_pos = close.where(close > 0)

    # ---- source B (SEC cash-flow statement): reconciliation + payout / coverage ----
    def _fd(field: str) -> pd.DataFrame:
        if fundamentals is None or fundamentals.empty:
            return pd.DataFrame()
        return fundamentals_to_daily(fundamentals, field, idx).reindex(columns=universe)

    div_paid = _fd("dividendsPaid")        # total cash dividends, TTM (source B)
    shares = _fd("sharesOutstanding")
    net_income = _fd("netIncome")
    fcf = _fd("freeCashflow")

    # RECONCILED total cash dividends: per-share x shares where the name actually paid
    # (source A, ttm_ps>0), else the SEC `dividendsPaid` total (source B) fills the gap.
    mcap = (shares * close).where(lambda x: x > 0) if not shares.empty else pd.DataFrame()
    total_a = (ttm_ps * shares).where(ttm_ps > 0) if not shares.empty else pd.DataFrame()
    if not total_a.empty:
        total = total_a.combine_first(div_paid) if not div_paid.empty else total_a
    else:
        total = div_paid

    F: dict[str, pd.DataFrame] = {}

    # ---- dividend yield (reconciled): precise per-share/price where paid, source-B
    # fill for names the ex-date history misses, real 0 for true non-payers ----
    yield_a = sanitize(ttm_ps.where(ttm_ps > 0) / close_pos)
    if not mcap.empty and not total.empty:
        F["dividend_yield"] = yield_a.combine_first(sanitize(total / mcap)).fillna(0.0)
    else:
        F["dividend_yield"] = sanitize(ttm_ps / close_pos).fillna(0.0)

    # ---- growth (1y + 5y CAGR), per-share source A primary, source-B total fills ----
    g1 = sanitize(ttm_ps / ttm_ps.shift(_YOY).where(lambda x: x > 0)) - 1.0
    g5 = _cagr(ttm_ps, _FIVE_Y, 5.0)
    if not div_paid.empty:
        g1 = g1.combine_first(sanitize(div_paid / div_paid.shift(_YOY).where(lambda x: x > 0)) - 1.0)
        g5 = g5.combine_first(_cagr(div_paid, _FIVE_Y, 5.0))
    F["dividend_growth"] = g1.replace([np.inf, -np.inf], np.nan)
    F["dividend_growth_5y"] = g5

    # ---- payer flag: paid in the trailing year per EITHER source ----
    payer = ttm_ps > 0
    if not div_paid.empty:
        payer = payer | (div_paid > 0)
    F["dividend_payer"] = payer.astype("float64")

    # ---- payout ratio + FCF coverage (dividend safety) off the reconciled total ----
    if not total.empty and not net_income.empty:
        F["dividend_payout_ratio"] = sanitize(total / net_income.where(net_income > 0)
                                                   ).clip(lower=0.0, upper=3.0)
    if not total.empty and not fcf.empty:
        # FCF / dividends: >1 => free cash flow covers the payout (safe); <1 or <0 => not.
        F["dividend_coverage"] = sanitize(fcf / total.where(total > 0))

    # ---- shareholder yield = dividend yield + buyback yield (- share issuance) ----
    if not shares.empty and shares.notna().any().any():
        buyback_yield = -(shares / shares.shift(_YOY) - 1.0)   # >0 => net buyback
        F["shareholder_yield"] = (F["dividend_yield"].add(buyback_yield, fill_value=0.0)
                                  .replace([np.inf, -np.inf], np.nan))
    return F


def build_dividend_feature_panel(
    dividends_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    stock_close: pd.DataFrame,
    fundamentals_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format dividend feature panel (`f_<name>_vs_peers`, `f_<name>_xs`).
    Empty if no dividend history is available."""
    if (dividends_history is None or dividends_history.empty
            or "dividends" not in dividends_history.columns or stock_close is None):
        return pd.DataFrame(columns=["date", "ticker"])
    close = stock_close.reindex(trading_index)
    fields = _dividend_fields(dividends_history, close, fundamentals_history)
    return build_peer_relative_panel(fields, peer_dict)
