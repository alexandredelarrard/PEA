"""
dividend_features.py  (src/data_aggregate/utils/dividend_features.py)
---------------------------------------------------------------------
Peer-relative DIVIDEND / SHAREHOLDER-YIELD features from the ex-date dividend
history (fetch_dividends). All point-in-time (a dividend enters TTM only on/after
its ex-date, and growth compares past-vs-past):

    dividend_yield     trailing-12m cash dividends / price   (income / value tilt)
    dividend_growth    TTM dividends vs TTM one year ago      (payout trajectory)
    dividend_payer     1 if the firm paid a dividend in the trailing year
    shareholder_yield  dividend_yield + buyback yield
                       (buyback yield = -YoY change in shares outstanding; net
                        issuance/dilution lowers it, buybacks raise it)

Non-payers get a real 0 dividend yield (not NaN) so they rank correctly in the
cross-section; shareholder_yield still captures their buybacks/dilution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily
from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel

_YOY = 252  # ~1 trading year


def _ttm_dividends(dividends_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                   universe: list[str]) -> pd.DataFrame:
    """Daily trailing-12m cash dividend per share (date x ticker), 0 for names
    that never paid, aligned to the trading calendar. Point-in-time: the rolling
    sum at t only includes ex-dates <= t."""
    piv = dividends_hist.pivot_table(index="date", columns="ticker",
                                     values="dividend", aggfunc="sum")
    piv.index = pd.to_datetime(piv.index).normalize()
    # full universe so non-payers are a real 0, aligned to trading days
    piv = piv.reindex(index=idx, columns=universe).fillna(0.0)
    return piv.rolling(_YOY, min_periods=1).sum()


def _dividend_fields(dividends_hist: pd.DataFrame, close: pd.DataFrame,
                     fundamentals: pd.DataFrame | None) -> dict:
    idx = close.index
    ttm = _ttm_dividends(dividends_hist, idx, list(close.columns))

    close_pos = close.where(close > 0)
    F: dict[str, pd.DataFrame] = {}
    F["dividend_yield"] = (ttm / close_pos).replace([np.inf, -np.inf], np.nan)
    prev = ttm.shift(_YOY)
    F["dividend_growth"] = (ttm / prev.where(prev > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
    F["dividend_payer"] = (ttm > 0).astype("float64")

    # shareholder yield = dividend yield + buyback yield (- share issuance)
    if fundamentals is not None and not fundamentals.empty:
        shares = fundamentals_to_daily(fundamentals, "sharesOutstanding", idx)
        if not shares.empty and shares.notna().any().any():
            shares = shares.reindex(columns=close.columns)
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
            or "dividend" not in dividends_history.columns or stock_close is None):
        return pd.DataFrame(columns=["date", "ticker"])
    close = stock_close.reindex(trading_index)
    fields = _dividend_fields(dividends_history, close, fundamentals_history)
    return build_peer_relative_panel(fields, peer_dict)
