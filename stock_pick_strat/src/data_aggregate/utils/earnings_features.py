"""
earnings_features.py  (src/data_aggregate/utils/earnings_features.py)
---------------------------------------------------------------------
Peer-relative features built from the HISTORICAL earnings-surprise archive
(fetch_earnings_surprises: per quarter, the consensus EPS estimate, the reported
EPS, and the surprise). These finally give a genuinely historical, backtestable
view of "what did the market expect, and did the company beat or miss?".

Point-in-time discipline (two directions):
  * FORWARD expectation (`fwd_eps`): the consensus estimate for the NEXT, not-yet
    reported quarter, back-filled only into the ~quarter of trading days leading
    up to that report (`FWD_FILL_LIMIT`). It is an estimate of a FUTURE outcome,
    so it never leaks the reported number. (We cannot reconstruct how the
    estimate was revised intra-quarter -- yfinance gives one consensus per date
    -- so this is the final pre-report consensus, applied within its own quarter.)
  * REALIZED (`eps_surprise_*`, `last_actual`): forward-filled only FROM the
    earnings date onward, i.e. once the number is public. No look-ahead.

Characteristics
    fwd_eps_yield          next-quarter consensus EPS / price (forward E/P)
    eps_expectation_growth next-quarter estimate vs last reported EPS
                           (high => market pricing in aggressive growth)
    eps_surprise_last      most recent reported surprise % (beat > 0 / miss < 0)
    eps_surprise_4q_avg    trailing 4-quarter mean surprise (consistency of beats)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamental_features import _ratio, build_peer_relative_panel

# how many trading days before a report the forward estimate is allowed to apply
FWD_FILL_LIMIT = 95
SURPRISE_WINDOW = 4


def _prep(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.normalize()
    for c in ("eps_estimate", "eps_actual", "surprise_pct"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["ticker", "earnings_date"])


def _forward_to_daily(df: pd.DataFrame, value_col: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Value keyed on a FUTURE earnings date, back-filled into the trading days
    leading up to it (bounded to one quarter)."""
    sub = df.dropna(subset=[value_col])
    if sub.empty:
        return pd.DataFrame(index=idx)
    wide = sub.pivot_table(index="earnings_date", columns="ticker",
                           values=value_col, aggfunc="last").sort_index()
    return wide.reindex(wide.index.union(idx)).bfill(limit=FWD_FILL_LIMIT).reindex(idx)


def _realized_to_daily(df: pd.DataFrame, value_col: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Value keyed on a PAST earnings date, forward-filled from the day it was
    reported (point-in-time, no look-ahead)."""
    sub = df.dropna(subset=[value_col])
    if sub.empty:
        return pd.DataFrame(index=idx)
    wide = sub.pivot_table(index="earnings_date", columns="ticker",
                           values=value_col, aggfunc="last").sort_index()
    return wide.reindex(wide.index.union(idx)).ffill().reindex(idx)


def _reported_rolling_to_daily(df: pd.DataFrame, value_col: str,
                               idx: pd.DatetimeIndex, window: int) -> pd.DataFrame:
    """Trailing mean over the last `window` REPORTED quarters, forward-filled."""
    sub = df.dropna(subset=[value_col]).copy()
    if sub.empty:
        return pd.DataFrame(index=idx)
    sub["v"] = sub.groupby("ticker")[value_col].transform(
        lambda s: s.rolling(window, min_periods=1).mean())
    wide = sub.pivot_table(index="earnings_date", columns="ticker",
                           values="v", aggfunc="last").sort_index()
    return wide.reindex(wide.index.union(idx)).ffill().reindex(idx)


def _derived_earnings_fields(hist: pd.DataFrame, idx: pd.DatetimeIndex,
                             close: pd.DataFrame) -> dict:
    df = _prep(hist)
    F: dict[str, pd.DataFrame] = {}

    fwd_eps = _forward_to_daily(df, "eps_estimate", idx)
    last_actual = _realized_to_daily(df, "eps_actual", idx)
    price = close.reindex(idx)

    if not fwd_eps.empty:
        F["fwd_eps_yield"] = _ratio(fwd_eps, price)                 # forward E/P (may be <0)
        if not last_actual.empty:
            F["eps_expectation_growth"] = _ratio(fwd_eps, last_actual, positive_den=True) - 1.0

    last_surprise = _realized_to_daily(df, "surprise_pct", idx)
    if not last_surprise.empty:
        F["eps_surprise_last"] = last_surprise
    avg_surprise = _reported_rolling_to_daily(df, "surprise_pct", idx, SURPRISE_WINDOW)
    if not avg_surprise.empty:
        F["eps_surprise_4q_avg"] = avg_surprise
    return F


def build_earnings_feature_panel(
    earnings_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    stock_close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format earnings-expectation feature panel (`f_<name>_vs_peers`,
    `f_<name>_xs`). Empty if the earnings history or prices are unavailable."""
    if (earnings_history is None or earnings_history.empty
            or stock_close is None or "earnings_date" not in earnings_history.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    fields = _derived_earnings_fields(earnings_history, trading_index, stock_close)
    return build_peer_relative_panel(fields, peer_dict)
