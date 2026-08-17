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
    forward_earnings_yield NTM (annual, forward-rolled) EPS / price = 1 / forward P/E
                           -- the historical, backtestable replacement for the
                           yfinance forwardPE snapshot. NTM EPS = next-quarter
                           consensus estimate + trailing 3 reported actuals (only
                           the newest quarter is an estimate -> leak-free).
    eps_expectation_growth next-quarter estimate vs last reported EPS
                           (high => market pricing in aggressive growth)
    eps_surprise_last      most recent reported surprise % (beat > 0 / miss < 0)
    eps_surprise_4q_avg    trailing 4-quarter mean surprise (consistency of beats)

Also exposes `ntm_ttm_eps()` (NTM & TTM annual EPS) for PEGY's projected-growth term.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.frames import ratio
from src.data_aggregate.utils.common.panel import build_peer_relative_panel

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


def _trailing_actual_sum(df: pd.DataFrame, idx: pd.DatetimeIndex, window: int) -> pd.DataFrame:
    """Trailing `window`-quarter SUM of REPORTED eps_actual, forward-filled from each
    report date (point-in-time; requires a full `window` of reported quarters)."""
    sub = df.dropna(subset=["eps_actual"]).copy()
    if sub.empty:
        return pd.DataFrame(index=idx)
    sub["v"] = sub.groupby("ticker")["eps_actual"].transform(
        lambda s: s.rolling(window, min_periods=window).sum())
    wide = sub.pivot_table(index="earnings_date", columns="ticker",
                           values="v", aggfunc="last").sort_index()
    return wide.reindex(wide.index.union(idx)).ffill().reindex(idx)


def _ntm_ttm_from_prepped(df: pd.DataFrame, idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ntm_eps, ttm_eps) daily wide frames from a prepped earnings frame.

    ttm_eps = trailing 4 REPORTED quarterly actuals (annual trailing EPS).
    ntm_eps = next-quarter consensus estimate + trailing 3 reported actuals -- a
    leak-free, forward-ROLLED annual EPS: only the newest quarter is an estimate
    (the same near-report consensus used by fwd_eps_yield), the other three are
    reported actuals. A true 4-quarter-ahead consensus is NOT reconstructable
    leak-free here (yfinance stores one estimate per quarter, so q+2..q+4 would use
    values not known at the as-of date)."""
    fwd_eps = _forward_to_daily(df, "eps_estimate", idx)     # next quarter (est), PIT window
    ttm4 = _trailing_actual_sum(df, idx, 4)
    ttm3 = _trailing_actual_sum(df, idx, 3)
    ntm = fwd_eps + ttm3 if not fwd_eps.empty and not ttm3.empty else pd.DataFrame(index=idx)
    return ntm, ttm4


def ntm_ttm_eps(earnings_history: pd.DataFrame | None,
                idx: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Public helper: forward-rolled annual EPS (NTM) and trailing annual EPS (TTM)
    from the earnings-surprise archive. Reused by fundamental_features for PEGY's
    projected-growth term. Empty frames when the archive is unavailable."""
    if (earnings_history is None or earnings_history.empty
            or "earnings_date" not in earnings_history.columns):
        return pd.DataFrame(index=idx), pd.DataFrame(index=idx)
    return _ntm_ttm_from_prepped(_prep(earnings_history), idx)


_EPOCH = pd.Timestamp("1970-01-01")


def days_since_earnings(df: pd.DataFrame, idx: pd.DatetimeIndex,
                        cap_days: int = 180) -> pd.DataFrame:
    """Wide [date x ticker] CALENDAR days since the most recent PAST earnings report
    (0 on the report day, rising to ~90+ as the next report approaches, then resetting).

    Leak-free / point-in-time: on date t it uses only earnings dates on/before t (the
    last report's date is public). NaN before a ticker's first report; clipped to
    [0, cap_days] so a late/skipped quarter or data gap can't produce an outlier.
    A near-zero value flags the post-earnings-announcement-drift (PEAD) window."""
    sub = df.dropna(subset=["earnings_date"])[["ticker", "earnings_date"]].copy()
    if sub.empty:
        return pd.DataFrame(index=idx)
    # numeric ordinal (days since epoch) so the pivot/ffill stays a clean float path
    sub["ord"] = (sub["earnings_date"] - _EPOCH).dt.days
    wide = sub.pivot_table(index="earnings_date", columns="ticker", values="ord",
                           aggfunc="last").sort_index()
    # last report ordinal known on/before each trading day (forward-fill the date)
    wide = wide.reindex(wide.index.union(idx)).ffill().reindex(idx)
    idx_ord = pd.Series((idx - _EPOCH).days, index=idx)
    days = wide.rsub(idx_ord, axis=0)                       # idx_ord - last_report_ord
    return days.clip(lower=0, upper=cap_days)


def _derived_earnings_fields(hist: pd.DataFrame, idx: pd.DatetimeIndex,
                             close: pd.DataFrame) -> dict:
    df = _prep(hist)
    F: dict[str, pd.DataFrame] = {}

    fwd_eps = _forward_to_daily(df, "eps_estimate", idx)
    last_actual = _realized_to_daily(df, "eps_actual", idx)
    price = close.reindex(idx)

    if not fwd_eps.empty:
        F["fwd_eps_yield"] = ratio(fwd_eps, price)                 # next-quarter forward E/P
        if not last_actual.empty:
            F["eps_expectation_growth"] = ratio(fwd_eps, last_actual, positive_den=True) - 1.0

    # NTM (annual, forward-rolled) forward-earnings yield = 1 / forward P/E -- the
    # historical, backtestable replacement for the yfinance forwardPE snapshot.
    ntm_eps, _ = _ntm_ttm_from_prepped(df, idx)
    if not ntm_eps.empty and ntm_eps.notna().any().any():
        F["forward_earnings_yield"] = ratio(ntm_eps, price)        # NTM E/P (higher = cheaper)

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

    prepped = _prep(earnings_history)
    fields = _derived_earnings_fields(earnings_history, trading_index, stock_close)
    panel = build_peer_relative_panel(fields, peer_dict)

    # RAW calendar signal (NOT peer-relative): days since the most recent earnings.
    # Same meaning for every name, so it is emitted as a plain `f_days_since_earnings`
    # (the model splits/loads on the raw value; PEAD decays as this rises).
    dse = days_since_earnings(prepped, trading_index)
    if not dse.empty and dse.notna().any().any():
        long = dse.stack()
        long.index.set_names(["date", "ticker"], inplace=True)
        long = long.rename("f_days_since_earnings").reset_index()
        panel = long if (panel is None or panel.empty) else \
            panel.merge(long, on=["date", "ticker"], how="outer")
    return panel
