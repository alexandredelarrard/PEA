"""
common.py  (src/strategies/analysis/common.py)
----------------------------------------------
Shared analytics primitives for the per-strategy + portfolio analysis modules: daily
cross-sectional IC, rolling Sharpe / drawdown / beta / correlation, rolling pairwise
correlation, and a loader for the market/energy reference return series (from
`prices_macro`) used in the L/S neutrality and trend crisis-alpha checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import MACRO_MARKET_SERIES
from src.strategies.utils.accuracy import forward_return
from src.utils.macro import load_macro_wide

_ANN: float = 252.0


def daily_ic(signal: pd.DataFrame, stock_ret: pd.DataFrame, horizon: int,
             min_frac: float = 0.6, min_names: int = 10) -> pd.Series:
    """Per-date cross-sectional rank IC: Spearman(signal_t, horizon-forward return_t) across
    names. This is the L/S model's day-by-day predictive power (rank correlation)."""
    fwd = forward_return(stock_ret, horizon, min_frac)
    out: dict = {}
    for date in signal.index:
        if date not in fwd.index:
            continue
        s = signal.loc[date].dropna()
        f = fwd.loc[date]
        common = s.index.intersection(f.dropna().index)
        if len(common) < min_names:
            continue
        out[date] = float(s[common].corr(f[common], method="spearman"))
    return pd.Series(out, dtype=float).sort_index()


def rolling_sharpe(ret: pd.Series, window: int = 126) -> pd.Series:
    r = ret.astype(float)
    return (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(_ANN)


def drawdown_series(ret: pd.Series) -> pd.Series:
    eq = (1.0 + ret.fillna(0.0)).cumprod()
    return (eq - eq.cummax()) / eq.cummax()


def rolling_beta(y: pd.Series, x: pd.Series, window: int = 126) -> pd.Series:
    y, x = y.align(x, join="inner")
    return y.rolling(window).cov(x) / x.rolling(window).var()


def rolling_corr(y: pd.Series, x: pd.Series, window: int = 126) -> pd.Series:
    y, x = y.align(x, join="inner")
    return y.rolling(window).corr(x)


def rolling_pairwise_corr(df: pd.DataFrame, window: int = 126) -> tuple[dict[str, pd.Series], pd.Series]:
    """Rolling correlation for every column pair + the average pairwise correlation over time.
    Each pair is ALIGNED (drop rows where either is NaN) BEFORE the rolling window, then
    reindexed — so a sleeve on a slightly different calendar (e.g. L/S vs the macro sleeves)
    isn't blanked out by `rolling().corr`'s default min_periods == window on the sparse union."""
    cols = list(df.columns)
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    d: dict[str, pd.Series] = {}
    for a, b in pairs:
        pair = df[[a, b]].dropna()
        c = pair[a].rolling(window).corr(pair[b]) if len(pair) > window else pd.Series(dtype=float)
        d[f"{a}-{b}"] = c.reindex(df.index)
    # ffill each pair before averaging so a sleeve missing the odd date (calendar gaps) doesn't
    # spike the overall average to the one pair that happens to have a value that day
    avg = (pd.concat(list(d.values()), axis=1).ffill().mean(axis=1) if d else pd.Series(dtype=float))
    return d, avg


def load_market_refs(store) -> dict[str, pd.Series]:
    """Reference daily returns from `prices_macro`: {'sp': the market series, 'energy': energy}.
    Used to check L/S market-neutrality (beta vs SP) and idiosyncrasy (corr vs energy)."""
    df = load_macro_wide(store, series=[MACRO_MARKET_SERIES, "energy"])
    out: dict[str, pd.Series] = {}
    if df is None:
        return out
    d = df.sort_values("date").set_index("date")
    if MACRO_MARKET_SERIES in d.columns:
        out["sp"] = d[MACRO_MARKET_SERIES].astype(float).pct_change(fill_method=None)
    if "energy" in d.columns:
        out["energy"] = d["energy"].astype(float).pct_change(fill_method=None)
    return out
