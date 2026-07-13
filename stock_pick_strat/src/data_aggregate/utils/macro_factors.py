"""
macro_factors.py  (src/data_aggregate/utils/macro_factors.py)
-------------------------------------------------------------
Build the macro/commodity factors we NEUTRALIZE, using only series that move at
DAILY frequency -- so their beta is estimable and the neutralization is real.

Why this exists: monthly CPI and the weekly Fed balance sheet, forward-filled to
daily and first-differenced, are ~zero on almost every day. As daily regressors
they carry no variance, so no meaningful beta can be fit and the risk is NOT
neutralized (dropping them, as the old try/except did, just hides this). The fix
is to proxy each slow macro risk with a market-traded series that moves daily:

    rate / fed risk  -> daily Treasury yield changes (d_yield_10y, d_yield_curve)
    inflation risk   -> daily 10y BREAKEVEN change (FRED T10YIE)  [replaces CPI]
    commodity risk   -> daily OIL and GOLD returns (CL=F/GC=F or USO/GLD)
    volatility risk  -> daily VIX change

Classification for targets.py:
  * commodity factors are RETURNS  -> compounded forward (NOT in macro_cols)
  * yield/vix/breakeven are CHANGES -> cumulative forward (IN macro_cols)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# FRED level -> daily-change factor name. ONLY daily-moving series belong here.
# NOTE: cpi_yoy_pct (monthly) and fed_balance_sheet (weekly) are deliberately
# EXCLUDED -- their daily change is ~always zero. Inflation risk is captured by
# the daily breakeven instead.
DAILY_MACRO_LEVELS = {
    "yield_10y": "d_yield_10y",
    "yield_curve_10y2y": "d_yield_curve",
    "vix": "d_vix",
    "breakeven_10y": "d_breakeven_10y",   # FRED T10YIE (add to fetch_macro)
}


def macro_change_factors(
    macro_df: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    level_to_change: dict | None = None,
) -> pd.DataFrame:
    """Daily first-differences of the daily-moving macro levels only."""
    mapping = level_to_change or DAILY_MACRO_LEVELS
    m = macro_df.copy()
    if "date" in m.columns:
        m["date"] = pd.to_datetime(m["date"])
        m = m.set_index("date")
    m = m.sort_index()

    out = {}
    for level, change in mapping.items():
        if level in m.columns:
            s = m[level].reindex(m.index.union(trading_index)).ffill().reindex(trading_index)
            out[change] = s.diff()
    return pd.DataFrame(out, index=trading_index)


def commodity_factor_returns(
    close: pd.DataFrame,
    tickers: dict | None = None,
) -> pd.DataFrame:
    """
    Daily RETURNS of commodity proxies, taken from the price panel (they flow
    through the normal price pipeline as `other_tickers`).

    tickers maps factor name -> price column, e.g.
        {"oil": "CL=F", "gold": "GC=F"}   or   {"oil": "USO", "gold": "GLD"}
    """
    tickers = tickers or {"oil": "CL=F", "gold": "GC=F"}
    out = {}
    for name, col in tickers.items():
        if col in close.columns:
            out[name] = close[col].pct_change()
    return pd.DataFrame(out, index=close.index)


def filter_daily_factors(
    panel: pd.DataFrame,
    max_zero_frac: float = 0.30,
    max_nan_frac: float = 0.50,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Keep only columns that genuinely move at daily frequency. A return/change
    column that is exactly zero on > max_zero_frac of days (a stale low-frequency
    series resampled to daily) cannot support a beta and is dropped. This is the
    principled replacement for the hardcoded try/except drop.

    Returns (clean_panel, dropped_columns).
    """
    keep, dropped = [], []
    for c in panel.columns:
        s = panel[c]
        nan_frac = float(s.isna().mean())
        nonnan = s.dropna()
        zero_frac = float((nonnan.abs() < 1e-12).mean()) if len(nonnan) else 1.0
        if nan_frac > max_nan_frac or zero_frac > max_zero_frac:
            dropped.append(c)
        else:
            keep.append(c)
    if verbose and dropped:
        print(f"[filter_daily_factors] dropped non-daily-moving factors: {dropped}")
    return panel[keep], dropped


def assemble_factor_panel(
    market_ret: pd.Series,
    style_factors: pd.DataFrame,      # size, value, momentum, quality, resvol (returns)
    commodity_returns: pd.DataFrame,  # oil, gold (returns)
    macro_changes: pd.DataFrame,      # d_yield_10y, d_vix, d_breakeven_10y (changes)
) -> tuple[pd.DataFrame, list[str]]:
    """
    Assemble the shared factor panel and return (panel, macro_cols).

    macro_cols = the CHANGE columns (yields/vix/breakeven), which targets.py
    forward-accumulates via cumulative sum. Market, style, and COMMODITY columns
    are returns and are compounded forward -- so commodity is NOT in macro_cols.
    """
    panel = pd.concat(
        [market_ret.rename("market"), style_factors, commodity_returns, macro_changes],
        axis=1,
    )
    panel, dropped = filter_daily_factors(panel)
    macro_cols = [c for c in macro_changes.columns if c in panel.columns]
    return panel, macro_cols