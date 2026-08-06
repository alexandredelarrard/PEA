"""
factors.py  (src/data_aggregate/utils/factors.py)
--------------------------------------------------
Build the FACTOR RETURN SERIES we residualize the label against, so epsilon is
the pure firm-specific move after stripping every common, cheaply-harvestable
driver -- not just market and sector.

Two families:

STYLE factors (cross-sectional, one long-short return series each):
    size        -log(market cap)          small-minus-big
    value       earnings + fcf yield      cheap-minus-expensive
    momentum    12-1 price return         winners-minus-losers
    quality     ROE+margins-leverage      quality-minus-junk
    resvol      -trailing vol             low-vol-minus-high-vol
  Built as CHARACTERISTIC-WEIGHTED portfolios: each day, cross-sectionally
  standardize the characteristic, use the LAGGED standardized score as the
  weight on the next day's return. weight formed at t, return realized at t+1
  => no look-ahead. factor_ret(t) = sum_i w_i(t-1) * ret_i(t), scaled.

MACRO factors (one CHANGE series each; a stock's sensitivity is its beta):
    d_yield_10y, d_yield_curve, d_vix, d_cpi_yoy, d_fed_balance_sheet,
    d_ig_spread, d_hy_spread
  Raw macro LEVELS are identical across stocks => zero cross-sectional info.
  We regress returns on macro CHANGES; each stock's macro beta is what we
  neutralize (oil doesn't pick stocks, oil-beta does).

DATA-AVAILABILITY NOTE
----------------------
momentum / size / resvol  -> price-only, fully historical.
macro                     -> FRED, fully historical.
value / quality           -> need fundamentals HISTORY. With snapshot-only
                             fundamentals these exist only from when you started
                             collecting; earlier dates come back NaN and the
                             factor is simply skipped for those dates. Backfill
                             via SimFin for real history.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd

# the point-in-time pivot + daily market cap moved to utils/common/pit.py, and the
# price-derived primitives to utils/common/prices.py: they are generic, and keeping them
# here made this factor-RETURN module a dependency of nine feature builders that only
# wanted a point-in-time frame or a rolling std.
from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.common.prices import momentum_characteristic, trailing_vol
from src.data_aggregate.utils.common.xs import XS_CLIP_CHARACTERISTIC, xs_z

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Characteristics                                                              #
# --------------------------------------------------------------------------- #
def build_characteristics(
    stock_close: pd.DataFrame,
    stock_ret: pd.DataFrame,
    fundamentals_history: pd.DataFrame | None,
    resvol_window: int = 63,
) -> dict:
    """
    Returns {factor_name: characteristic DataFrame (date x ticker)}, higher =
    more of that factor's long side. Only price-based factors are guaranteed;
    value/quality appear only if fundamentals are available.
    """
    idx = stock_close.index
    chars: dict[str, pd.DataFrame] = {}

    # Momentum 12-1 (skip most recent month).
    chars["momentum"] = momentum_characteristic(stock_close)

    # Residual/low volatility: negative trailing vol (low vol = long side).
    chars["resvol"] = -trailing_vol(stock_ret, resvol_window)

    if fundamentals_history is not None and not fundamentals_history.empty:
        # Historical daily market cap from SEC shares * price (moves daily).
        mcap = daily_market_cap(fundamentals_history, stock_close)
        if mcap.empty:
            # fallback to old proxy if only a current marketCap snapshot exists
            snap = fundamentals_to_daily(fundamentals_history, "marketCap", idx)
            if not snap.empty:
                latest_px = stock_close.reindex(columns=snap.columns).ffill().iloc[-1]
                sh = (snap.iloc[-1] / latest_px).replace([np.inf, -np.inf], np.nan)
                mcap = stock_close.reindex(columns=snap.columns).mul(sh, axis=1)

        if not mcap.empty:
            # Size: -log market cap (small = long side).
            chars["size"] = -np.log(mcap.where(mcap > 0))

            # Value: earnings yield + FCF yield + book/price, all vs market cap.
            ni = fundamentals_to_daily(fundamentals_history, "netIncome", idx)
            fcf = fundamentals_to_daily(fundamentals_history, "freeCashflow", idx)
            eq = fundamentals_to_daily(fundamentals_history, "stockholdersEquity", idx)
            val_parts = []
            for num in (ni, fcf, eq):
                if not num.empty:
                    common = [c for c in num.columns if c in mcap.columns]
                    if common:
                        yld = (num[common] / mcap[common]).replace([np.inf, -np.inf], np.nan)
                        val_parts.append(xs_z(yld, clip=XS_CLIP_CHARACTERISTIC))
            if val_parts:
                chars["value"] = sum(val_parts) / len(val_parts)

        # Quality: ROE + gross margin + profit margin - leverage (SEC-derived).
        roe = fundamentals_to_daily(fundamentals_history, "returnOnEquity", idx)
        gm = fundamentals_to_daily(fundamentals_history, "grossMargins", idx)
        pm = fundamentals_to_daily(fundamentals_history, "profitMargins", idx)
        de = fundamentals_to_daily(fundamentals_history, "debtToEquity", idx)
        q_parts = []
        for f in (roe, gm, pm):
            if not f.empty:
                q_parts.append(xs_z(f, clip=XS_CLIP_CHARACTERISTIC))
        if not de.empty:
            q_parts.append(-xs_z(de, clip=XS_CLIP_CHARACTERISTIC))
        if q_parts:
            chars["quality"] = sum(q_parts) / len(q_parts)

    return chars


# --------------------------------------------------------------------------- #
# Characteristic -> factor return                                              #
# --------------------------------------------------------------------------- #
def characteristic_to_factor_return(char: pd.DataFrame, stock_ret: pd.DataFrame) -> pd.Series:
    """
    Characteristic-weighted long-short daily factor return.
      w_i(t-1) = xs-standardized characteristic (mean 0), normalized to unit
                 gross so it is dollar-neutral and scale-stable;
      f(t)     = sum_i w_i(t-1) * ret_i(t).
    Lagged weights => no look-ahead.
    """
    z = xs_z(char, clip=XS_CLIP_CHARACTERISTIC)
    z = z.sub(z.mean(axis=1), axis=0)                      # ensure long-short (mean 0)
    gross = z.abs().sum(axis=1).replace(0, np.nan)
    w = z.div(gross, axis=0)                               # unit gross exposure
    aligned = stock_ret.reindex_like(w)
    f = (w.shift(1) * aligned).sum(axis=1, min_count=1)
    return f.rename(char.name if char.name else "factor")


def build_style_factor_returns(
    stock_close: pd.DataFrame,
    stock_ret: pd.DataFrame,
    fundamentals_history: pd.DataFrame | None,
    resvol_window: int = 63,
) -> pd.DataFrame:
    """Return DataFrame (date x factor) of daily style-factor returns."""
    chars = build_characteristics(stock_close, stock_ret, fundamentals_history, resvol_window)
    cols = {}
    for name, c in chars.items():
        c.name = name
        cols[name] = characteristic_to_factor_return(c, stock_ret)
    return pd.DataFrame(cols)


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


# `commodity_factor_returns` / `currency_factor_returns` had byte-identical bodies and are
# now the single `prices.price_column_returns(close, tickers)`; `assemble_factor_panel`
# already treated them identically (both concatenated as returns, neither in `macro_cols`).

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
        logger.warning("dropped non-daily-moving factors: %s", dropped)
    return panel[keep], dropped


def assemble_factor_panel(
    market_ret: pd.Series,
    style_factors: pd.DataFrame,      # size, value, momentum, quality, resvol (returns)
    commodity_returns: pd.DataFrame,  # oil, gold (returns)
    currency_returns: pd.DataFrame,  # USD/EUR (returns)
    macro_changes: pd.DataFrame,      # d_yield_10y, d_vix, d_breakeven_10y (changes)
) -> tuple[pd.DataFrame, list[str]]:
    """
    Assemble the shared factor panel and return (panel, macro_cols).

    macro_cols = the CHANGE columns (yields/vix/breakeven), which targets.py
    forward-accumulates via cumulative sum. Market, style, and COMMODITY columns
    are returns and are compounded forward -- so commodity is NOT in macro_cols.
    """
    panel = pd.concat(
        [market_ret.rename("market"), style_factors, commodity_returns, currency_returns, macro_changes],
        axis=1,
    )
    panel, dropped = filter_daily_factors(panel)
    macro_cols = [c for c in macro_changes.columns if c in panel.columns]
    return panel, macro_cols
