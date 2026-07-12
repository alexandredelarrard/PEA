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
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Point-in-time fundamentals -> daily frame                                    #
# --------------------------------------------------------------------------- #
def fundamentals_to_daily(
    fundamentals_history: pd.DataFrame,
    field: str,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Turn a (ticker, as_of, <fields>) history into a daily wide frame for one
    field, forward-filled point-in-time: value on date d is the most recent
    as_of <= d. No look-ahead.
    """
    if field not in fundamentals_history.columns:
        return pd.DataFrame(index=trading_index)
    df = fundamentals_history[["ticker", "as_of", field]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    wide = df.pivot_table(index="as_of", columns="ticker", values=field, aggfunc="last")
    wide = wide.sort_index().reindex(
        wide.index.union(trading_index)
    ).ffill().reindex(trading_index)
    return wide


def _xs_z(df: pd.DataFrame, clip: float = 4.0) -> pd.DataFrame:
    """Cross-sectional z-score per day, clipped."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-clip, clip)


def momentum_characteristic(stock_close: pd.DataFrame) -> pd.DataFrame:
    """12-1 price momentum characteristic (skip the most recent month).

    Single source of truth: feeds the momentum style factor, the `mom_12_1`
    model feature, AND the target's momentum neutralization. Point-in-time
    (uses only past prices), so it never leaks future information.
    """
    return stock_close.shift(21) / stock_close.shift(252) - 1.0


def daily_market_cap(fundamentals_history: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    Historical daily market cap = point-in-time shares outstanding (from SEC,
    forward-filled) * daily close. This is the correct historical mcap (moves
    with price every day), replacing the old current-mcap*price-ratio proxy.
    Requires a 'sharesOutstanding' column in the fundamentals history.
    """
    shares = fundamentals_to_daily(fundamentals_history, "sharesOutstanding", close.index)
    if shares.empty:
        return pd.DataFrame(index=close.index)
    cols = [c for c in shares.columns if c in close.columns]
    if not cols:
        return pd.DataFrame(index=close.index)
    mcap = close[cols].mul(shares[cols])
    return mcap.where(mcap > 0)


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
    chars["resvol"] = -stock_ret.rolling(resvol_window).std()

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
                        val_parts.append(_xs_z(yld))
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
                q_parts.append(_xs_z(f))
        if not de.empty:
            q_parts.append(-_xs_z(de))
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
    z = _xs_z(char)
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


# --------------------------------------------------------------------------- #
# Macro factor changes                                                         #
# --------------------------------------------------------------------------- #
DEFAULT_MACRO_LEVELS = {
    "yield_10y": "d_yield_10y",
    "yield_curve_10y2y": "d_yield_curve",
    "vix": "d_vix",
    # "cpi_yoy_pct": "d_cpi_yoy",
    # "fed_balance_sheet": "d_fed_balance_sheet",
    # "ig_credit_spread": "d_ig_spread",
    # "hy_credit_spread": "d_hy_spread",
}


def build_macro_factor_changes(
    macro_df: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    level_to_change: dict | None = None,
) -> pd.DataFrame:
    """
    Reindex macro levels onto trading days (ffill) and first-difference them.
    Weekly/monthly series (fed balance sheet, CPI) are ffilled then differenced,
    so the change lands on the day new info arrives -- point-in-time.
    """
    mapping = level_to_change or DEFAULT_MACRO_LEVELS
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


# --------------------------------------------------------------------------- #
# Assemble the shared factor panel (everything except sector, which is per-stock)
# --------------------------------------------------------------------------- #
def assemble_factor_panel(
    market_ret: pd.Series,
    style_factors: pd.DataFrame,
    macro_changes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Shared regressors (same series for every stock): market + style + macro.
    Sector is per-stock and injected in the beta regression separately.
    """
    panel = pd.concat(
        [market_ret.rename("market"), style_factors, macro_changes],
        axis=1,
    )
    return panel
