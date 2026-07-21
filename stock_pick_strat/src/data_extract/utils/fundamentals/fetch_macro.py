"""
Fetch macro series from FRED (free, complete history, needs a free API key) and save
them to the `macro` DB table (PK: date). Series: Treasury yields (3M/2Y/10Y/30Y), the
10Y-2Y / 10Y-3M curve spreads, VIX, the Moody's Baa credit spread, 10Y breakeven
inflation — see SERIES below.

Credit spread: Moody's Seasoned Baa Corporate Bond yield over the 10Y Treasury
(`baa_credit_spread`, FRED `BAA10Y`). It replaces the ICE BofA IG/HY OAS spreads, which
FRED truncates to ~3 years (ICE licensing) — Baa-10Y is a single real series with one
consistent definition across the full history (same noise a decade ago as today).

The daily series drop the odd day (market holidays / one-off FRED misses); those short
interior gaps are filled with the mean of the two bracketing observed days (only when the
gap is shorter than a week).

Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html
Put it in .env as FRED_API_KEY=...
"""
import os

import pandas as pd
from fredapi import Fred

from src.context import Context

SERIES = {
    # Risk-free rate
    "DGS3MO": "yield_3m",
    # Yield curve
    "DGS2": "yield_2y",
    "DGS10": "yield_10y",
    "DGS30": "yield_30y",
    "T10Y2Y": "yield_curve_10y2y",   # FRED computes this spread directly
    "T10Y3M": "yield_curve_10y3m",
    # Volatility
    "VIXCLS": "vix",
    # Credit spread: Moody's Baa corporate bond yield over 10Y Treasury -> one real,
    # full-history, consistently-defined series (no ICE ~3y truncation, no splice).
    "BAA10Y": "baa_credit_spread",
    # Inflation expectations
    "T10YIE": "breakeven_10y",
}

MAX_GAP_DAYS = 7                 # fill sporadic daily gaps strictly shorter than this


def fill_short_gaps(df: pd.DataFrame, cols: list[str],
                    max_gap_days: int = MAX_GAP_DAYS) -> pd.DataFrame:
    """Fill sporadic interior NaN runs (market holidays / one-off FRED misses) in each of
    `cols` with the MEAN of the two bracketing observed values, but ONLY when the gap spans
    fewer than `max_gap_days` calendar days. Longer outages and leading / trailing NaNs are
    left untouched. `df` must be indexed by a DatetimeIndex."""
    df = df.sort_index()
    idx = pd.Series(df.index, index=df.index)
    for c in cols:
        s = df[c]
        gap = s.isna()
        if not gap.any():
            continue
        prev_val, next_val = s.ffill(), s.bfill()
        obs = idx.where(s.notna())                       # observation date, else NaT
        span_days = (obs.bfill() - obs.ffill()).dt.days  # bracketing-days distance
        fillable = gap & prev_val.notna() & next_val.notna() & (span_days < max_gap_days)
        df.loc[fillable, c] = (prev_val[fillable] + next_val[fillable]) / 2.0
    return df


# Core daily LEVEL series (Treasury yields + VIX): FRED publishes these with a ~1
# business-day lag. The COMPUTED series (T10Y2Y / T10Y3M spreads, T10YIE breakeven)
# publish SAME-DAY, so "up to date" must NOT be judged on the overall max date — a
# fast series would mark the table current while the level block is still stale,
# skipping the very refresh that would fill the yields/VIX (the "1-day gap" bug).
_CORE_LEVEL_SERIES = ["yield_3m", "yield_2y", "yield_10y", "yield_30y", "vix"]


def _macro_is_up_to_date(context: Context) -> bool:
    """True only when the CORE LEVEL series (Treasury yields + VIX) already reach the
    previous business day — FRED's realistic freshest for them (they lag ~1 BDay; the
    computed spreads/breakeven publish same-day and come along for free). Keying on
    the overall max let a same-day spread mask a stale level block and skip the
    refresh; keying the level block on `today` would instead re-pull every run (they
    never reach today). The previous business day settles both."""
    existing = context.store.load("macro")
    if existing is None or existing.empty or "date" not in existing.columns:
        return False
    existing["date"] = pd.to_datetime(existing["date"])
    last_expected = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    core = [c for c in _CORE_LEVEL_SERIES if c in existing.columns]
    if not core:                                   # legacy table -> fall back to overall max
        return existing["date"].max() >= last_expected
    # the LAGGIEST core series drives it: up-to-date only if ALL of them reach yesterday
    core_last = min(existing.loc[existing[c].notna(), "date"].max() for c in core)
    return bool(pd.notna(core_last) and core_last >= last_expected)


def _refresh_macro(context: Context) -> None:
    """Download every series, fill short daily gaps, and rewrite the `macro` DB table."""
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    start = pd.Timestamp.today() - pd.DateOffset(years=context.config.data_extract.years_history + 1)

    macro = pd.DataFrame({name: fred.get_series(sid, observation_start=start)
                          for sid, name in SERIES.items()})
    macro.index.name = "date"

    # sporadic daily gaps (holidays / one-off FRED misses) -> mean of the bracketing days
    macro = fill_short_gaps(macro, list(macro.columns))

    macro = macro.reset_index()
    # replace (not upsert): the full history is re-fetched every run, so a clean rewrite
    # keeps the table consistent and never leaves stale rows if a series/definition changes.
    context.store.replace("macro", macro)
    context.log.info("Saved %d rows of macro data to DB table 'macro' (short gaps filled)", len(macro))


def fetch_macro(context: Context):
    if not os.getenv("FRED_API_KEY"):
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and add it "
            "to your .env file."
        )
    if _macro_is_up_to_date(context):
        context.log.info("Macro data already up to date - skipping (DB table 'macro')")
        return
    _refresh_macro(context)
