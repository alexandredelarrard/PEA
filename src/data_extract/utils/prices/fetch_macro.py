"""
fetch_macro.py  (src/data_extract/utils/prices/fetch_macro.py)
--------------------------------------------------------------
THE macro / market fetcher: every non-equity daily series the pipeline uses, written to the
long `prices_macro` table as (date, ticker, close) where `ticker` is the SERIES name.

Three legs, in order:
  1. PRICES  -- `download_ohlcv` (the shared yfinance entry point), close only. Nothing is
     written to `prices`, which is what keeps it the equity universe and nothing else.
  2. FRED    -- rates / credit / breakeven / FX LEVELS (needs a free key, see below).
  3. DERIVED -- the two curve spreads and the reconstructed 10Y total-return index.

WIDE is the compute contract (gap-filling and spreads need aligned columns), LONG is the
storage contract.

No incremental machinery: the whole history is re-pulled and `replace`d each run (~40k
yfinance rows + 7 FRED calls), so a changed series definition can never leave stale rows
behind. Skipped entirely when the core level series already reach the previous business day.

Needs a free FRED key (https://fred.stlouisfed.org/docs/api/api_key.html -> FRED_API_KEY in
.env) and network for yfinance.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from fredapi import Fred

from src.context import Context
from src.data_store.schema import Tables
from src.constants.constants_price import (MACRO_PRICE_SERIES, MACRO_FRED_SERIES,
                                     MACRO_SPREAD_SERIES, MACRO_BOND_TR_SERIES,
                                     MACRO_BOND_MATURITY_YEARS, MACRO_CORE_LEVEL_SERIES,
                                     MACRO_ALL_SERIES)
from src.utils.ssl_setup import configure_corporate_ca
from src.data_extract.utils.common.run_manifest import record_run

# ORDERING, not a side-effect to tidy away: importing `download_ohlcv` pulls in yfinance,
# which imports curl_cffi at module load and FREEZES its CA bundle then -- so the combined
# corporate bundle must exist BEFORE that import. Idempotent; a no-op when main.py ran it.
configure_corporate_ca()
from src.data_extract.utils.prices.fetch_prices import download_ohlcv   # noqa: E402

_TRADING_DAYS = 252
MAX_GAP_DAYS = 7                 # fill sporadic daily gaps strictly shorter than this


def fill_short_gaps(df: pd.DataFrame, cols: list[str],
                    max_gap_days: int = MAX_GAP_DAYS) -> pd.DataFrame:
    """Fill sporadic interior NaN runs (holidays / one-off source misses / the yfinance and
    FRED calendars not lining up) in each of `cols` with the MEAN of the two bracketing
    observations, but ONLY when the gap spans fewer than `max_gap_days` calendar days.
    Longer outages and leading / trailing NaNs are left untouched -- a series that genuinely
    starts in 2003 must not be back-filled to 1995. `df` must have a DatetimeIndex."""
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


def build_bond_total_return(yield_pct: pd.Series,
                            maturity_years: int = MACRO_BOND_MATURITY_YEARS,
                            periods_per_year: int = _TRADING_DAYS) -> pd.Series:
    """Reconstruct a constant-maturity bond TOTAL-RETURN index from a par-yield series.

    Daily total return of a rolled constant-maturity par bond ~=
        carry (yield accrual)  -  modified_duration * delta_yield
    where the par-bond modified duration at yield y over M years is
        D = (1/y) * (1 - (1+y)^-M)                       (-> M as y -> 0).
    Yields are in PERCENT (e.g. 4.25). Returns an index normalized to 100 at the first valid
    observation. Leading NaNs are dropped; interior day-1 return is 0."""
    y = yield_pct.astype(float) / 100.0
    y_prev = y.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dur = np.where(y_prev > 0,
                       (1.0 / y_prev) * (1.0 - (1.0 + y_prev) ** (-maturity_years)),
                       float(maturity_years))
    carry = y_prev / periods_per_year
    price_ret = -pd.Series(dur, index=y.index) * (y - y_prev)
    daily_ret = (carry + price_ret)
    daily_ret = daily_ret[y.notna() & y_prev.notna()]        # drop warmup / NaN days
    if daily_ret.empty:
        return pd.Series(dtype=float, index=yield_pct.index)
    index = (1.0 + daily_ret).cumprod() * 100.0
    return index.reindex(yield_pct.index)


# --------------------------------------------------------------------------- #
# the three legs                                                              #
# --------------------------------------------------------------------------- #
def _fetch_price_leg(context: Context, since: pd.Timestamp,
                     until: pd.Timestamp) -> pd.DataFrame:
    """The yfinance legs -> date-indexed wide frame of CLOSES under their series names.

    Keeps only `close` (the "trim the volume" step). `trim_prelisting_bars` is deliberately
    NOT applied: it is an equity-listing heuristic, and `^VIX` is legitimately 100%
    zero-volume, so it would erase the whole series."""
    raw = download_ohlcv(list(MACRO_PRICE_SERIES), since, until,
                         desc="Downloading macro/market prices")
    if raw is None or raw.empty:
        context.log.warning("yfinance returned nothing for the macro price legs %s",
                            list(MACRO_PRICE_SERIES))
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    df = raw[["date", "ticker", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["ticker"] = df["ticker"].astype(str).map(MACRO_PRICE_SERIES)
    wide = (df.dropna(subset=["ticker"])
              .pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
              .sort_index())
    wide.columns.name = None
    wide.index.name = "date"

    missing = [n for n in MACRO_PRICE_SERIES.values() if n not in wide.columns]
    if missing:
        context.log.warning("macro price legs missing from the yfinance response: %s", missing)
    return wide


def _fetch_fred_leg(since: pd.Timestamp) -> pd.DataFrame:
    """The FRED LEVEL legs -> date-indexed wide frame under their series names."""
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    frame = {name: fred.get_series(sid, observation_start=since)
             for sid, name in MACRO_FRED_SERIES.items()}
    df = pd.DataFrame(frame)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def derive_series(wide: pd.DataFrame, context: Context | None = None) -> pd.DataFrame:
    """Add the derived series to an already gap-filled wide frame: the curve spreads, then
    the 10Y total-return index. Pure. A spread whose inputs are absent is skipped rather
    than emitted as all-NaN, so `prices_macro` never carries a series with no data."""
    out = wide.copy()
    for name, (minuend, subtrahend) in MACRO_SPREAD_SERIES.items():
        if minuend in out.columns and subtrahend in out.columns:
            out[name] = out[minuend] - out[subtrahend]
        elif context is not None:
            context.log.warning("spread '%s' skipped: needs %s - %s", name, minuend, subtrahend)
    if "yield_10y" in out.columns:
        out[MACRO_BOND_TR_SERIES] = build_bond_total_return(out["yield_10y"])
    elif context is not None:
        context.log.warning("'%s' skipped: yield_10y absent", MACRO_BOND_TR_SERIES)
    return out


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide (date x series) -> the stored long frame [date, ticker, close].

    The dropna is the point of the long layout: each leg starts when its source starts
    (fx 1999, gold 2000, breakeven 2003), which a wide table had to pad with NaN."""
    if wide is None or wide.empty:
        return pd.DataFrame(columns=["date", "ticker", "close"])
    long = (wide.rename_axis("date").reset_index()
                .melt(id_vars="date", var_name="ticker", value_name="close")
                .dropna(subset=["close"]))
    long["close"] = long["close"].astype(float)
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# freshness + entry point                                                     #
# --------------------------------------------------------------------------- #
def _is_up_to_date(context: Context) -> bool:
    """True only when EVERY core level series already reaches the previous business day.

    Per-series, not the table's overall max date: that was the bug both predecessors had --
    one same-day-publishing series marked the table current while the lagging level block
    stayed stale. Previous business day, not today, because these sources publish with a
    ~1 BDay lag. One grouped aggregate; never reads the rows."""
    last_by_series = context.store.max_date_by(Tables.prices_macro, "ticker", "date")
    if not last_by_series:
        return False
    core = [last_by_series[s] for s in MACRO_CORE_LEVEL_SERIES if s in last_by_series]
    if len(core) < len(MACRO_CORE_LEVEL_SERIES):
        return False                      # a core series has no rows at all
    last_expected = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    return bool(min(core) >= last_expected)


def build_macro_frame(context: Context, years_history: int) -> pd.DataFrame:
    """Fetch, align, gap-fill and derive -> the long frame ready to store. Separated from
    `fetch_macro` so the shape can be tested without touching the DB."""
    today = pd.Timestamp.today().normalize()
    since = today - pd.DateOffset(years=years_history)

    prices = _fetch_price_leg(context, since, today)
    fred = _fetch_fred_leg(since)

    # union of both calendars (yfinance trades US market days, FRED publishes on its own),
    # then mean-fill only the sporadic short gaps that misalignment creates
    wide = prices.join(fred, how="outer").sort_index()
    wide = fill_short_gaps(wide, list(wide.columns))
    wide = derive_series(wide, context)
    return to_long(wide)


def fetch_macro(context: Context, years_history: int) -> None:
    """Refresh `prices_macro`: every macro / market series, long, one source each.

    `years_history` is passed IN (resolved by StepExtractPrices.run / the CLI) rather than
    read off the config here -- the same contract as fetch_price_history / fetch_dividends,
    which keeps both windows visible at the one place that owns them."""
    if not os.getenv("FRED_API_KEY"):
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and add it to your .env file.")

    if _is_up_to_date(context):
        context.log.info("Macro series already up to date - skipping (DB table '%s')",
                         Tables.prices_macro)
        record_run(context, Tables.prices_macro, 0, 0)
        return

    long = build_macro_frame(context, years_history)
    if long.empty:
        raise RuntimeError(f"No macro rows built -> refusing to wipe '{Tables.prices_macro}'")

    # replace, not upsert: the full history is re-fetched every run, so a clean rewrite never
    # leaves stale rows behind when a series definition changes.
    context.store.replace(Tables.prices_macro, long)
    stored = sorted(long["ticker"].unique())
    context.log.info("Saved %d rows / %d series to DB table '%s' (%d-year window)",
                     len(long), len(stored), Tables.prices_macro, years_history)

    missing = [s for s in MACRO_ALL_SERIES if s not in stored]
    if missing:
        context.log.warning("series expected by the registry but not stored: %s", missing)

    record_run(context, Tables.prices_macro, len(stored), len(long))
