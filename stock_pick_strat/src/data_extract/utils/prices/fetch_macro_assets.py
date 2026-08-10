"""
fetch_macro_assets.py  (src/data_extract/utils/prices/fetch_macro_assets.py)
----------------------------------------------------------------------------
Pull the LONG-HISTORY multi-asset ALLOCATION series and write them to the
`macro_asset_prices` DB table (PK: date, no ticker). These feed the risk-parity + trend
asset-allocation sleeve and its since-~2000 backtest — a SEPARATE, deeper pull than the
`macro` feature table (scoped by `data_extract.macro_asset_years_history`, default 31y).

HYBRID source (verified July 2026): FRED's API no longer serves a broad daily S&P (its
`SP500` is license-truncated to ~10y) or ANY gold series (the London fixes were removed
~2025). So the legs are split by where each is actually available with deep history:
  * FRED     -> yield_10y (DGS10), cash_rate (DTB3), fx_usdeur (DEXUSEU)
  * yfinance -> equity_tr (SPY, auto-adjusted = total-return proxy, since 1993)
                gold      (GC=F COMEX front future, since 2000)

Columns (see src/constants/constants.py):
  * equity_tr    -- SPY total-return proxy (yfinance).
  * gold         -- COMEX gold front-future close, USD/oz (yfinance).
  * yield_10y    -- 10Y constant-maturity Treasury yield (FRED), kept for transparency.
  * bond_10y_tr  -- 10Y TOTAL-RETURN index reconstructed from `yield_10y`
                    (carry + duration*Δyield); the return series you actually backtest.
  * cash_rate    -- 3-month T-bill secondary-market rate (FRED), the cash leg.
  * fx_usdeur    -- USD per EUR (FRED); NaN before the euro (1999).

Sporadic daily gaps (holidays / one-off misses / the two calendars not lining up) are
mean-filled with the shared `fill_short_gaps` helper before the bond-TR reconstruction.
The whole history is re-fetched and `replace`d each run (cheap), and skipped when the
core level series already reach the previous business day.

Needs a free FRED key (https://fred.stlouisfed.org/docs/api/api_key.html -> FRED_API_KEY
in .env) and network for yfinance. On a corporate TLS-proxy, the shared SSL setup
(src/utils/ssl_setup.configure_corporate_ca) must run before yfinance imports curl_cffi;
main.py does this, and this module calls it defensively before its lazy yfinance import.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from fredapi import Fred

from src.data_store.schema import Tables
from src.context import Context
from src.constants.constants import (MACRO_ASSET_FRED_SERIES, MACRO_ASSET_YF_SERIES, MACRO_ASSET_BOND_TR_COLUMN, MACRO_ASSET_BOND_MATURITY_YEARS, MACRO_ASSET_CORE_LEVEL_COLUMNS, DATE_FORMAT)
from src.utils.ssl_setup import configure_corporate_ca
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.fundamentals.fetch_macro import fill_short_gaps

# ORDERING, not a side-effect to tidy away: yfinance imports curl_cffi at module load and
# freezes its CA bundle then, so the combined corporate bundle must exist BEFORE that import.
# Doing it here (rather than inside the fetch function) is what lets `import yfinance` sit at
# the top of the file like it does in fetch_prices / fetch_dividends / fetch_earnings_surprises.
# Idempotent -- a no-op when main.py already ran it.
configure_corporate_ca()
import yfinance as yf                                           # noqa: E402 (after CA setup)

_TRADING_DAYS = 252


def build_bond_total_return(yield_pct: pd.Series,
                            maturity_years: int = MACRO_ASSET_BOND_MATURITY_YEARS,
                            periods_per_year: int = _TRADING_DAYS) -> pd.Series:
    """Reconstruct a constant-maturity bond TOTAL-RETURN index from a par-yield series.

    Daily total return of a rolled constant-maturity par bond ≈
        carry (yield accrual)  -  modified_duration * Δyield
    where the par-bond modified duration at yield y over M years is
        D = (1/y) * (1 - (1+y)^-M)                       (-> M as y -> 0).
    Yields are in PERCENT (e.g. 4.25). Returns an index normalized to 100 at the
    first valid observation. Leading NaNs are dropped; interior day-1 return is 0."""
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


def _fetch_fred(start: pd.Timestamp) -> pd.DataFrame:
    """FRED rates / cash / FX legs -> date-indexed frame with the mapped column names."""
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    frame = {name: fred.get_series(sid, observation_start=start)
             for sid, name in MACRO_ASSET_FRED_SERIES.items()}
    df = pd.DataFrame(frame)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def _fetch_yfinance(start: pd.Timestamp, context: Context) -> pd.DataFrame:
    """yfinance equity + gold legs -> date-indexed frame with the mapped column names.
    Auto-adjusted daily close (equity total-return proxy)."""
    cols: dict[str, pd.Series] = {}
    for sym, name in MACRO_ASSET_YF_SERIES.items():
        try:
            raw = yf.download(sym, start=start.strftime(DATE_FORMAT), auto_adjust=True,
                              progress=False, threads=False)
            if raw is None or raw.empty:
                context.log.warning("yfinance returned no data for %s -> '%s' empty", sym, name)
                continue
            close = raw["Close"]
            s = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            s.index = pd.to_datetime(s.index).tz_localize(None)
            cols[name] = s.astype(float)
        except Exception as exc:                                # noqa: BLE001
            context.log.warning("yfinance download failed for %s (%s) -> '%s' empty", sym, exc, name)
    df = pd.DataFrame(cols)
    df.index.name = "date"
    return df


def _macro_assets_up_to_date(context: Context) -> bool:
    """True only when the CORE daily level columns already reach the previous business
    day (their sources lag ~1 BDay). Missing table -> not fresh."""
    existing = context.store.load(Tables.macro_asset_prices, optional=True)
    if existing is None or "date" not in existing.columns:
        return False
    existing["date"] = pd.to_datetime(existing["date"])
    last_expected = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)
    core = [c for c in MACRO_ASSET_CORE_LEVEL_COLUMNS if c in existing.columns]
    if not core:
        return existing["date"].max() >= last_expected
    core_last = min(existing.loc[existing[c].notna(), "date"].max() for c in core)
    return bool(pd.notna(core_last) and core_last >= last_expected)


def _refresh_macro_assets(context: Context) -> None:
    """Download FRED + yfinance legs, align, fill short daily gaps, reconstruct the bond
    TR index, and rewrite the `macro_asset_prices` table."""
    years = context.config.data_extract.macro_asset_years_history
    start = pd.Timestamp.today() - pd.DateOffset(years=years + 1)

    fred = _fetch_fred(start)
    yfin = _fetch_yfinance(start, context)
    # union of both business-day calendars; sort; keep as a clean daily frame
    assets = fred.join(yfin, how="outer").sort_index()

    # sporadic daily gaps (holidays / one-off misses / calendars not lining up) ->
    # mean of the bracketing observed days (only gaps shorter than a week)
    assets = fill_short_gaps(assets, list(assets.columns))

    # reconstruct the 10Y bond TOTAL-RETURN index from the (gap-filled) yield
    assets[MACRO_ASSET_BOND_TR_COLUMN] = build_bond_total_return(assets["yield_10y"])

    assets = assets.reset_index()
    context.store.replace(Tables.macro_asset_prices, assets)
    context.log.info("Saved %d rows of macro-asset series to DB table '%s' (%d-year history; "
                     "FRED rates/cash/fx + yfinance equity/gold)",
                     len(assets), Tables.macro_asset_prices, years)
    record_run(context, Tables.macro_asset_prices, 0, len(assets))


def fetch_macro_assets(context: Context) -> None:
    if not os.getenv("FRED_API_KEY"):
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and add it to your .env file.")
    if _macro_assets_up_to_date(context):
        context.log.info("Macro-asset series already up to date - skipping (DB table '%s')",
                         Tables.macro_asset_prices)
        record_run(context, Tables.macro_asset_prices, 0, 0)
        return
    _refresh_macro_assets(context)
