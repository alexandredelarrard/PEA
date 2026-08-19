"""
fetch_short_interest.py  (src/data_extract/utils/prices/fetch_short_interest.py)
---------------------------------------------------------------------------------
FINRA RegSHO CONSOLIDATED short-sale volume (`CNMSshvol` daily files, free, no
auth, full history from ~2009). This is short-selling PRESSURE (daily short vs
total volume) -- a proxy for short interest. Saved long
[date, ticker, short_volume, total_volume]; each day's file is disseminated the
next morning, so the aggregation step lags it one trading day (point-in-time).

Resume is on the table's GLOBAL max date, not per ticker: one RegSHO file covers
the whole market, so once day D is stored every ticker has D and a per-ticker
frontier would only re-download days already held -- one lagging symbol (index
churn, a renamed ticker) would drag the loop back over thousands of files on every
run. `fails_to_deliver` lives in its own table precisely to keep its semi-monthly,
~2-month-lagged files out of this max (see schema.py).
"""

from __future__ import annotations

import io
import time

import pandas as pd
import requests
import logging

from src.constants.constants import DATE_FORMAT_COMPACT
from src.context import Context
from src.data_store.schema import Tables
from src.data_extract.utils.common.run_manifest import record_run

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"
_HEADERS = {"User-Agent": "stock_pick_strat/1.0 (research; contact@example.com)"}

logger = logging.getLogger(__name__)

def _parse_regsho(text: str) -> pd.DataFrame:
    """Parse one CNMSshvol pipe-delimited file -> [date, ticker, short_volume,
    total_volume], aggregated per (date, ticker). Pure. Ignores the trailer."""

    if not text or "|" not in text:
        return pd.DataFrame(columns=["date", "ticker", "short_volume", "total_volume"])

    df = pd.read_csv(io.StringIO(text), sep="|")
    df = df[df.get("Symbol").notna()] if "Symbol" in df.columns else df.iloc[0:0]
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "short_volume", "total_volume"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce"),
        "ticker": df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False),
        "short_volume": pd.to_numeric(df["ShortVolume"], errors="coerce"),
        "total_volume": pd.to_numeric(df["TotalVolume"], errors="coerce"),
    }).dropna(subset=["date", "ticker"])
    return (out.groupby(["date", "ticker"], as_index=False)[["short_volume", "total_volume"]]
            .sum())


def _fetch_day(day: pd.Timestamp) -> str | None:
    """Network call for one date, isolated for mocking. None if no file."""
    r = requests.get(_URL.format(yyyymmdd=day.strftime(DATE_FORMAT_COMPACT)),
                     headers=_HEADERS, timeout=30)
    return r.text if r.status_code == 200 else None


def _resume_day(context: Context, years_history: int) -> pd.Timestamp:
    """First business day still to download: the day after the table's global max,
    or `years_history` back on a cold table. One scalar query, no table read."""
    stored_max = context.store.max_date(Tables.short_interest)
    if stored_max is None:
        return pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    return stored_max + pd.Timedelta(days=1)


def fetch_short_interest(context: Context, tickers: list[str],
                         years_history: int = 15, pause: float = 0.05) -> None:
    """Download the RegSHO daily short-volume files not yet stored, keep only
    `tickers`, and upsert them into `short_interest`. Returns the new rows."""

    today = pd.Timestamp.today().normalize()
    days = pd.bdate_range(_resume_day(context, years_history), today)
    logger.info(f"Fetching {len(days)} RegSHO day-file(s) for {len(tickers)} tickers")

    frames: list[pd.DataFrame] = []
    for day in days:
        try:
            text = _fetch_day(day)
        except Exception as e:                     # one bad day must not abort the run
            logger.error(f"RegSHO {day.date()} failed: {e}")
            continue
        if not text:
            continue
        df_day = _parse_regsho(text)
        df_day = df_day[df_day["ticker"].isin(tickers)]
        if not df_day.empty:
            frames.append(df_day)
        time.sleep(pause)

    df_short = (pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=["date", "ticker", "short_volume", "total_volume"]))

    # upsert the freshly-downloaded days; the DB merges on (ticker, date)
    context.store.save(Tables.short_interest, df_short)
    logger.info(f"Saved {len(df_short)} new short-volume rows to DB table "
                f"'{Tables.short_interest}'")
    record_run(context, Tables.short_interest, len(tickers), len(df_short))
