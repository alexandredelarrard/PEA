"""
fetch_short_interest.py (src/data_extract/utils/institutionals/fetch_short_interest.py)
---------------------------------------------------------------------------------
FINRA RegSHO CONSOLIDATED short-sale volume (`CNMSshvol` daily files, free, no auth). This is
short-selling PRESSURE (daily short vs total volume) -- a proxy for short interest, NOT reported
short interest. Saved long [date, ticker, short_volume, total_volume]; each day's file is
disseminated the next morning, so the aggregation step lags it one trading day (point-in-time).

Missing the Lit exchange short volumes from NYSE / Nasdaq and CBOE equities.

⚠ THE CDN KEEPS A ROLLING ~8-YEAR WINDOW. There is no deep history to backfill, and this is a
RETENTION limit at the source, not a gap in this fetcher -- so no future reader should spend a
day trying. Probed at the URL pattern below on 2026-09-08:

    20100415 403 · 20130415 403 · 20160415 403 · 20170103 403 · 20180112 403 · 20180712 403
    20180731 403 · 20180801 200 · 20180814 200 · 20190701 200 · ... · 20260901 200

Binary-searched to the day: last 403 is 2018-07-31, first 200 is 2018-08-01 -- a boundary on a
month start, ~8.10 years before the probe date, which is why it MOVES FORWARD. Rows already
stored below it cannot be re-fetched if lost.

The stored `min(date)` is 2017-12-29, which is NOT the history start: 20171229 is a lone file
that survives outside the window (probed 200 while every other 2017 and early-2018 date returns
403). Treating it as a floor would claim eight months of coverage that do not exist.

NAMING: the table is `sec_short_interest` but it holds short-sale VOLUME. The misnomer is a live
table with consumers, so it is fixed at the feature level (`ic_shortvol_*`), not here.
"""

from __future__ import annotations

import io
import time
import pandas as pd
import requests
import logging
from tqdm import tqdm

from src.constants.constants import DATE_FORMAT_COMPACT, _HEADERS
from src.context import Context
from src.data_store.schema import Tables
from src.data_extract.utils.common.run_manifest import record_run

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"

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


def _resume_day(context: Context, years_history: int = 15) -> pd.Timestamp:
    """The first day to download: the day after the GLOBAL stored max, or the full
    `years_history` window on a cold table.

    Global and not per-ticker on purpose. A RegSHO day-file carries every symbol at once, so
    one lagging ticker would drag the whole download back to its own last date and re-fetch
    days already stored for all the others.
    """
    today = pd.Timestamp.today().normalize()
    stored_max = context.store.max_date(Tables.short_interest)
    if stored_max is None:
        return today - pd.DateOffset(years=years_history)
    return stored_max + pd.Timedelta(days=1)


def fetch_short_interest(context: Context, tickers: list[str],
                         years_history: int = 15, pause: float = 0.05) -> None:
    """Download the RegSHO daily short-volume files not yet stored, keep only
    `tickers`, and upsert them into `sec_short_interest`."""

    today = pd.Timestamp.today().normalize()
    days = pd.bdate_range(_resume_day(context, years_history), today)
    logger.info(f"Fetching {len(days)} RegSHO day-file(s) for {len(tickers)} tickers")

    frames: list[pd.DataFrame] = []
    for day in tqdm(days, "short_interest OffExchange - fetch RegSHO"):
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
