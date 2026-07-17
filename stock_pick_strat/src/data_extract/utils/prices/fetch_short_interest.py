"""
fetch_short_interest.py  (src/data_extract/utils/fetch_short_interest.py)
-------------------------------------------------------------------------
FINRA RegSHO CONSOLIDATED short-sale volume (`CNMSshvol` daily files, free, no
auth, full history from ~2009). This is short-selling PRESSURE (daily short vs
total volume) -- a proxy for short interest. Saved long
[date, ticker, short_volume, total_volume]; each day's file is disseminated the
next morning, so the aggregation step lags it one trading day (point-in-time).

NOTE: true short-INTEREST positions (bi-monthly open shorts, the classic anomaly)
now require FINRA's OAuth Query API; if you later fetch those into the same
parquet with a `short_interest`/`avg_daily_volume` schema, the feature builder
picks them up too. Network is isolated in `_fetch_day`; `_parse_regsho` is pure.
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

from src.context import Context

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"
_HEADERS = {"User-Agent": "stock_pick_strat/1.0 (research; contact@example.com)"}


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
    r = requests.get(_URL.format(yyyymmdd=day.strftime("%Y%m%d")), headers=_HEADERS, timeout=30)
    return r.text if r.status_code == 200 else None


def _load_existing(path):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def fetch_short_interest(context: Context, tickers: list[str] | None = None,
                         years_history: int = 10, pause: float = 0.05) -> pd.DataFrame:
    """Download RegSHO daily short-volume files incrementally; cache to parquet."""
    path = context.paths["SHORT_INTEREST_PATH"]
    existing = _load_existing(path)
    start = (existing["date"].max() + pd.Timedelta(days=1)) if existing is not None \
        else pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    days = pd.bdate_range(start, pd.Timestamp.today().normalize())

    universe = set(tickers) if tickers is not None else None
    frames = []
    for d in days:
        try:
            text = _fetch_day(d)
        except Exception as e:
            print(f"RegSHO {d.date()} failed: {e}")
            continue
        if not text:
            continue
        day_df = _parse_regsho(text)
        if universe is not None and not day_df.empty:
            day_df = day_df[day_df["ticker"].isin(universe)]
        if not day_df.empty:
            frames.append(day_df)
        time.sleep(pause)

    parts = [df for df in (existing, *frames) if df is not None and not df.empty]
    if not parts:
        print("No short-volume data available.")
        return pd.DataFrame(columns=["date", "ticker", "short_volume", "total_volume"])
    out = (pd.concat(parts, ignore_index=True)
           .drop_duplicates(["ticker", "date"], keep="last")
           .sort_values(["ticker", "date"]).reset_index(drop=True))
    out.to_parquet(path, index=False)
    print(f"Saved {len(out)} short-volume rows to {path}")
    return out
