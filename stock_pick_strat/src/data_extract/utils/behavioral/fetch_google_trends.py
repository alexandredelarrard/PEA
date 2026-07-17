"""
fetch_google_trends.py  (src/data_extract/utils/fetch_google_trends.py)
-----------------------------------------------------------------------
Google Trends search-interest per company (via the optional `pytrends` library).
A RETAIL-ATTENTION proxy. Saved long [date, ticker, search_interest].

CAVEATS (research-grade signal):
  * Google Trends returns interest normalized 0-100 WITHIN the requested window,
    and the series is REVISED over time -> it is not perfectly point-in-time.
    Use it as an attention proxy, not a precise as-of value; the attention_features
    builder only uses within-name relative spikes, which are the robust part.
  * Weekly resolution and heavy rate-limiting -> we pause between queries and skip
    failures. `pytrends` is optional; if not installed, extraction is skipped.

Network/library access is isolated in `_interest_over_time`; the parser
(`_df_to_long`) is pure and unit-tested.
"""

from __future__ import annotations

import random
import time
import pandas as pd
from tqdm import tqdm
import re

from src.context import Context
from src.data_extract.utils.common.rate_limit import call_with_retries
from pytrends.request import TrendReq


# Anti-429 (Google throttles scrapers hard): rotate a realistic desktop User-Agent
# per request, send full browser-like headers with a Referer, use a FRESH TrendReq
# (fresh session/cookie) each call so no sticky cookie fingerprint builds up, add
# pytrends' native retry/backoff, and jitter the delay between queries. See
# cloro.dev / pytrends issues #243, #535.
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


def _random_header() -> dict:
    """Browser-like headers with a rotated User-Agent + Referer (harder to flag)."""
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                   "image/avif,image/webp,*/*;q=0.8"),
        "Referer": "https://trends.google.com/trends/explore",
        "Connection": "keep-alive",
    }

def _df_to_long(interest: pd.DataFrame, keyword: str, ticker: str) -> pd.DataFrame:
    """pytrends interest_over_time() frame -> [date, ticker, search_interest]. Pure."""
    if interest is None or interest.empty or keyword not in interest.columns:
        return pd.DataFrame(columns=["date", "ticker", "search_interest"])
    s = interest[keyword]
    if "isPartial" in interest.columns:            # drop the still-forming last bucket
        s = s[~interest["isPartial"].astype(bool)]
    if s.empty:
        return pd.DataFrame(columns=["date", "ticker", "search_interest"])
    return pd.DataFrame({
        "date": pd.to_datetime(s.index).tz_localize(None).normalize(),
        "ticker": ticker,
        "search_interest": s.to_numpy(dtype="float64"),
    })


def _interest_over_time(keyword: str, timeframe: str):
    """Network/lib call, isolated for mocking. Fresh TrendReq (fresh session/cookie)
    + rotated header + pytrends native retry/backoff each call, to dodge 429s."""
    pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5,
                  requests_args={"headers": _random_header()})
    pt.build_payload([keyword], timeframe=timeframe)
    return pt.interest_over_time()


def _load_existing(context: Context):
    df = context.store.load("google_trends")
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def fetch_google_trends(context: Context, tickers: list[str] | None = None,
                        timeframe: str = "all", pause: float = 1.0,
                        refetch_window_days: int = 7) -> pd.DataFrame:
    """Download search interest per company name; cache to parquet.

    * Skips cleanly if `pytrends` is not installed.
    * INCREMENTAL: a ticker whose cached data is within `refetch_window_days` of
      today is skipped (Trends is weekly and re-normalizes the whole window, so we
      skip current names rather than request a sub-range); only ex-weeks after the
      cached max are appended for the rest.
    * RATE LIMIT (429): instead of skipping the ticker, WAIT and retry with
      exponential backoff (>=3 retries) via call_with_retries.
    """
    names = context.store.load("sp500_tickers")
    names['name'] = names['name'].apply(lambda x: re.sub(r"\s*\([^)]*\)", "", x).strip())

    if tickers is not None:
        names = names[names["ticker"].isin(tickers)]

    existing = _load_existing(context)
    last_by_ticker = ({} if existing is None
                      else existing.groupby("ticker")["date"].max().to_dict())
    today = pd.Timestamp.today().normalize()

    frames, skipped = [], 0
    for _, row in tqdm(list(names.iterrows()), desc="Google Trends"):
        tkr, keyword = row["ticker"], str(row["name"])
        last = last_by_ticker.get(tkr)
        if last is not None and (today - last).days <= refetch_window_days:
            skipped += 1                        # already current -> skip
            continue
        try:
            # wait + retry on 429 (>=3 retries) rather than dropping the ticker
            interest = call_with_retries(
                lambda: _interest_over_time(keyword, timeframe),
                retries=3, base_wait=60.0, label=f"trends {tkr}")
        except Exception as e:
            print(f"Trends fetch failed for {tkr} ({keyword}) after retries: {e}")
            continue
        long = _df_to_long(interest, keyword, tkr)
        if last is not None and not long.empty:
            long = long[long["date"] > last]     # append only new weeks
        if not long.empty:
            frames.append(long)
        time.sleep(pause + random.uniform(2.0, 8.0))   # jitter -> less bot-like
    print(f"Google Trends: {skipped}/{len(names)} tickers already current (skipped).")

    parts = [df for df in (existing, *frames) if df is not None and not df.empty]
    if not parts:
        print("No Google Trends data available.")
        return existing if existing is not None else pd.DataFrame(
            columns=["date", "ticker", "search_interest"])
    out = (pd.concat(parts, ignore_index=True)
           .drop_duplicates(subset=["ticker", "date"], keep="last")
           .sort_values(["ticker", "date"]).reset_index(drop=True))
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not new.empty:
        context.store.save("google_trends", new)
    print(f"Saved {len(new)} new Google Trends rows to DB table 'google_trends'")
    return out
