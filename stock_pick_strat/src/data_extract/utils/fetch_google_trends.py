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

import time

import pandas as pd
from tqdm import tqdm
from pytrends.request import TrendReq

from src.context import Context


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
    """Network/lib call, isolated for mocking. Requires the optional pytrends dep."""
    pt = TrendReq(hl="en-US", tz=0)
    pt.build_payload([keyword], timeframe=timeframe)
    return pt.interest_over_time()


def fetch_google_trends(context: Context, tickers: list[str] | None = None,
                        timeframe: str = "today 5-y", pause: float = 1.0) -> pd.DataFrame:
    """Download search interest per company name; cache to parquet. Skips cleanly
    if `pytrends` is not installed."""
    path = context.paths["GOOGLE_TRENDS_PATH"]
    names = pd.read_csv(context.paths["TICKERS_PATH"])
    if tickers is not None:
        names = names[names["ticker"].isin(tickers)]

    frames = []
    for _, row in tqdm(list(names.iterrows()), desc="Google Trends"):
        keyword = str(row["name"])
        try:
            long = _df_to_long(_interest_over_time(keyword, timeframe), keyword, row["ticker"])
            time.sleep(pause*3)
        except Exception as e:
            print(f"Trends fetch failed for {row['ticker']} ({keyword}): {e}")
            time.sleep(pause * 3)                  # back off on rate-limit
            continue
        if not long.empty:
            frames.append(long)
        time.sleep(pause)

    if not frames:
        print("No Google Trends data available.")
        return pd.DataFrame(columns=["date", "ticker", "search_interest"])
    out = (pd.concat(frames, ignore_index=True)
           .drop_duplicates(subset=["ticker", "date"], keep="last")
           .sort_values(["ticker", "date"]).reset_index(drop=True))
    out.to_parquet(path, index=False)
    print(f"Saved {len(out)} Google Trends rows to {path}")
    return out
