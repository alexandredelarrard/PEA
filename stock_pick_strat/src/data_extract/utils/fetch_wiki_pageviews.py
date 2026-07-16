"""
fetch_wiki_pageviews.py  (src/data_extract/utils/fetch_wiki_pageviews.py)
-------------------------------------------------------------------------
Daily Wikipedia pageviews per company (Wikimedia REST API, free, daily history
from 2015-07). A proxy for RETAIL ATTENTION. Saved long [date, ticker, pageviews]
keyed on the view date -> point-in-time.

The ticker->article mapping is best-effort from the company `name` in the tickers
CSV; unmatched/failed articles are skipped. Network is isolated in
`_fetch_article`; parsing (`_json_to_long`) and title cleaning
(`_company_to_article`) are pure and unit-tested.
"""
from __future__ import annotations

import re
import time

import pandas as pd
import requests
from tqdm import tqdm

from src.context import Context

_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia/all-access/user/{article}/daily/{start}/{end}")
_HEADERS = {"User-Agent": "stock_pick_strat/1.0 (research; contact@example.com)"}
_SUFFIXES = re.compile(
    r"\b(inc|inc\.|incorporated|corp|corp\.|corporation|company|co|co\.|ltd|"
    r"plc|holdings|group|the|class [abc]|&)\b", re.IGNORECASE)


def _company_to_article(name: str) -> str:
    """Best-effort Wikipedia article title from a company name: strip common
    corporate suffixes/punctuation, collapse spaces, use underscores."""
    s = _SUFFIXES.sub(" ", str(name))
    s = re.sub(r"[.,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "_")


def _json_to_long(items: list[dict], ticker: str) -> pd.DataFrame:
    """Wikimedia 'items' list -> [date, ticker, pageviews]. Pure."""
    if not items:
        return pd.DataFrame(columns=["date", "ticker", "pageviews"])
    rows = [{"date": pd.to_datetime(str(it["timestamp"])[:8], format="%Y%m%d"),
             "ticker": ticker, "pageviews": float(it.get("views", 0))}
            for it in items if it.get("timestamp")]
    return pd.DataFrame(rows)


def _fetch_article(article: str, start: str, end: str) -> list[dict]:
    """Network call, isolated for mocking. Returns the 'items' list ([] on miss)."""
    url = _API.format(article=requests.utils.quote(article, safe=""), start=start, end=end)
    r = requests.get(url, headers=_HEADERS, timeout=30)
    if r.status_code != 200:
        return []
    return r.json().get("items", [])


def fetch_wiki_pageviews(context: Context, tickers: list[str] | None = None,
                         years_history: int = 10, pause: float = 0.1) -> pd.DataFrame:
    """Download daily pageviews for the S&P 500 names and cache to parquet."""
    path = context.paths["WIKI_PAGEVIEWS_PATH"]
    names = pd.read_csv(context.paths["TICKERS_PATH"])
    if tickers is not None:
        names = names[names["ticker"].isin(tickers)]
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)).strftime("%Y%m%d")
    end = pd.Timestamp.today().normalize().strftime("%Y%m%d")

    frames = []
    for _, row in tqdm(list(names.iterrows()), desc="Wikipedia pageviews"):
        article = _company_to_article(row["name"])
        try:
            long = _json_to_long(_fetch_article(article, start, end), row["ticker"])
        except Exception as e:
            print(f"Wiki fetch failed for {row['ticker']} ({article}): {e}")
            continue
        if not long.empty:
            frames.append(long)
        time.sleep(pause)

    if not frames:
        print("No Wikipedia pageview data available.")
        return pd.DataFrame(columns=["date", "ticker", "pageviews"])
    out = (pd.concat(frames, ignore_index=True)
           .drop_duplicates(subset=["ticker", "date"], keep="last")
           .sort_values(["ticker", "date"]).reset_index(drop=True))
    out.to_parquet(path, index=False)
    print(f"Saved {len(out)} Wikipedia pageview rows to {path}")
    return out
