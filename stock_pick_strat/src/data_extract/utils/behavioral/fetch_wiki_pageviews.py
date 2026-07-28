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

import pandas as pd
import requests
from tqdm import tqdm

from src.constants.constants import DATE_FORMAT_COMPACT
from src.context import Context
from src.utils import polite_http as ph          # per-host paced inter-request sleep
from src.utils.crawler import Crawler
from src.data_extract.utils.common.incremental import load_existing

_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia/all-access/user/{article}/daily/{start}/{end}")
_HEADERS = {"User-Agent": "stock_pick_strat/1.0 (research; contact@example.com)"}

# shared crawler for Wikimedia: impersonate=False (a FRIENDLY API — send our descriptive contact
# UA, don't spoof a browser) + IP rotation over PEA_SCRAPE_PROXIES on a block + fast retry.
_WIKI_CRAWLER: Crawler | None = None


def _wiki_crawler() -> Crawler:
    global _WIKI_CRAWLER
    if _WIKI_CRAWLER is None:
        _WIKI_CRAWLER = Crawler(retries=5, backoff=1.0, timeout=30, impersonate=False)
    return _WIKI_CRAWLER
_SUFFIXES = re.compile(
    r"\b(inc|inc\.|incorporated|corp|corp\.|corporation|company|co|co\.|ltd|"
    r"plc|holdings|group|the|class [abc]|&)\b", re.IGNORECASE)


def _company_to_article(name: str) -> str:
    """Best-effort Wikipedia article title from a company name: strip common
    corporate suffixes/punctuation, collapse spaces, use underscores. Kept as the
    FALLBACK when the search resolver (below) finds nothing."""
    s = _SUFFIXES.sub(" ", str(name))
    s = re.sub(r"[.,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "_")


# The S&P "Security" names carry list artifacts that break the title guess AND a raw
# search: "Coca-Cola Company (The)", "Alphabet Inc. (Class A)", "Lilly (Eli)".
_THE_SUFFIX = re.compile(r"^(.*?)\s*\(\s*the\s*\)\s*$", re.IGNORECASE)
_CLASS_SUFFIX = re.compile(r"\s*\(\s*class\s+[abc]\s*\)\s*$", re.IGNORECASE)
_PAREN_FIRST = re.compile(r"^([^()]+?)\s*\(([^)]+)\)\s*$")
_SEARCH_API = "https://en.wikipedia.org/w/api.php"


def _clean_company_name(name: str) -> str:
    """Normalize an S&P 'Security' name into a natural search query: 'X (The)' -> 'The X',
    drop '(Class A/B/C)', reorder 'Surname (First)' -> 'First Surname' (e.g. 'Lilly (Eli)'
    -> 'Eli Lilly'). Leaves ordinary names ('Deere & Company', 'S&P Global') unchanged."""
    s = re.sub(r"\s+", " ", str(name)).strip()
    m = _THE_SUFFIX.match(s)
    if m:
        return f"The {m.group(1).strip()}"
    s = _CLASS_SUFFIX.sub("", s).strip()
    m = _PAREN_FIRST.match(s)
    if m and m.group(2).strip().lower() != "the":
        return f"{m.group(2).strip()} {m.group(1).strip()}"
    return s


def _wiki_search(query: str) -> str | None:
    """Top main-namespace Wikipedia article title for `query`, or None. Isolated for
    mocking. Searching the full (cleaned) COMPANY name biases the hit to the company
    article (e.g. 'The Coca-Cola Company') over the brand ('Coca-Cola')."""
    # IP-rotating crawler with the descriptive contact UA (Wikimedia is a friendly API).
    data = _wiki_crawler().get_json(_SEARCH_API, headers=_HEADERS, params={
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": 1, "srnamespace": 0, "format": "json"})
    if not data:
        return None
    hits = data.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def _resolve_wiki_article(name: str, search_fn=_wiki_search) -> str:
    """Company name -> Wikipedia article title via search on the cleaned name (handles
    the 'The'/'Class'/'Surname (First)' artifacts + redirects that the naive heuristic
    misses). Fallback when search yields nothing is the CLEANED name itself — usually
    the real article ('The Coca-Cola Company', 'Alphabet Inc.', 'The Home Depot') and
    strictly better than the aggressive suffix-strip, which drops 'Company'/'Inc'/'The'."""
    cleaned = _clean_company_name(name)
    try:
        title = search_fn(cleaned)
    except Exception:                                            # noqa: BLE001
        title = None
    return (title or cleaned).replace(" ", "_")


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
    data = _wiki_crawler().get_json(url, headers=_HEADERS)     # IP-rotating crawler, contact UA
    return data.get("items", []) if data else []


def fetch_wiki_pageviews(context: Context, tickers: list[str] | None = None,
                         years_history: int = 10, pause: float = 1.0,
                         refetch_window_days: int = 2) -> pd.DataFrame:
    """Download daily pageviews for the S&P 500 names and upsert to the DB.

    Incremental (point-in-time): the last-extracted day is read PER TICKER from the stored
    `wiki_pageviews` table (max date per ticker), and we request only days AFTER it (the
    Wikimedia API takes an explicit start/end), so a re-run downloads ONLY the missing tail.
    A ticker whose latest stored day is within `refetch_window_days` of today is CURRENT and
    makes NO request at all -- pageviews publish with a ~1-2 day lag, so without this tolerance
    the freshest stored day never reaches `end_ts` (yesterday) and EVERY run fires ~500 empty
    API calls (the "always redone" bug).

    Pace: `pause` defaults to 1.0s so `sleep_pace` (base + 0.1-0.7 jitter, x any post-429
    per-host slowdown) never issues more than ~1 request/second to the Wikimedia API."""
    names = context.store.load("sp500_tickers")
    if tickers is not None:
        names = names[names["ticker"].isin(tickers)]

    existing = load_existing(context, "wiki_pageviews")
    last_by_ticker = ({} if existing is None
                      else existing.groupby("ticker")["date"].max().to_dict())
    today = pd.Timestamp.today().normalize()
    default_start = today - pd.DateOffset(years=years_history)
    # pageviews for a day are available the next day; stop at yesterday
    end_ts = today - pd.Timedelta(days=1)
    end = end_ts.strftime(DATE_FORMAT_COMPACT)

    frames, skipped = [], 0
    for _, row in tqdm(list(names.iterrows()), desc="Wikipedia pageviews"):
        last = last_by_ticker.get(row["ticker"])
        # already current within the publication lag -> no API call at all
        if last is not None and (today - last).days <= refetch_window_days:
            skipped += 1
            continue
        start_ts = (last + pd.Timedelta(days=1)) if last is not None else default_start
        if start_ts > end_ts:                       # nothing new to request
            skipped += 1
            continue
        # resolve the real Wikipedia article via search (handles 'Deere & Company' ->
        # John Deere, 'Alphabet Inc. (Class A)' -> Alphabet Inc., 'Home Depot (The)' ->
        # The Home Depot, ...); the naive suffix-strip is only the fallback.
        article = _resolve_wiki_article(row["name"])
        try:
            long = _json_to_long(
                _fetch_article(article, start_ts.strftime(DATE_FORMAT_COMPACT), end),
                row["ticker"])
        except Exception as e:
            print(f"Wiki fetch failed for {row['ticker']} ({article}): {e}")
            continue
        if not long.empty:
            frames.append(long)
        ph.sleep_pace(pause, _API)                       # per-host paced (honours 429 slowdown)
    print(f"Wikipedia: {skipped}/{len(names)} tickers already current (skipped).")

    parts = [df for df in (existing, *frames) if df is not None and not df.empty]
    if not parts:
        print("No Wikipedia pageview data available.")
        return existing if existing is not None else pd.DataFrame(columns=["date", "ticker", "pageviews"])
    out = (pd.concat(parts, ignore_index=True)
           .drop_duplicates(subset=["ticker", "date"], keep="last")
           .sort_values(["ticker", "date"]).reset_index(drop=True))
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not new.empty:
        context.store.save("wiki_pageviews", new)
    print(f"Saved {len(new)} new Wikipedia pageview rows to DB table 'wiki_pageviews'")
    return out
