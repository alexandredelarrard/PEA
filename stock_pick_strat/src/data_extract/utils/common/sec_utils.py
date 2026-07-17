"""
Shared helpers for talking to SEC EDGAR (free, no API key, but requires a
descriptive User-Agent and respectful rate limiting — SEC's fair-access
policy asks for <=10 requests/second; we stay just under it).

https://www.sec.gov/os/webmaster-faq#developers

The rate limiter is THREAD-SAFE: request *initiation* is serialized and spaced
by `_MIN_INTERVAL` across all threads (so the global rate never exceeds SEC's
limit), while the network transfer happens outside the lock so downloads from a
ThreadPoolExecutor overlap. This is what lets the EDGAR fetchers parallelize.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.context import Context

_MIN_INTERVAL = 0.11          # ~9 req/sec, safely under SEC's 10/sec limit
_DEFAULT_TIMEOUT = 30         # seconds; avoid a hung socket stalling a worker
_rate_lock = threading.Lock()
_next_slot = [0.0]            # monotonic time of the next allowed request start


def _sec_headers() -> dict[str, str]:
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. SEC EDGAR blocks requests without a "
            "descriptive User-Agent (name + email). Add to your .env file, e.g.\n"
            '  SEC_USER_AGENT="Your Name your.email@example.com"\n'
            "See https://www.sec.gov/os/webmaster-faq#developers"
        )
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def _reserve_slot() -> None:
    """Reserve the next evenly-spaced request slot (thread-safe). The wait
    happens OUTSIDE the lock so concurrent transfers overlap while request
    starts stay spaced by `_MIN_INTERVAL`."""
    with _rate_lock:
        start = max(time.monotonic(), _next_slot[0])
        _next_slot[0] = start + _MIN_INTERVAL
    delay = start - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def sec_get(url: str, **kwargs) -> requests.Response:
    """Rate-limited GET with the required SEC User-Agent header. Safe to call
    from multiple threads."""
    kwargs.setdefault("timeout", _DEFAULT_TIMEOUT)
    _reserve_slot()
    resp = requests.get(url, headers=_sec_headers(), **kwargs)
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------- #
# Incremental-extraction meta sidecar (D = last filing date already parsed)    #
# --------------------------------------------------------------------------- #
def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _meta_path(parquet_path: Path) -> Path:
    return parquet_path.with_name(parquet_path.stem + "_meta.json")


def load_extract_meta(parquet_path: Path) -> dict | None:
    """Sidecar metadata for an incremental extract (or None if never built)."""
    p = _meta_path(parquet_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_extract_meta(parquet_path: Path, last_filing_date: str | None,
                      ticker_count: int, universe_size: int) -> None:
    """Record today's build so a same-day re-run can skip, and the max filing
    date parsed (`D`) so the next run only fetches filings after it."""
    _meta_path(parquet_path).write_text(
        json.dumps({
            "last_built": today_iso(),
            "last_filing_date": last_filing_date,
            "ticker_count": int(ticker_count),
            "universe_size": int(universe_size),
        }, indent=2),
        encoding="utf-8",
    )


def build_cik_mapping(context: Context, sp500_tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch SEC's full ticker->CIK mapping and filter to our S&P 500 universe.
    Source: https://www.sec.gov/files/company_tickers.json (free, no key).
    """
    resp = sec_get("https://www.sec.gov/files/company_tickers.json")
    raw = resp.json()  # dict of {index: {cik_str, ticker, title}}
    df = pd.DataFrame(raw.values())
    df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)
    df = df.rename(columns={"cik_str": "cik", "title": "company_name"})

    if sp500_tickers:
        df = df[df["ticker"].isin(sp500_tickers)]

    context.store.save("cik_mapping", df)
    print(f"Saved CIK mapping for {len(df)} tickers to DB table 'cik_mapping'")
    return df


def load_cik_mapping(context: Context) -> pd.DataFrame:
    df = context.store.load("cik_mapping")
    if df.empty:
        tickers = context.store.load("sp500_tickers", columns=["ticker"])["ticker"].tolist()
        return build_cik_mapping(context, tickers)
    # CIK is stored numerically but SEC URLs need the 10-digit zero-padded form
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    return df
