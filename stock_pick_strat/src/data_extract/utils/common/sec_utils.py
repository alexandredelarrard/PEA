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
from sqlalchemy import text

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


def meta_path(artifact_path: Path) -> Path:
    """`<artifact>_meta.json` next to the artifact it describes."""
    return artifact_path.with_name(artifact_path.stem + "_meta.json")


def load_extract_meta(parquet_path: Path) -> dict | None:
    """Sidecar metadata for an incremental extract (or None if never built)."""
    p = meta_path(parquet_path)
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
    meta_path(parquet_path).write_text(
        json.dumps({
            "last_built": today_iso(),
            "last_filing_date": last_filing_date,
            "ticker_count": int(ticker_count),
            "universe_size": int(universe_size),
        }, indent=2),
        encoding="utf-8",
    )


def existing_filings(context: Context, table: str) -> set[str]:
    """Accession numbers already stored in a filing table -- the dedup set every
    per-filing fetcher (13D, 8-K, DEF 14A edgar, ...) uses to skip a filing it has
    already extracted. Returns empty when the table does not exist yet, so a first
    run fetches full history.

    Deliberately accession-only, NOT a per-ticker max-filing-date cutoff: an earlier
    version also returned a `{ticker: max(filing_date)}` dict so each ticker's listing
    window could start from its own last-seen date, but that silently never re-checks
    any date range already scanned -- a filing missed by a prior bug, or one that
    posts to EDGAR out of date order, stays missing forever. Every run now lists each
    ticker's FULL `years_history` window and relies solely on this accession set to
    avoid re-work, matching `fetch_def14a_llm.py`'s gap-filling convention."""
    try:
        df = context.store.load(table, columns=["accession_number"])
    except Exception:                                   # noqa: BLE001 (table not created yet)
        return set()
    if df is None or df.empty:
        return set()
    return set(df["accession_number"].dropna().astype(str))


def bulk_ingested_quarters(store, table: str) -> set[str]:
    """Distinct source-zip `quarter` tags already stored in a bulk table -> the
    set of quarters an incremental re-run can SKIP (a past quarter's data set is
    final once the quarter ends). Empty when the table doesn't exist yet."""
    if not store.exists(table):
        return set()
    
    with store.engine.connect() as c:
        return set(pd.read_sql(text(f'SELECT DISTINCT quarter FROM "{table}"'), c)
                   ["quarter"].dropna())


def load_processed_universe(cache_dir: Path, table: str) -> set[str]:
    """The ticker universe a bulk table was last built against (sidecar JSON). Used
    to decide whether cached zips must be re-parsed to back-fill NEW tickers.
    Comparing to the processed set (not to the tickers that happened to file) is
    what makes the re-parse converge instead of firing every run."""
    p = cache_dir / f"{table}_universe.json"
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")).get("universe", []))
    except Exception:
        return set()


def save_processed_universe(cache_dir: Path, table: str, universe: set[str]) -> None:
    (cache_dir / f"{table}_universe.json").write_text(
        json.dumps({"universe": sorted(universe), "saved": today_iso()}),
        encoding="utf-8")


def load_cik_mapping(context: Context) -> pd.DataFrame:
    """Ticker -> CIK (+ name / GICS) resolution for the SEC EDGAR fetchers.

    Single source of truth is `sp500_tickers` (built by fetch_prices), which already
    carries `cik` alongside `name` / `sector` / `industry_group` / `sub_industry`.
    `company_name` is exposed (aliased from `name`) for callers that log it.

    Formerly a separate `cik_mapping` table rebuilt from SEC's company_tickers.json;
    dropped because it merely duplicated `sp500_tickers` AND its CIK source mismapped
    active tickers (e.g. XOM -> a non-filing "ExxonMobil Holdings Corp" shell), while
    sp500_tickers already held the correct CIKs.
    """
    df = context.store.load("sp500_tickers")
    if df.empty or "cik" not in df.columns:
        return df
    df = df.copy()
    # SEC URLs need the 10-digit zero-padded CIK
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    if "company_name" not in df.columns and "name" in df.columns:
        df["company_name"] = df["name"]
    return df
