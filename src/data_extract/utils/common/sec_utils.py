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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.data_store.schema import Tables
from src.context import Context
from src.data_extract.utils.common.registrant import load_registrants

_MIN_INTERVAL = 0.11          # ~9 req/sec, safely under SEC's 10/sec limit
_DEFAULT_TIMEOUT = 30         # seconds; avoid a hung socket stalling a worker
_rate_lock = threading.Lock()
_next_slot = [0.0]            # monotonic time of the next allowed request start


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


def sec_get(context: Context, url: str, **kwargs) -> requests.Response:
    """Rate-limited GET on `context.sec_session` (the required SEC User-Agent header is
    pre-set on it). Safe to call from multiple threads: the session's connection pool is
    thread-safe and only request *initiation* is serialized, by `_reserve_slot`."""
    kwargs.setdefault("timeout", _DEFAULT_TIMEOUT)
    _reserve_slot()
    resp = context.sec_session.get(url, **kwargs)
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------- #
# Incremental-extraction helpers                                              #
# --------------------------------------------------------------------------- #
def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def existing_filings(context: Context, table) -> frozenset[str]:
    """Accession numbers already stored in a filing table -- the dedup set every
    per-filing fetcher (13D, 8-K, DEF 14A, filing text) uses to skip a filing it has
    already extracted. Empty when the table does not exist yet, so a first run
    fetches full history.

    Deliberately accession-only, NOT a per-ticker max-filing-date cutoff: that was
    tried and reverted, because it never re-checks a date range already scanned --
    a filing missed by a prior bug, or one that posts to EDGAR out of date order,
    stays missing forever. Each run lists a ticker's whole window and relies solely
    on this set to avoid re-work."""
    return frozenset(str(a) for a in context.store.distinct(table, "accession_number"))


def bulk_ingested_quarters(store, table: str) -> set[str]:
    """Distinct source-zip `quarter` tags already stored in a bulk table -> the
    set of quarters an incremental re-run can SKIP (a past quarter's data set is
    final once the quarter ends). Empty when the table doesn't exist yet."""
    return {str(q) for q in store.distinct(table, "quarter")}


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


#: The `sp500_tickers` projection every SEC fetcher resolves its universe through. Module-level
#: so a test fixture standing in for that table can be built FROM it -- a fixture that pinned its
#: own column list passed while production read a column the fixture never wrote, and the
#: resulting `KeyError` surfaced only as an unrelated-looking driver failure.
CIK_MAPPING_COLS: tuple[str, ...] = ("ticker", "cik", "name", "sector",
                                     "industry_group", "sub_industry")


def load_cik_mapping(context: Context, tickers: list[str] | None = None) -> pd.DataFrame:
    """Ticker -> CIK (+ name / GICS) resolution for the SEC EDGAR fetchers, filtered
    server-side to `tickers` when given. `company_name` is aliased from `name` for
    callers that log it.

    Single source of truth is `sp500_tickers` (built by fetch_prices), which already
    carries `cik` alongside `name` / `sector` / `industry_group` / `sub_industry`.
    A separate `cik_mapping` table rebuilt from SEC's company_tickers.json was dropped:
    it duplicated `sp500_tickers` AND mismapped active tickers (e.g. XOM -> a non-filing
    "ExxonMobil Holdings Corp" shell).
    """
    df = context.store.load(Tables.sp500_tickers, columns=list(CIK_MAPPING_COLS),
                            where={"ticker": list(tickers)} if tickers is not None else None)
    
    # SEC URLs need the 10-digit zero-padded CIK
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)
    if "company_name" not in df.columns and "name" in df.columns:
        df["company_name"] = df["name"]
    return df

def cik_to_ticker(cikmap: pd.DataFrame, *, config_dir: str | None = None) -> dict[str, str]:
    """CIK -> ticker, INCLUDING every predecessor CIK in the registrant register.

    ⚠ THE BULK DATA SETS HAD NO CUTOVER CONCEPT AT ALL, and this one dict is why. The SEC's
    insider, financial-notes and financial-statement data sets are keyed by CIK, and these
    fetchers map a row to a ticker through this map alone. With one CIK per ticker, every row
    a PREDECESSOR filed resolved to nothing and was dropped -- silently, since a row for an
    unknown CIK is indistinguishable from a row for a company outside the universe.

    Measured 2026-09-09, the two cases where the trading symbol moved too, so even the
    insider fetcher's symbol-first path could not save it:

        GOOGL  insider_transactions starts 2015-10-08   (boundary 2015-10-02, predecessor GOOG)
        VTRS   insider_transactions starts 2020-11-16   (predecessor traded MYL)
        APA    insider_transactions starts 2006-01-04   -- saved ONLY because APA never moved

    ⚠ THE MAP ALONE IS NOT ENOUGH FOR A `SPLIT` TABLE. `notes_*` and `pension_facts` are
    consolidating, so their callers must additionally drop rows whose `filed` date falls
    outside the matched segment's window -- otherwise they re-acquire the Apache-subsidiary
    problem the register was built to prevent. `insider_transactions` is UNION (Forms 3/4/5
    are events) and takes no date filter. See `registrant.FORM_POLICY`.

    Returns `dict[str, str]` exactly as before, so no call site changes shape.
    """
    if cikmap.empty or "ticker" not in cikmap.columns:
        return {}
    out = {str(c): str(t).upper() for c, t in zip(cikmap["cik"], cikmap["ticker"])}
    universe = set(out.values())
    for ticker, entry in load_registrants(config_dir).items():
        if ticker.upper() not in universe:
            continue                       # a register entry for a ticker this run is not
        for cik in entry.all_ciks():       # walking adds nothing and would widen the map
            out.setdefault(cik, ticker.upper())
    return out