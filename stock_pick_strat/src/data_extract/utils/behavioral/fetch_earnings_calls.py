"""
fetch_earnings_calls.py  (src/data_extract/utils/behavioral/fetch_earnings_calls.py)
------------------------------------------------------------------------------------
FREE earnings-call transcripts from The Motley Fool (fool.com) -- full text, no API
key, no paywall, good S&P 500 coverage. Deliberately NOT FMP (paid). Three stages,
all incremental:

  1. build_transcript_index()  -> crawl MF's paginated transcript index, parse
     (ticker, quarter, fiscal year, call date, url) straight out of each transcript
     URL slug, keep the universe, and persist the WHOLE link list to a big JSON in
     the repo cache: data/call_transcripts/transcript_index.json .
  2. download_transcripts()    -> read the JSON, fetch each not-yet-cached
     transcript's raw HTML and cache it to
     data/call_transcripts/{TICKER}/{YYYY}Q{Q}.html (skip already-downloaded).
  3. ingest_earnings_calls()   -> parse each cached transcript into the high-signal
     SECTIONS funds analyse -- management prepared remarks, the analyst Q&A, and the
     call participants -- and upsert to the `earnings_call_sections` DB table
     (ticker, quarter, as_of, tag, text, url) for later sentiment / embedding.

Transcript URL shape (ticker/quarter/year are IN the slug, no page parse needed):
  /earnings/call-transcripts/{YYYY}/{MM}/{DD}/{company-slug}-{ticker}-q{Q}-{FY}-earnings-call-transcript/
"""
from __future__ import annotations

import json
import logging
import random
import re
from tqdm import tqdm

import pandas as pd
from bs4 import BeautifulSoup

from src.constants.constants import (
    EARNINGS_CALL_REPORT_GRACE_DAYS,
    EARNINGS_CALL_REQUEST_PAUSE,
    EARNINGS_CALL_SECTIONS_TABLE,
    MOTLEY_FOOL_BASE_URL,
    MOTLEY_FOOL_TRANSCRIPT_INDEX_URL,
)
from src.context import Context

from src.data_extract.utils.behavioral.fetch_hf_transcripts import download_hf_parquet
from src.data_extract.utils.behavioral.fetch_hf_transcripts import ingest_hf_transcripts
from src.data_extract.utils.behavioral.fetch_roic_transcripts import fetch_roic_transcripts
from src.data_extract.utils.behavioral.utils_behavior import (
    _cache_dir, 
    _index_path,
    _load_index,
    _get,
    _sleep_pace)
from src.data_extract.utils.behavioral.utils_missing_quarters import (
    _quarter_index,
    _parse_quarter,
    _missing_for,
    _released_quarter_idx_by_ticker,
    _db_quarters_by_ticker,
    _since_floor_index,
    _latest_expected_quarter_index,
    _index_to_quarter,
    _quarter_index
)
from src.data_extract.utils.behavioral.utils_split_qa import split_prepared_qa

logger = logging.getLogger(__name__)

# one shared crawler per process: headless, no cookies/JS/images, rolling browser fingerprint, and
# MOVING IPs over PEA_SCRAPE_PROXIES on a Cloudflare block. Shared so a rotated (good) IP persists
# across the whole crawl instead of resetting to a flagged one on every request.

_BASE = MOTLEY_FOOL_BASE_URL
_INDEX = MOTLEY_FOOL_TRANSCRIPT_INDEX_URL
_TABLE = EARNINGS_CALL_SECTIONS_TABLE

# transcript-link path + the (year, month, day, slug, quarter, fiscal-year) groups
_LINK_RE = re.compile(
    r"/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/(.+?)-q([1-4])-(\d{4})-earnings-call-transcript")
_HREF_RE = re.compile(r'href="(/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/[^"?#]+)"')


# --------------------------------------------------------------------------- #
# Stage 1: discover -> big JSON of transcript links (PURE parse unit-tested)     #
# --------------------------------------------------------------------------- #
def _universe_slug_map(universe: list[str]) -> dict[str, str]:
    """ticker -> its lowercase MF slug form (BRK-B -> 'brk-b'), for suffix matching."""
    return {str(t): str(t).lower() for t in universe}


def _parse_link(href: str, slug_map: dict[str, str]) -> dict | None:
    """One transcript href -> {ticker, quarter, call_date, url}. The ticker is the
    universe ticker whose slug form is the trailing token(s) of the URL slug (so
    'berkshire-hathaway-brk-b' resolves to BRK-B, not 'b'). None if not in universe."""
    m = _LINK_RE.search(href)
    if not m:
        return None
    yr, mo, dy, slug, q, fy = m.groups()
    tkr = next((t for t, s in slug_map.items() if slug == s or slug.endswith("-" + s)), None)
    if tkr is None:
        return None
    return {"ticker": tkr, "quarter": f"{fy}Q{q}", "call_date": f"{yr}-{mo}-{dy}",
            "url": _BASE + href.rstrip("/") + "/"}


def _links_on_page(html: str, slug_map: dict[str, str]) -> list[dict]:
    recs, seen = [], set()
    for href in _HREF_RE.findall(html or ""):
        rec = _parse_link(href, slug_map)
        if rec and rec["url"] not in seen:
            seen.add(rec["url"])
            recs.append(rec)
    return recs


def _page_call_dates(html: str) -> list[str]:
    """Every transcript's call date (YYYY-MM-DD) on the page — universe or not. The MF
    feed is globally reverse-chronological, so the NEWEST date on a page gauges how far
    back the crawl has paged (used for the history-horizon stop)."""
    out = []
    for href in _HREF_RE.findall(html or ""):
        m = _LINK_RE.search(href)
        if m:
            out.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return out


def _page_converged(recs: list[dict], added: int) -> bool:
    """Whether a page counts toward the convergence stop. TRUE only when the page HAS
    universe transcripts but they are ALL already indexed (an incremental re-run catching
    up to what it saw last time). A page with NO universe names is just small-cap noise in
    the all-companies feed and must NOT stop the crawl — conflating the two is the bug that
    truncated the index to ~16 links."""
    return bool(recs) and added == 0


def build_transcript_index(context: Context, tickers: list[str] | None = None,
                           max_pages: int = 490, stop_after_empty: int = 4,
                           pause: float = 0.6, history_years: float = 6.0) -> dict[str, dict]:
    """Crawl the MF transcript index (a global, all-companies, newest-first feed) and
    MERGE every universe transcript link into the big JSON.

    STOP conditions (both bounded so a first build goes DEEP, a re-run returns fast):
      * convergence — `stop_after_empty` consecutive pages that HAVE universe transcripts
        but all already indexed (see `_page_converged`). Crucially, a page with NO universe
        names does NOT count: those are small-caps in the global feed, and treating them as
        "empty" was the bug that stopped the crawl at ~16 links.
      * history horizon — the feed has paged back past `history_years` (the newest call on
        the page is older than the cutoff), so we have enough per-ticker history.
      * `max_pages` / end-of-feed (a page that won't load) as hard safety caps.

    `tickers` restricts the kept universe (None = all)."""
    universe = list(context.store.load("sp500_tickers", columns=["ticker"])["ticker"])
    if tickers is not None:                          # scope to a subset (e.g. a test run)
        keep = set(tickers)
        universe = [t for t in universe if t in keep]
    slug_map = _universe_slug_map(universe)
    path = _index_path(context)
    index = _load_index(path)
    before = len(index)
    min_date = ((pd.Timestamp.today().normalize() - pd.DateOffset(years=history_years))
                .strftime("%Y-%m-%d"))

    empty_streak = 0
    for page in tqdm(range(1, max_pages + 1), "transcript index urls"):
        # MF paginates at /earnings-call-transcripts/page/N/ (page 1 = the base URL);
        # the old ?page=N query param is ignored by the site (always served page 1).
        html = _get(_INDEX if page == 1 else f"{_INDEX}page/{page}/")
        if not html:
            logger.warning("MF index page %d did not load (blocked or end of feed) -> stop "
                           "at %d links", page, len(index))
            break
        recs = _links_on_page(html, slug_map)
        added = 0
        for r in recs:
            if r["url"] not in index:
                index[r["url"]] = r
                added += 1
        newest = max(_page_call_dates(html), default=None)
        empty_streak = empty_streak + 1 if _page_converged(recs, added) else 0
        logger.info("MF index page %d: %d universe links (%d new), newest %s, "
                    "converge-streak %d/%d (index total %d)", page, len(recs), added,
                    newest, empty_streak, stop_after_empty, len(index))
        if empty_streak >= stop_after_empty:
            logger.info("Index converged (%d consecutive pages of already-seen universe "
                        "links) -> stop", stop_after_empty)
            break
        if newest and newest < min_date:
            logger.info("Reached %.0fy history horizon (newest call on page %s < %s) -> stop",
                        history_years, newest, min_date)
            break
        _sleep_pace(pause)

    path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.warning("Transcript index: %d links across %d tickers (%d new this run) -> %s",
                   len(index), len({r["ticker"] for r in index.values()}),
                   len(index) - before, path)
    return index


# --------------------------------------------------------------------------- #
# Stage 1b: TARGETED discovery via per-ticker quote pages (no global-feed cap)  #
# --------------------------------------------------------------------------- #
# Transcript path as it appears on a QUOTE page's raw text/JSON (not always href="...").
_QUOTE_PATH_RE = re.compile(
    r"/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/[a-z0-9-]+?-q[1-4]-\d{4}-earnings-call-transcript")


def _quote_links(html: str, slug_map: dict[str, str]) -> list[dict]:
    """Transcript links on a company QUOTE page. Their URLs live in the page's raw
    text/JSON, so match the transcript PATH directly (vs the index crawl's href regex),
    then resolve each via `_parse_link`."""
    recs, seen = [], set()
    for m in _QUOTE_PATH_RE.finditer((html or "").lower()):
        rec = _parse_link(m.group(0), slug_map)
        if rec and rec["url"] not in seen:
            seen.add(rec["url"])
            recs.append(rec)
    return recs


def _quote_page(ticker: str, exchanges: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Fetch a ticker's MF quote page, trying each exchange until one resolves (the wrong
    exchange 404s -- quietly, since it's an expected probe). `/quote/{exchange}/{ticker}/`
    needs no company slug (MF redirects the slug form to it). Returns (html, exchange)."""
    for exch in exchanges:
        html = _get(f"{_BASE}/quote/{exch}/{ticker.lower()}/", log_missing=False)
        if html:
            return html, exch
    return None, None


def build_transcript_index_by_ticker(
    context: Context, tickers: list[str] | None = None, since: str = "2025-01-01",
    exchanges: tuple[str, ...] = ("nasdaq", "nyse"), pause: float = EARNINGS_CALL_REQUEST_PAUSE,
    grace_days: int = EARNINGS_CALL_REPORT_GRACE_DAYS,
) -> dict[str, dict]:
    """TARGETED, HF-aware discovery of the RECENT gap that the HuggingFace backbone does NOT cover.

    For every ticker it computes the PRECISE set of quarters still missing, and ONLY then (if that
    set is non-empty) spends ONE request on its MF quote page `fool.com/quote/{exchange}/{ticker}/`
    (which lists that name's recent transcript URLs with the exact date+slug baked in). This is the
    key to avoiding the 429: most names are already complete once HF + prior downloads are counted,
    so their quote page is never fetched.

    Per-ticker gap logic (bullets, in order):
      1. HF horizon — read the backbone's LATEST quarter for the ticker; the fool gap starts at the
         quarter AFTER it (or the `since` date floor when HF has nothing for the name).
      2. required   — every quarter from that gap-start up to the latest expected quarter today
         (calendar quarter of today - `grace_days`, so an unreported quarter isn't demanded).
      3. have        — quarters already on DISK (data/call_transcripts/{ticker}/*.html) ∪ already in
         the DB sections table (any source) ∪ already in the JSON index.
      4. missing = required - have. Empty -> SKIP the ticker (no request). Else fetch the quote page
         and keep only links whose quarter is in the gap (>= gap-start) and not already have.

    Slow by design (`pause` defaults to the polite EARNINGS_CALL_REQUEST_PAUSE) and resume-safe: the
    JSON index is saved after every ticker that adds links, so a 429/interrupt loses no progress.
    `tickers` restricts the universe (None = all)."""
    from src.data_extract.utils.behavioral.fetch_hf_transcripts import hf_latest_quarter_by_ticker

    universe = list(context.store.load("sp500_tickers", columns=["ticker"])["ticker"])
    if tickers is not None:
        keep = set(tickers)
        universe = [t for t in universe if t in keep]
    slug_map = _universe_slug_map(universe)
    path = _index_path(context)
    index = _load_index(path)
    cache_dir = _cache_dir(context)
    since = str(since)

    end_idx = _latest_expected_quarter_index(grace_days)   # newest quarter expected to exist today
    floor_idx = _since_floor_index(since)                  # gap start when HF has nothing for a name
    hf_latest = hf_latest_quarter_by_ticker(context, tickers=universe)   # {ticker: (year, quarter)}
    have_db = _db_quarters_by_ticker(context)
    released = _released_quarter_idx_by_ticker(context)   # latest ACTUALLY-reported quarter per ticker
    have_json: dict[str, set] = {}
    for r in index.values():
        have_json.setdefault(str(r["ticker"]), set()).add(str(r["quarter"]))

    def _missing(tk: str) -> set[str]:
        """Quarters the fool quote page should still supply for `tk` (shared gap logic)."""
        return _missing_for(tk, hf_latest, floor_idx, end_idx, cache_dir, have_db, have_json, released)

    # process tickers in RANDOM order (not universe/alphabetical): spreads the fool.com load and
    # keeps an interrupted / throttled run from always dying on the same tail names.
    order = list(universe)
    random.shuffle(order)

    added_total, missing_page, skipped = 0, [], 0
    for tkr in tqdm(order, "quote-page transcript urls"):
        need = _missing(tkr)
        if not need:                          # already complete -> NO request (the 429 fix)
            skipped += 1
            continue
        gap_min = min(_quarter_index(*_parse_quarter(q)) for q in need)
        html, exch = _quote_page(tkr, exchanges)
        if html is None:
            missing_page.append(tkr)
            _sleep_pace(pause)                # still pace: a 404 probe cost 2 requests
            continue
        added = 0
        for r in _quote_links(html, slug_map):
            pq = _parse_quarter(r["quarter"])
            if (r["ticker"] != tkr or pq is None or _quarter_index(*pq) < gap_min
                    or r["url"] in index):
                continue
            index[r["url"]] = r
            have_json.setdefault(tkr, set()).add(r["quarter"])
            added += 1
        added_total += added
        logger.info("MF quote %s (%s): need %d quarters (%s..), %d new links (index total %d)",
                    tkr, exch, len(need), _index_to_quarter(gap_min), added, len(index))
        if added:                             # DYNAMIC save -> progress survives a 429 / interrupt
            path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
        _sleep_pace(pause)

    path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.warning("Quote-page discovery: +%d new links | skipped %d already-complete (HF+disk+DB) "
                   "| %d/%d had no quote page -> %s", added_total, skipped,
                   len(missing_page), len(universe), path)
    if missing_page:
        logger.info("No MF quote page (exchange miss / not covered) for %d tickers: %s",
                    len(missing_page), missing_page[:40])
    return index


# --------------------------------------------------------------------------- #
# Stage 2 + 3: download the text, parse the high-signal sections                #
# --------------------------------------------------------------------------- #
def _is_caps_header(line: str) -> bool:
    """A standalone ALL-CAPS heading line (MF markers: 'CALL PARTICIPANTS', 'TAKEAWAYS')."""
    s = line.strip().rstrip(":")
    return (2 <= len(s) <= 45 and s == s.upper() and any(c.isalpha() for c in s)
            and len(s.split()) <= 5)

def parse_transcript_sections(html: str) -> dict[str, str]:
    """Motley Fool transcript HTML -> sections. Extracts the nested transcript body, adds
    the caps-headed `participants` block, then defers to `split_prepared_qa` for
    full/prepared_remarks/qa."""
    soup = BeautifulSoup(html, "html.parser")
    div = (soup.find("div", class_=lambda c: c and "transcript-content" in c)
           or soup.find("div", class_=lambda c: c and "article-body" in c))
    if div is None:
        return {}
    text = div.get_text("\n", strip=True)
    if len(text) < 200:
        return {}
    out = split_prepared_qa(text)

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    parts, grab = [], False
    for ln in lines:                                     # participants = caps-bounded block
        if _is_caps_header(ln):
            grab = "participant" in ln.lower()
            continue
        if grab:
            parts.append(ln)
    if parts:
        out["participants"] = "\n".join(parts)
    return out


def download_transcripts(context: Context, tickers: list[str] | None = None,
                         pause: float = EARNINGS_CALL_REQUEST_PAUSE, limit: int | None = None) -> int:
    """Download each indexed transcript's HTML and cache the raw HTML to disk
    (data/call_transcripts/{ticker}/{quarter}.html). Skips already-downloaded files.
    Returns the number newly downloaded. `tickers` restricts to a subset (None = all);
    `limit` bounds a test run."""
    cache_dir = _cache_dir(context)
    index = _load_index(_index_path(context))
    keep = set(tickers) if tickers is not None else None
    todo = [r for r in index.values()
            if (keep is None or r["ticker"] in keep)
            and not (cache_dir / r["ticker"] / f"{r['quarter']}.html").exists()]
    # RANDOM order (not index/alphabetical): spreads the load and, with `limit`, samples a random
    # subset rather than always the same head of the index.
    random.shuffle(todo)
    if limit is not None:
        todo = todo[:limit]
        
    n = 0
    for rec in tqdm(todo, "EC download"):
        html = _get(rec["url"])
        if not html or "call-transcripts" not in html.lower():
            continue
        out = cache_dir / rec["ticker"] / f"{rec['quarter']}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        n += 1
        _sleep_pace(pause)
    logger.info("Downloaded %d new transcripts (%d already cached) -> %s",
                   n, len(index) - n, cache_dir)
    return n


def _existing_section_keys(context: Context) -> set[tuple[str, str]]:
    """(ticker, quarter) already present in `earnings_call_sections` (from ANY source). Lets the MF
    ingest SKIP transcripts already parsed instead of re-reading + re-parsing every cached HTML each
    run. Empty set when the table is missing / unreadable (-> full ingest)."""
    try:
        df = context.store.load(_TABLE, columns=["ticker", "quarter"])
        if df is None or df.empty:
            return set()
        return set(map(tuple, df[["ticker", "quarter"]].astype(str).drop_duplicates().to_numpy()))
    except Exception:                                   # noqa: BLE001 (table not created yet)
        return set()


def ingest_earnings_calls(context: Context, tickers: list[str] | None = None,
                          force: bool = False) -> int:
    """Parse cached transcript HTML into sections and upsert to `earnings_call_sections`
    (ticker, quarter, as_of, tag, text, url). Returns rows upserted.

    INCREMENTAL: a (ticker, quarter) already in the table is SKIPPED (no HTML read / no
    BeautifulSoup parse) — so a re-run on a full cache is instant instead of silently re-parsing
    thousands of files (the "nothing happens" stall). `force=True` re-parses everything; `tickers`
    restricts to a subset (None = every cached transcript)."""
    cache_dir = _cache_dir(context)
    index = {(r["ticker"], r["quarter"]): r for r in _load_index(_index_path(context)).values()}
    keep = set(tickers) if tickers is not None else None
    existing = set() if force else _existing_section_keys(context)

    rows: list[dict] = []
    parsed = skipped = 0
    for html_path in tqdm(sorted(cache_dir.glob("*/*.html")), "EC ingestion db"):
        ticker, quarter = html_path.parent.name, html_path.stem
        if keep is not None and ticker not in keep:
            continue
        if (ticker, quarter) in existing:               # already ingested -> skip (incremental)
            skipped += 1
            continue
        rec = index.get((ticker, quarter), {})
        sections = parse_transcript_sections(html_path.read_text(encoding="utf-8", errors="replace"))
        added = False
        for tag, text in sections.items():
            if len(text) < 40:                          # skip empty / stub sections
                continue
            rows.append({"ticker": ticker, "quarter": quarter, "tag": tag,
                         "as_of": rec.get("call_date"), "url": rec.get("url"), "text": text})
            added = True
        if added:
            parsed += 1
            existing.add((ticker, quarter))             # de-dup within this run too

    if not rows:
        logger.info("MF ingest: nothing new — %d cached transcript(s) already ingested.", skipped)
        return 0
    df = pd.DataFrame(rows)
    saved = context.store.save(_TABLE, df)
    logger.info("MF ingest: +%d sections from %d NEW transcripts (%d cached skipped, %d tickers) -> '%s'",
                saved, parsed, skipped, df["ticker"].nunique(), _TABLE)
    return saved


def download_earnings_calls(context: Context, tickers: list[str] | None = None,
                            limit: int | None = None, include_hf: bool = True,
                            recent_since: str = "2025-01-01", use_global_crawl: bool = False,
                            mf_history_years: float = 2.0, use_roic: bool = True) -> None:
    """DOWNLOAD / extract stage. Recent-gap sources are tried in PRIORITY order:
      1. HuggingFace backbone parquet cached (deep history ~2005->2025Q1; disk).
      2. ROIC AI (`use_roic`): fill each ticker's MISSING recent quarters -> sections in the DB. This
         is the primary recent source; because it writes to the DB, step 3's fool gap logic then
         skips whatever Roic already covered.
      3. Motley Fool LAST RESORT — per-ticker quote-page discovery of the STILL-missing quarters
         (+ optional legacy global crawl), then download each indexed transcript's HTML to disk.
    Incremental throughout (HF parquet + MF HTML skipped when on disk; Roic/MF gap excludes anything
    already stored). `tickers` restricts the subset; `limit` bounds the MF download for a test;
    `include_hf=False` skips the backbone; `use_roic=False` skips the Roic layer."""
    if include_hf:
        # deferred import: fetch_hf_transcripts imports helpers from THIS module
        download_hf_parquet(context)
    if use_roic:
        # deferred import: fetch_roic_transcripts imports helpers from THIS module
        fetch_roic_transcripts(context, tickers=tickers, since=recent_since)
    build_transcript_index_by_ticker(context, tickers=tickers, since=recent_since)
    if use_global_crawl:
        build_transcript_index(context, tickers=tickers, history_years=mf_history_years)
    download_transcripts(context, tickers=tickers, limit=limit)


def ingest_all_earnings_calls(context: Context, tickers: list[str] | None = None,
                              include_hf: bool = True, force: bool = False) -> int:
    """INGEST stage — parse the already-downloaded transcripts into `earnings_call_sections`:
      * the cached HuggingFace parquet (`ingest_hf_transcripts`; skips when the backbone is already
        ingested), and
      * the cached Motley Fool HTML (`ingest_earnings_calls`; skips (ticker,quarter) already ingested).
    Runs after `download_earnings_calls`. Both stages are INCREMENTAL + deduped by (ticker, quarter),
    so a re-run with nothing new returns fast. `force=True` re-ingests both regardless."""
    saved = 0
    if include_hf:
        saved += ingest_hf_transcripts(context, tickers=tickers, force=force)   # cached parquet -> sections
    saved += ingest_earnings_calls(context, tickers=tickers, force=force)       # cached MF HTML -> sections
    return saved


def fetch_earnings_calls(context: Context, tickers: list[str] | None = None,
                         limit: int | None = None, include_hf: bool = True,
                         recent_since: str = "2025-01-01", use_global_crawl: bool = False,
                         mf_history_years: float = 2.0) -> int:
    """Full transcript pipeline = download + ingest (kept for main.py / tests / the monolithic
    extraction step). The Airflow DAG runs the two stages as SEPARATE tasks
    (download_earnings_calls -> ingest_all_earnings_calls) instead."""
    download_earnings_calls(context, tickers=tickers, limit=limit, include_hf=include_hf,
                            recent_since=recent_since, use_global_crawl=use_global_crawl,
                            mf_history_years=mf_history_years)
    return ingest_all_earnings_calls(context, tickers=tickers, include_hf=include_hf)
