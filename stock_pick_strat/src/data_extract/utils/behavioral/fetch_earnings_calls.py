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
import re
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from bs4 import BeautifulSoup

from src.constants.constants import (
    EARNINGS_CALL_CACHE_DIR,
    EARNINGS_CALL_SECTIONS_TABLE,
    MOTLEY_FOOL_BASE_URL,
    MOTLEY_FOOL_TRANSCRIPT_INDEX_URL,
)
from src.context import Context
from src.utils import polite_http as ph

logger = logging.getLogger(__name__)

_BASE = MOTLEY_FOOL_BASE_URL
_INDEX = MOTLEY_FOOL_TRANSCRIPT_INDEX_URL
_TABLE = EARNINGS_CALL_SECTIONS_TABLE

# transcript-link path + the (year, month, day, slug, quarter, fiscal-year) groups
_LINK_RE = re.compile(
    r"/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/(.+?)-q([1-4])-(\d{4})-earnings-call-transcript")
_HREF_RE = re.compile(r'href="(/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/[^"?#]+)"')


# --------------------------------------------------------------------------- #
# IO helpers                                                                    #
# --------------------------------------------------------------------------- #
def _cache_dir(context: Context) -> Path:
    d = context.paths["DATA_STORE"] / EARNINGS_CALL_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(context: Context) -> Path:
    return _cache_dir(context) / "transcript_index.json"


def _load_index(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get(url: str, timeout: int = 30, retries: int = 4, backoff: float = 3.0,
         log_missing: bool = True) -> str | None:
    """MF transcript GET -> HTML text on 200 (else None). Delegates to the SHARED anti-429
    transport `src/utils/polite_http.py` (rotated real-browser impersonation for Cloudflare,
    Retry-After, backoff+jitter, per-host slowdown, BYO proxy). `log_missing=False` silences
    the expected wrong-exchange 404 when probing quote pages."""
    return ph.get_text(url, timeout=timeout, retries=retries, backoff=backoff,
                       impersonate=True, log_missing=log_missing)


def _sleep_pace(base: float) -> None:
    """Paced inter-request sleep for fool.com (shared per-host slowdown + jitter)."""
    ph.sleep_pace(base, _BASE)


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


def _expected_quarter_count(since: str, grace_days: int = 50) -> int:
    """How many quarterly calls we EXPECT per ticker since `since` — calendar quarters from
    `since` up to ~today, minus a grace window so the just-ended (maybe not-yet-reported)
    quarter isn't required. A ticker with at least this many post-`since` transcripts is
    treated as COMPLETE and its quote page is skipped (fewer requests -> fewer 429s)."""
    start = pd.Timestamp(since)
    end = pd.Timestamp.today() - pd.Timedelta(days=grace_days)
    if end < start:
        return 1
    return max(1, (end.year - start.year) * 4 + (end.quarter - start.quarter) + 1)


def _post_since_quarters(triples, since: str) -> dict[str, set]:
    """{ticker: {quarters}} for rows whose call/as_of date is >= `since`."""
    out: dict[str, set] = {}
    for tk, q, d in triples:
        if d and str(d)[:10] >= since:
            out.setdefault(str(tk), set()).add(q)
    return out


def build_transcript_index_by_ticker(
    context: Context, tickers: list[str] | None = None, since: str = "2025-01-01",
    exchanges: tuple[str, ...] = ("nasdaq", "nyse"), pause: float = 0.6,
) -> dict[str, dict]:
    """TARGETED discovery for the RECENT gap: ONE request per ticker to its MF quote page
    `fool.com/quote/{exchange}/{ticker}/`, which lists that name's recent transcript URLs
    (exact date + slug already in them). Keeps links with `call_date >= since` and merges
    them into the same big JSON. Unlike the global-feed crawl this is NOT capped at ~500
    pages and is complete per ticker for the recent window -> use it to get EVERY S&P 500
    call since `since` (default 2025-01-01). `tickers` restricts the universe (None = all)."""
    universe = list(context.store.load("sp500_tickers", columns=["ticker"])["ticker"])
    if tickers is not None:
        keep = set(tickers)
        universe = [t for t in universe if t in keep]
    slug_map = _universe_slug_map(universe)
    path = _index_path(context)
    index = _load_index(path)
    since = str(since)
    target = _expected_quarter_count(since)

    # already-covered tickers are SKIPPED (no request) -> fewer hits, fewer 429s, resumable.
    # A ticker is complete when it has >= `target` post-`since` quarters BOTH in the JSON index
    # AND in the DB sections table (if the DB is reachable; else JSON coverage alone).
    have_json = _post_since_quarters(
        ((r["ticker"], r["quarter"], r.get("call_date")) for r in index.values()), since)
    have_db = None
    try:
        db = context.store.load(_TABLE, columns=["ticker", "quarter", "as_of"])
        if db is not None and not db.empty:
            have_db = _post_since_quarters(
                zip(db["ticker"], db["quarter"], db["as_of"].astype(str)), since)
    except Exception:
        have_db = None                        # DB unavailable -> resume on JSON coverage alone

    def _covered(tk: str) -> bool:
        if len(have_json.get(tk, ())) < target:
            return False
        return have_db is None or len(have_db.get(tk, ())) >= target

    added_total, missing, skipped = 0, [], 0
    for tkr in tqdm(universe, "quote-page transcript urls"):
        if _covered(tkr):
            skipped += 1
            continue
        html, exch = _quote_page(tkr, exchanges)
        if html is None:
            missing.append(tkr)
            continue
        recs = [r for r in _quote_links(html, slug_map)
                if r["ticker"] == tkr and r["call_date"] >= since]
        added = 0
        for r in recs:
            if r["url"] not in index:
                index[r["url"]] = r
                have_json.setdefault(tkr, set()).add(r["quarter"])
                added += 1
        added_total += added
        logger.info("MF quote %s (%s): %d links >= %s, %d new (index total %d)",
                    tkr, exch, len(recs), since, added, len(index))
        if added:                             # DYNAMIC save -> progress survives a 429 / interrupt
            path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
        _sleep_pace(pause)

    path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.warning("Quote-page discovery: +%d new links since %s | skipped %d already-complete "
                   "| %d/%d had no quote page -> %s", added_total, since, skipped,
                   len(missing), len(universe), path)
    if missing:
        logger.info("No MF quote page (exchange miss / not covered) for %d tickers: %s",
                    len(missing), missing[:40])
    return index


# --------------------------------------------------------------------------- #
# Stage 2 + 3: download the text, parse the high-signal sections                #
# --------------------------------------------------------------------------- #
def _is_caps_header(line: str) -> bool:
    """A standalone ALL-CAPS heading line (MF markers: 'CALL PARTICIPANTS', 'TAKEAWAYS')."""
    s = line.strip().rstrip(":")
    return (2 <= len(s) <= 45 and s == s.upper() and any(c.isalpha() for c in s)
            and len(s.split()) <= 5)


# the operator's phrase that opens the analyst Q&A (splits prepared remarks from Q&A).
# Broadened across 2005-2025 transcript phrasings: the classic "question-and-answer session",
# operator-instructions, "(first) question comes/is/will be from" hand-off, "go to the line of X",
# "for our first question, we'll go/take", "we'll (now) take/go to our first question", the generic
# "we'll now begin/open/take/turn ... to questions", and "open the floor/line for questions".
_QA_MARKER = re.compile(
    r"(?i)"
    r"question[-\s]and[-\s]answer session|"
    r"questions?\s*(?:and|&)\s*answers?|"                 # standalone Q&A heading (no 'session')
    r"(?:first|next|final)\s+question\s+(?:comes?|is|will\s+come|will\s+be)\s+from|"
    r"(?:go|turn|move)\s+(?:ahead\s+)?to\s+(?:the\s+)?line\s+of\b|"          # "go to the line of X"
    r"(?:for\s+)?(?:our|your)\s+(?:first|next)\s+question,?\s+"              # "for our first question, we'll go"
    r"(?:we(?:'ll| will)|let's|i(?:'ll| will)|please)\b|"
    r"we(?:'ll| will)\s+(?:now\s+)?(?:go|move|turn|take)\s+(?:ahead\s+)?to\s+"  # "we'll take our first question"
    r"(?:our\s+)?(?:first|next)\s+(?:question|caller|line)|"
    r"(?:we(?:'ll| will| are going to| would like to)|now|let's|i(?:'ll| will))\b"
    r"[^.]{0,45}?(?:begin|open|take|start|conduct|move\s+to|go\s+to|turn[^.]{0,20}?to)"
    r"[^.]{0,30}?questions?|"
    r"open\s+(?:up\s+)?(?:the\s+)?(?:floor|line|lines|call)\b[^.]{0,25}?questions?|"
    r"\[?operator instructions\]?")


def split_prepared_qa(text: str) -> dict[str, str]:
    """Source-agnostic split of a full transcript TEXT into the high-signal sections funds
    analyse: `full` (ALWAYS kept -> format-proof), `prepared_remarks` (scripted management
    comments from call-open to the Q&A) and `qa` (the analyst Q&A, after the operator's
    hand-off). Used for BOTH the Motley Fool HTML-extracted text AND the HuggingFace
    `content` field. Call prose starts at the first 'Operator' line (skips logo/date/
    takeaways preamble); the Q&A hand-off is searched past the first ~2000 chars first
    (operators PREVIEW the Q&A in the intro), then anywhere. If NO phrase matches, fall back to the
    SECOND 'Operator' turn (the first opens the call, the second hands off to the Q&A) so a call with
    an unusual hand-off phrasing still yields a Q&A section. prepared/qa only when confidently split
    (>300 chars each)."""
    out: dict[str, str] = {"full": text}
    op = re.search(r"(?im)^\s*operator\b", text)
    prose = text[op.start():] if op else text
    m = _QA_MARKER.search(prose, 2000) or _QA_MARKER.search(prose)
    if m is None:                                        # no phrase matched -> second 'Operator' turn
        ops = list(re.finditer(r"(?im)^\s*operator\b", prose))
        m = ops[1] if len(ops) >= 2 else None
    if m:
        pre, post = prose[:m.start()].strip(), prose[m.start():].strip()
        if len(pre) > 300:
            out["prepared_remarks"] = pre
        if len(post) > 300:
            out["qa"] = post
    elif op:
        out["prepared_remarks"] = prose.strip()          # no Q&A hand-off found -> all remarks
    return out


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
                         pause: float = 0.6, limit: int | None = None) -> int:
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


def ingest_earnings_calls(context: Context, tickers: list[str] | None = None) -> int:
    """Parse every cached transcript HTML into sections and upsert to
    `earnings_call_sections` (ticker, quarter, as_of, tag, text, url). Returns rows
    upserted. Only sections with real text are stored. `tickers` restricts to a subset
    (None = every cached transcript)."""
    cache_dir = _cache_dir(context)
    index = {(r["ticker"], r["quarter"]): r for r in _load_index(_index_path(context)).values()}
    keep = set(tickers) if tickers is not None else None
    rows: list[dict] = []
    for html_path in tqdm(cache_dir.glob("*/*.html"), "EC ingestion db"):
        ticker, quarter = html_path.parent.name, html_path.stem
        if keep is not None and ticker not in keep:
            continue
        rec = index.get((ticker, quarter), {})
        sections = parse_transcript_sections(html_path.read_text(encoding="utf-8", errors="replace"))
        for tag, text in sections.items():
            if len(text) < 40:                          # skip empty / stub sections
                continue
            rows.append({"ticker": ticker, "quarter": quarter, "tag": tag,
                         "as_of": rec.get("call_date"), "url": rec.get("url"), "text": text})
    if not rows:
        logger.warning("No transcript sections parsed to ingest.")
        return 0
    
    df = pd.DataFrame(rows)
    saved = context.store.save(_TABLE, df)
    logger.info("Ingested %d transcript sections (%d transcripts, %d tickers) -> '%s'",
                   saved, df.groupby(["ticker", "quarter"]).ngroups, df["ticker"].nunique(), _TABLE)
    return saved


def download_earnings_calls(context: Context, tickers: list[str] | None = None,
                            limit: int | None = None, include_hf: bool = True,
                            recent_since: str = "2025-01-01", use_global_crawl: bool = False,
                            mf_history_years: float = 2.0) -> None:
    """DOWNLOAD / extract stage — writes only to DISK (no DB), so it can run as its own DAG task:
      1. cache the HuggingFace backbone parquet (once; ~1.8 GB, skipped if present),
      2. per-ticker QUOTE-PAGE discovery of the recent Motley Fool transcript URLs (since
         `recent_since`; one request/ticker, uncapped),
      3. (optional) the legacy global-feed crawl (`use_global_crawl=True`) fallback,
      4. download each indexed MF transcript's HTML to the cache dir.
    Incremental: the HF parquet + MF HTML are skipped when already on disk. `tickers` restricts the
    subset (None = full universe); `limit` bounds the MF download for a test; `include_hf=False`
    skips the backbone."""
    if include_hf:
        # deferred import: fetch_hf_transcripts imports helpers from THIS module
        from src.data_extract.utils.behavioral.fetch_hf_transcripts import download_hf_parquet
        download_hf_parquet(context)
    build_transcript_index_by_ticker(context, tickers=tickers, since=recent_since)
    if use_global_crawl:
        build_transcript_index(context, tickers=tickers, history_years=mf_history_years)
    download_transcripts(context, tickers=tickers, limit=limit)


def ingest_all_earnings_calls(context: Context, tickers: list[str] | None = None,
                              include_hf: bool = True) -> int:
    """INGEST stage — parse the already-downloaded transcripts into `earnings_call_sections`:
      * the cached HuggingFace parquet (`ingest_hf_transcripts`; re-reads the cached file), and
      * the cached Motley Fool HTML (`ingest_earnings_calls`).
    Runs after `download_earnings_calls`. Incremental + deduped by (ticker, quarter). Returns the
    number of section rows upserted."""
    saved = 0
    if include_hf:
        from src.data_extract.utils.behavioral.fetch_hf_transcripts import ingest_hf_transcripts
        saved += ingest_hf_transcripts(context, tickers=tickers)   # cached parquet -> sections
    saved += ingest_earnings_calls(context, tickers=tickers)       # cached MF HTML -> sections
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
