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
import time
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.constants.constants import (
    EARNINGS_CALL_CACHE_DIR,
    EARNINGS_CALL_SECTIONS_TABLE,
    MOTLEY_FOOL_BASE_URL,
    MOTLEY_FOOL_HEADERS,
    MOTLEY_FOOL_TRANSCRIPT_INDEX_URL,
)
from src.context import Context

logger = logging.getLogger(__name__)

_BASE = MOTLEY_FOOL_BASE_URL
_INDEX = MOTLEY_FOOL_TRANSCRIPT_INDEX_URL
# fuller browser headers (fool.com is behind Cloudflare; UA-only requests are more
# likely to be challenged). Extends the UA-only constant WITHOUT editing constants.py.
_HEADERS = {**MOTLEY_FOOL_HEADERS,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive"}
_TABLE = EARNINGS_CALL_SECTIONS_TABLE

_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    """Shared keep-alive session (connection reuse + stable headers looks less bot-like
    to Cloudflare than a fresh connection per request)."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(_HEADERS)
        _SESSION = s
    return _SESSION

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


def _get(url: str, timeout: int = 30, retries: int = 3, backoff: float = 2.0) -> str | None:
    """GET with retry + exponential backoff on rate-limit / transient errors. A non-200
    is LOGGED (not swallowed) so Cloudflare throttling — the classic "downloads stop after
    N" symptom — is visible instead of silently returning None and dropping the record."""
    for attempt in range(retries + 1):
        try:
            r = _session().get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
            if r.status_code in (403, 429) or r.status_code >= 500:  # blocked / throttled / transient
                if attempt < retries:
                    wait = backoff * (2 ** attempt)
                    logger.warning("GET %s -> HTTP %d (rate-limited/blocked); backoff %.1fs "
                                   "(retry %d/%d)", url, r.status_code, wait, attempt + 1, retries)
                    time.sleep(wait)
                    continue
                logger.warning("GET %s -> HTTP %d after %d retries; giving up (site is "
                               "throttling — raise `pause` / lower request rate)", url,
                               r.status_code, retries)
            else:                                        # 4xx that won't fix on retry (404 etc.)
                logger.warning("GET %s -> HTTP %d", url, r.status_code)
            return None
        except Exception as e:                          # noqa: BLE001
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            logger.warning("GET failed %s: %s", url, e)
            return None
    return None


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
                           max_pages: int = 1000, stop_after_empty: int = 4,
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
        time.sleep(pause)

    path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.warning("Transcript index: %d links across %d tickers (%d new this run) -> %s",
                   len(index), len({r["ticker"] for r in index.values()}),
                   len(index) - before, path)
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
# Broadened across 2005-2025 transcript phrasings (tuned on real HF rows: split coverage
# 50% -> 93%): the classic "question-and-answer session", the operator-instructions +
# "first question comes from" hand-off, "we'll now begin/open/take/turn ... to questions",
# "open the floor/line for questions".
_QA_MARKER = re.compile(
    r"(?i)"
    r"question[-\s]and[-\s]answer session|"
    r"questions?\s*(?:and|&)\s*answers?|"                 # standalone Q&A heading (no 'session')
    r"(?:first|next)\s+question\s+(?:comes?|is|will\s+come)\s+from|"
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
    (operators PREVIEW the Q&A in the intro), then anywhere. prepared/qa only when
    confidently split (>300 chars each)."""
    out: dict[str, str] = {"full": text}
    op = re.search(r"(?im)^\s*operator\b", text)
    prose = text[op.start():] if op else text
    m = _QA_MARKER.search(prose, 2000) or _QA_MARKER.search(prose)
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


def _transcript_path(cache_dir: Path, rec: dict) -> Path:
    d = cache_dir / rec["ticker"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{rec['quarter']}.txt"


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
    for rec in todo:
        html = _get(rec["url"])
        if not html or "call-transcripts" not in html.lower():
            continue
        out = cache_dir / rec["ticker"] / f"{rec['quarter']}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        n += 1
        time.sleep(pause)
    logger.warning("Downloaded %d new transcripts (%d already cached) -> %s",
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
    for html_path in cache_dir.glob("*/*.html"):
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
    logger.warning("Ingested %d transcript sections (%d transcripts, %d tickers) -> '%s'",
                   saved, df.groupby(["ticker", "quarter"]).ngroups, df["ticker"].nunique(), _TABLE)
    return saved


def fetch_earnings_calls(context: Context, tickers: list[str] | None = None,
                         limit: int | None = None, mf_history_years: float = 2.0,
                         include_hf: bool = True) -> int:
    """Full transcript pipeline. The HuggingFace dataset `kurry/sp500_earnings_transcripts`
    is the 2005-2025 BACKBONE (deep history in one clean download); the Motley Fool crawl
    only fills the RECENT gap past the dataset's ~2025 cut, bounded to `mf_history_years`
    where MF's shallow crawl is reliable (going 15y deep on MF's global feed is impractical
    and gets throttled). Every stage is incremental and deduped by (ticker, quarter).
    `tickers` restricts to a subset (None = full universe); `limit` bounds the MF download
    for a test; set `include_hf=False` to skip the backbone (MF-only)."""
    saved = 0
    if include_hf:
        # deferred import: fetch_hf_transcripts imports helpers from THIS module
        from src.data_extract.utils.behavioral.fetch_hf_transcripts import ingest_hf_transcripts
        saved += ingest_hf_transcripts(context, tickers=tickers)
    build_transcript_index(context, tickers=tickers, history_years=mf_history_years)
    download_transcripts(context, tickers=tickers, limit=limit)
    saved += ingest_earnings_calls(context, tickers=tickers)
    return saved
