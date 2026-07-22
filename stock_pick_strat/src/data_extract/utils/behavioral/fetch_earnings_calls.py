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
import os
import random
import re
import time
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import requests
from curl_cffi import requests as cr
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

# --- anti-429 transport ---------------------------------------------------- #
# We COOPERATE with the rate limit (Cloudflare 429), we do NOT evade a ban: rotate a REAL
# browser IMPERSONATION profile per request (curl_cffi -> a coherent UA+TLS+header
# fingerprint, so the request looks like Chrome/Safari to Cloudflare, not a python-requests
# JA3), honour Retry-After, back off exponentially WITH JITTER, and PERMANENTLY slow the whole
# run after repeated 429s. Optional bring-your-own proxy via env (PEA_SCRAPE_PROXY / HTTPS_PROXY);
# we deliberately do NOT ship or rotate anonymous/residential proxy pools.
_IMPERSONATE_POOL = ("chrome124", "chrome123", "chrome120", "chrome131", "safari17_0", "edge101")
_PROXY_ENV = ("PEA_SCRAPE_PROXY", "HTTPS_PROXY", "https_proxy")
_PACE = {"mult": 1.0}          # run-wide slowdown multiplier, ratcheted up on each 429
_PACE_CAP = 8.0
_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    """requests fallback session (used only if curl_cffi is unavailable)."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(_HEADERS)
        _SESSION = s
    return _SESSION


def _proxy() -> dict | None:
    for k in _PROXY_ENV:
        v = os.getenv(k)
        if v:
            return {"http": v, "https": v}
    return None


def _retry_after_seconds(resp) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date) into seconds, if present."""
    try:
        ra = resp.headers.get("Retry-After")
    except Exception:
        return None
    if not ra:
        return None
    try:
        return float(ra)
    except (TypeError, ValueError):
        try:
            import datetime as _dt
            from email.utils import parsedate_to_datetime
            return max(0.0, (parsedate_to_datetime(ra)
                             - _dt.datetime.now(_dt.timezone.utc)).total_seconds())
        except Exception:
            return None

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


def _http_get(url: str, timeout: int):
    """One GET via curl_cffi with a ROTATED browser impersonation (coherent UA+TLS+headers ->
    looks like a real browser to Cloudflare); falls back to the requests session. Returns a
    response (.status_code/.text/.headers) or None on a transport error."""
    proxies = _proxy()
    prof = random.choice(_IMPERSONATE_POOL)
    try:
        try:
            return cr.get(url, impersonate=prof, timeout=timeout, proxies=proxies)
        except Exception:                       # unknown profile / TLS quirk -> generic chrome
            return cr.get(url, impersonate="chrome", timeout=timeout, proxies=proxies)
    except Exception:
        try:
            return _session().get(url, timeout=timeout, proxies=proxies)
        except Exception:
            return None


def _get(url: str, timeout: int = 30, retries: int = 4, backoff: float = 3.0,
         log_missing: bool = True) -> str | None:
    """GET with browser-impersonation + adaptive rate-limit handling. On 429/403/5xx it
    honours Retry-After, backs off exponentially WITH JITTER, and ratchets up a run-wide pace
    multiplier so the whole crawl slows down after being throttled (the correct response to a
    429 — cooperate, don't hammer). A terminal non-200 is logged unless `log_missing=False`
    (the quiet wrong-exchange probe)."""
    for attempt in range(retries + 1):
        r = _http_get(url, timeout)
        if r is None:                                    # transport error (all paths failed)
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt) + random.uniform(0.5, 2.0))
                continue
            logger.warning("GET failed (transport) %s", url)
            return None
        code = getattr(r, "status_code", 0)
        if code == 200:
            return r.text
        if code in (403, 429) or code >= 500:            # blocked / throttled / transient
            if attempt < retries:
                wait = max(_retry_after_seconds(r) or 0.0,
                           backoff * (2 ** attempt)) + random.uniform(0.5, 2.5)
                if code == 429:
                    _PACE["mult"] = min(_PACE_CAP, _PACE["mult"] * 1.6)   # slow the WHOLE run
                logger.warning("GET %s -> HTTP %d (rate-limited); wait %.1fs, run-pace x%.1f "
                               "(retry %d/%d)", url, code, wait, _PACE["mult"], attempt + 1, retries)
                time.sleep(wait)
                continue
            logger.warning("GET %s -> HTTP %d after %d retries; giving up. If this persists, "
                           "lower the rate or set PEA_SCRAPE_PROXY.", url, code, retries)
        elif log_missing:                                # 4xx that won't fix on retry (404 etc.)
            logger.warning("GET %s -> HTTP %d", url, code)
        return None
    return None


def _sleep_pace(base: float) -> None:
    """Inter-request sleep = base * run-wide-pace-multiplier + jitter -> a non-robotic cadence
    that also honours the post-429 slowdown. `base<=0` disables throttling (tests)."""
    if base <= 0:
        return
    time.sleep(base * _PACE["mult"] + random.uniform(0.1, 0.7))


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
        time.sleep(pause)
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


def fetch_earnings_calls(context: Context, tickers: list[str] | None = None,
                         limit: int | None = None, include_hf: bool = True,
                         recent_since: str = "2025-01-01", use_global_crawl: bool = False,
                         mf_history_years: float = 2.0) -> int:
    """Full transcript pipeline, combining the best free sources:
      1. HuggingFace `kurry/sp500_earnings_transcripts` = clean 2005->Q1'25 BACKBONE.
      2. Per-ticker QUOTE-PAGE discovery = COMPLETE Motley Fool transcripts since
         `recent_since` (default 2025-01-01): one request per ticker, NOT capped at MF's
         ~500-page global feed, so it fills the whole recent gap.
      3. (optional) the legacy global-feed crawl (`use_global_crawl=True`) -- capped/partial,
         kept only as a fallback.
    Then download + ingest. Every stage is incremental and deduped by (ticker, quarter).
    `tickers` restricts to a subset (None = full universe); `limit` bounds the download for a
    test; `include_hf=False` skips the backbone."""
    saved = 0
    if include_hf:
        # deferred import: fetch_hf_transcripts imports helpers from THIS module
        from src.data_extract.utils.behavioral.fetch_hf_transcripts import ingest_hf_transcripts
        saved += ingest_hf_transcripts(context, tickers=tickers)
    build_transcript_index_by_ticker(context, tickers=tickers, since=recent_since)
    if use_global_crawl:
        build_transcript_index(context, tickers=tickers, history_years=mf_history_years)
    download_transcripts(context, tickers=tickers, limit=limit)
    saved += ingest_earnings_calls(context, tickers=tickers)
    return saved
