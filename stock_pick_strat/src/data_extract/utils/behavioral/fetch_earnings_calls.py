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
_HEADERS = MOTLEY_FOOL_HEADERS
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


def _get(url: str, timeout: int = 30) -> str | None:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        return r.text if r.status_code == 200 else None
    except Exception as e:                              # noqa: BLE001
        logger.warning("GET failed %s: %s", url, e)
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


def build_transcript_index(context: Context, tickers: list[str] | None = None,
                           max_pages: int = 800, stop_after_empty: int = 3,
                           pause: float = 0.4) -> dict[str, dict]:
    """Crawl the MF transcript index (newest first) and MERGE every universe
    transcript link into the big JSON. Incremental: stops after `stop_after_empty`
    consecutive pages that add no NEW links (a converged re-run only reads a page or
    two of the newest transcripts). `tickers` restricts the kept universe (None = all)."""
    universe = list(context.store.load("sp500_tickers", columns=["ticker"])["ticker"])
    if tickers is not None:                          # scope to a subset (e.g. a test run)
        keep = set(tickers)
        universe = [t for t in universe if t in keep]
    slug_map = _universe_slug_map(universe)
    path = _index_path(context)
    index = _load_index(path)

    empty_streak = 0
    for page in range(1, max_pages + 1):
        # MF paginates at /earnings-call-transcripts/page/N/ (page 1 = the base URL);
        # the old ?page=N query param is ignored by the site (always served page 1).
        html = _get(_INDEX if page == 1 else f"{_INDEX}page/{page}/")
        if not html:
            break
        recs = _links_on_page(html, slug_map)
        added = 0
        for rec in recs:
            if rec["url"] not in index:
                index[rec["url"]] = rec
                added += 1
        logger.info("MF index page %d: %d universe links, %d new (total %d)",
                    page, len(recs), added, len(index))
        empty_streak = empty_streak + 1 if added == 0 else 0
        if empty_streak >= stop_after_empty:
            break
        time.sleep(pause)

    path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.warning("Transcript index: %d links across %d tickers -> %s",
                   len(index), len({r["ticker"] for r in index.values()}), path)
    return index


# --------------------------------------------------------------------------- #
# Stage 2 + 3: download the text, parse the high-signal sections                #
# --------------------------------------------------------------------------- #
def _is_caps_header(line: str) -> bool:
    """A standalone ALL-CAPS heading line (MF markers: 'CALL PARTICIPANTS', 'TAKEAWAYS')."""
    s = line.strip().rstrip(":")
    return (2 <= len(s) <= 45 and s == s.upper() and any(c.isalpha() for c in s)
            and len(s.split()) <= 5)


# the operator's phrase that opens the analyst Q&A (splits prepared remarks from Q&A)
_QA_MARKER = re.compile(
    r"(?i)question[- ]and[- ]answer session|questions?\s+and\s+answers?|"
    r"(?:we(?:'ll| will)|now)\b[^.]{0,40}?(?:begin|open|take|start)[^.]{0,25}?question")


def parse_transcript_sections(html: str) -> dict[str, str]:
    """Split a Motley Fool transcript into the high-signal sections funds analyse:
      participants     (who was on the call -- a caps-headed list),
      prepared_remarks (scripted management comments, from call-open to the Q&A),
      qa               (the analyst Q&A -- the richest part, after the operator's
                        'question-and-answer session' hand-off),
    plus `full` (the whole transcript, ALWAYS kept -> format-proof: even if the split
    heuristics miss, the raw text is stored for later sentiment / embedding).

    MF's current format has no 'Prepared Remarks'/'Q&A' headings, so we take the
    participants block from its ALL-CAPS header, then split the CALL PROSE (from the
    first 'Operator:' line) at the operator's Q&A hand-off phrase."""
    soup = BeautifulSoup(html, "html.parser")
    div = (soup.find("div", class_=lambda c: c and "transcript-content" in c)
           or soup.find("div", class_=lambda c: c and "article-body" in c))
    if div is None:
        return {}
    text = div.get_text("\n", strip=True)
    if len(text) < 200:
        return {}
    out: dict[str, str] = {"full": text}

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

    # call prose starts at the first "Operator:" line (skips logo/date/takeaways preamble)
    op = re.search(r"(?im)^\s*operator\b", text)
    prose = text[op.start():] if op else text
    # operators often PREVIEW the Q&A in their opening remarks, so search for the real
    # hand-off past the intro first (>=2000 chars in), then fall back to anywhere.
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


def _transcript_path(cache_dir: Path, rec: dict) -> Path:
    d = cache_dir / rec["ticker"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{rec['quarter']}.txt"


def download_transcripts(context: Context, tickers: list[str] | None = None,
                         pause: float = 0.4, limit: int | None = None) -> int:
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
                         limit: int | None = None) -> int:
    """Full pipeline: (re)build the link index, download new transcripts, ingest
    sections. Every stage is incremental. `tickers` restricts to a subset (None = the
    full universe); `limit` bounds the number of transcripts downloaded for a test."""
    build_transcript_index(context, tickers=tickers)
    download_transcripts(context, tickers=tickers, limit=limit)
    return ingest_earnings_calls(context, tickers=tickers)
