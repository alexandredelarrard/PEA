"""Earnings-call transcripts: the HF backbone, the Roic API, then Motley Fool (free HTML) for what is still missing.

Download stage: one gap computation, then each source gets only what the previous one left; Fool links are
discovered per ticker from quote pages (or an optional global index crawl) into a JSON index, and raw HTML is
cached to `{call_transcripts}/{TICKER}/{quarter}.html`. Ingest stage: parse cached HTML into sections and
upsert `earnings_call_sections` (ticker, quarter, as_of, tag, text, url). `quarter` is the FISCAL quarter from
the transcript URL slug; `as_of` is the URL's publication date. Both stages are incremental.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import cast

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.constants.constants import EARNINGS_CALL_REPORT_GRACE_DAYS, EARNINGS_CALL_REQUEST_PAUSE, EARNINGS_CALL_SCORED_TAGS, FOOL_BASE
from src.context import Context
from src.data_extract.utils.behavioral.fetch_hf_transcripts import download_hf_parquet, ingest_hf_transcripts
from src.data_extract.utils.behavioral.fetch_roic_transcripts import fetch_roic_transcripts
from src.data_extract.utils.behavioral.utils_behavior import _get, _index_path, _load_index, _sleep_pace
from src.data_extract.utils.behavioral.utils_earnings_call_cache import (
    invalidate_earnings_call_derivatives,
    save_earnings_call_sections,
)

# The one gap definition; never re-derived here.
from src.data_extract.utils.behavioral.utils_missing_quarters import (
    _index_to_quarter,
    _parse_quarter,
    _quarter_index,
    missing_quarters_by_ticker,
    remaining_after,
    stored_call_quarters,
)
from src.data_extract.utils.behavioral.utils_split_qa import split_prepared_qa
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.run_manifest import record_run
from src.data_store.schema import Tables
from src.utils.text_metrics import assess_earnings_call_sections

logger = logging.getLogger(__name__)

_INDEX = "https://www.fool.com/earnings-call-transcripts/"

# Transcript-link path with (year, month, day, slug, quarter, fiscal-year) groups.
_LINK_RE = re.compile(r"/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/(.+?)-q([1-4])-(\d{4})-earnings-call-transcript")
_HREF_RE = re.compile(r'href="(/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/[^"?#]+)"')


# --- Stage 1: discover transcript links into the JSON index ---
def _universe_slug_map(universe: list[str]) -> dict[str, str]:
    """ticker -> its lowercase MF slug form (BRK-B -> 'brk-b'), for suffix matching."""
    return {str(t): str(t).lower() for t in universe}


def _parse_link(href: str, slug_map: dict[str, str]) -> dict | None:
    """One transcript href -> {ticker, quarter, call_date, url}, or None if not in the universe.

    The ticker is the universe ticker whose slug form ends the URL slug ('...-brk-b' -> BRK-B, not 'b')."""
    m = _LINK_RE.search(href)
    if not m:
        return None
    yr, mo, dy, slug, q, fy = m.groups()
    tkr = next((t for t, s in slug_map.items() if slug == s or slug.endswith("-" + s)), None)
    if tkr is None:
        return None
    return {"ticker": tkr, "quarter": f"{fy}Q{q}", "call_date": f"{yr}-{mo}-{dy}", "url": FOOL_BASE + href.rstrip("/") + "/"}


def _links_on_page(html: str, slug_map: dict[str, str]) -> list[dict]:
    recs, seen = [], set()
    for href in _HREF_RE.findall(html or ""):
        rec = _parse_link(href, slug_map)
        if rec and rec["url"] not in seen:
            seen.add(rec["url"])
            recs.append(rec)
    return recs


def _page_call_dates(html: str) -> list[str]:
    """Every transcript's call date (YYYY-MM-DD) on the page, universe or not; the feed is newest-first,
    so the newest date gauges how far back the crawl has paged."""
    out = []
    for href in _HREF_RE.findall(html or ""):
        m = _LINK_RE.search(href)
        if m:
            out.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return out


def _page_converged(recs: list[dict], added: int) -> bool:
    """True only when the page HAS universe transcripts and all are already indexed; a page with no
    universe names (small-cap noise in the global feed) must not count toward the convergence stop."""
    return bool(recs) and added == 0


def build_transcript_index(
    context: Context,
    tickers: list[str] | None = None,
    max_pages: int = 490,
    stop_after_empty: int = 4,
    pause: float = 0.6,
    history_years: float = 6.0,
) -> dict[str, dict]:
    """Crawl the MF transcript index (a global, all-companies, newest-first feed) and
    MERGE every universe transcript link into the big JSON.

    Stops on convergence (`stop_after_empty` consecutive `_page_converged` pages), on passing the
    `history_years` horizon, or at `max_pages` / a page that will not load. `tickers` restricts the
    kept universe (None = all)."""
    roster = context.store.load(Tables.sp500_tickers, columns=["ticker"])
    assert roster is not None
    universe = list(cast(pd.Series, roster["ticker"]))
    if tickers is not None:  # scope to a subset (e.g. a test run)
        keep = set(tickers)
        universe = [t for t in universe if t in keep]
    slug_map = _universe_slug_map(universe)
    path = _index_path(context)
    index = _load_index(path)
    before = len(index)
    min_date = (pd.Timestamp.today().normalize() - pd.DateOffset(years=int(history_years))).strftime("%Y-%m-%d")

    empty_streak = 0
    for page in tqdm(range(1, max_pages + 1), "transcript index urls"):
        # MF paginates at /earnings-call-transcripts/page/N/ (page 1 = the base URL).
        html = _get(_INDEX if page == 1 else f"{_INDEX}page/{page}/")
        if not html:
            logger.warning("MF index page %d did not load (blocked or end of feed) -> stop at %d links", page, len(index))
            break
        recs = _links_on_page(html, slug_map)
        added = 0
        for r in recs:
            if r["url"] not in index:
                index[r["url"]] = r
                added += 1
        newest = max(_page_call_dates(html), default=None)
        empty_streak = empty_streak + 1 if _page_converged(recs, added) else 0
        logger.info(
            "MF index page %d: %d universe links (%d new), newest %s, converge-streak %d/%d (index total %d)",
            page,
            len(recs),
            added,
            newest,
            empty_streak,
            stop_after_empty,
            len(index),
        )
        if empty_streak >= stop_after_empty:
            logger.info("Index converged (%d consecutive pages of already-seen universe links) -> stop", stop_after_empty)
            break
        if newest and newest < min_date:
            logger.info("Reached %.0fy history horizon (newest call on page %s < %s) -> stop", history_years, newest, min_date)
            break
        _sleep_pace(pause)

    path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.warning(
        "Transcript index: %d links across %d tickers (%d new this run) -> %s",
        len(index),
        len({r["ticker"] for r in index.values()}),
        len(index) - before,
        path,
    )
    return index


# --- Stage 1b: targeted discovery via per-ticker quote pages ---
# Transcript path as it appears in a quote page's raw text/JSON (not always inside href="...").
_QUOTE_PATH_RE = re.compile(r"/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/[a-z0-9-]+?-q[1-4]-\d{4}-earnings-call-transcript")


def _quote_links(html: str, slug_map: dict[str, str]) -> list[dict]:
    """Transcript links on a company quote page, matched on the raw path and resolved via `_parse_link`."""
    recs, seen = [], set()
    for m in _QUOTE_PATH_RE.finditer((html or "").lower()):
        rec = _parse_link(m.group(0), slug_map)
        if rec and rec["url"] not in seen:
            seen.add(rec["url"])
            recs.append(rec)
    return recs


def _quote_page(ticker: str, exchanges: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Fetch a ticker's MF quote page `/quote/{exchange}/{ticker}/`, trying each exchange (a miss 404s quietly).
    Returns (html, exchange), or (None, None)."""
    for exch in exchanges:
        html = _get(f"{FOOL_BASE}/quote/{exch}/{ticker.lower()}/", log_missing=False)
        if html:
            return html, exch
    return None, None


def build_transcript_index_by_ticker(
    context: Context,
    tickers: list[str] | None = None,
    missing: dict[str, list[str]] | None = None,
    since: str = "2025-01-01",
    exchanges: tuple[str, ...] = ("nasdaq", "nyse"),
    pause: float = EARNINGS_CALL_REQUEST_PAUSE,
    grace_days: int = EARNINGS_CALL_REPORT_GRACE_DAYS,
) -> dict[str, dict]:
    """Targeted discovery of the quarters still missing after the HF backbone and Roic: the last resort.

    `missing` is `{ticker: [quarters]}` from `missing_quarters_by_ticker` (computed here when None). Only
    tickers with a gap are requested, one quote page each, and links older than the gap start are dropped.
    The JSON index is saved after every ticker that adds links, so an interrupt loses no progress."""

    if missing is None:
        missing = missing_quarters_by_ticker(context, tickers=tickers, since=str(since), grace_days=grace_days)
    if not missing:
        logger.info("Quote-page discovery: nothing missing (HF backbone + Roic + DB current).")
        return _load_index(_index_path(context))

    slug_map = _universe_slug_map(list(missing))
    path = _index_path(context)
    index = _load_index(path)

    # Random order spreads the load so a throttled run does not always die on the same names.
    order = list(missing)
    random.shuffle(order)

    added_total, missing_page = 0, []
    for tkr in tqdm(order, "quote-page transcript urls"):
        need = set(missing[tkr])
        if not need:  # nothing to ask for -> no request
            continue
        gap_min = min(_quarter_index(*cast(tuple[int, int], _parse_quarter(q))) for q in need)
        html, exch = _quote_page(tkr, exchanges)
        if html is None:
            missing_page.append(tkr)
            _sleep_pace(pause)  # still pace: a 404 probe cost 2 requests
            continue
        added = 0
        for r in _quote_links(html, slug_map):
            pq = _parse_quarter(r["quarter"])
            if r["ticker"] != tkr or pq is None or _quarter_index(*pq) < gap_min or r["url"] in index:
                continue
            index[r["url"]] = r
            added += 1
        added_total += added
        logger.info(
            "MF quote %s (%s): need %d quarters (%s..), %d new links (index total %d)",
            tkr,
            exch,
            len(need),
            _index_to_quarter(gap_min),
            added,
            len(index),
        )
        if added:  # save now so progress survives a 429 / interrupt
            path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
        _sleep_pace(pause)

    path.write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.warning(
        "Quote-page discovery: +%d new links across %d ticker(s) with a gap | %d had no quote page -> %s",
        added_total,
        len(missing),
        len(missing_page),
        path,
    )
    if missing_page:
        logger.info("No MF quote page (exchange miss / not covered) for %d tickers: %s", len(missing_page), missing_page[:40])
    return index


# --- Stages 2 + 3: download the HTML, parse the sections ---
def _is_caps_header(line: str) -> bool:
    """A standalone ALL-CAPS heading line (MF markers: 'CALL PARTICIPANTS', 'TAKEAWAYS')."""
    s = line.strip().rstrip(":")
    return 2 <= len(s) <= 45 and s == s.upper() and any(c.isalpha() for c in s) and len(s.split()) <= 5


def parse_transcript_sections(html: str) -> dict[str, str]:
    """Motley Fool transcript HTML -> sections: `split_prepared_qa` (full/prepared_remarks/qa) plus the caps-headed `participants` block."""
    soup = BeautifulSoup(html, "html.parser")
    div = soup.find("div", class_=lambda c: bool(c and "transcript-content" in c)) or soup.find(
        "div", class_=lambda c: bool(c and "article-body" in c)
    )
    if div is None:
        return {}
    text = div.get_text("\n", strip=True)
    if len(text) < 200:
        return {}
    out = split_prepared_qa(text)

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    parts, grab = [], False
    for ln in lines:  # participants = caps-bounded block
        if _is_caps_header(ln):
            grab = "participant" in ln.lower()
            continue
        if grab:
            parts.append(ln)
    if parts:
        out["participants"] = "\n".join(parts)
    return out


def _valid_cached_transcript(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        sections = parse_transcript_sections(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return False
    return assess_earnings_call_sections(sections).valid


def download_transcripts(
    context: Context, tickers: list[str] | None = None, pause: float = EARNINGS_CALL_REQUEST_PAUSE, limit: int | None = None
) -> int:
    """Download each indexed transcript not yet validly cached to `{ticker}/{quarter}.html`; returns the count downloaded.
    `tickers` restricts to a subset (None = all); `limit` bounds a test run."""

    cache = cache_dir(context, context.config.local.paths.call_transcripts)
    index = _load_index(_index_path(context))
    keep = set(tickers) if tickers is not None else None
    todo = [
        r
        for r in index.values()
        if (keep is None or r["ticker"] in keep) and not _valid_cached_transcript(cache / r["ticker"] / f"{r['quarter']}.html")
    ]

    # Random order spreads the load and makes `limit` a random sample.
    random.shuffle(todo)
    if limit is not None:
        todo = todo[:limit]

    n = 0
    for rec in tqdm(todo, "EC download"):
        html = _get(rec["url"])
        if not html or "call-transcripts" not in html.lower():
            continue
        out = cache / rec["ticker"] / f"{rec['quarter']}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        n += 1
        _sleep_pace(pause)
    logger.info("Downloaded %d new transcripts (%d already cached) -> %s", n, len(index) - n, cache)
    return n


def _existing_section_keys(context: Context, tickers: list[str] | None) -> set[tuple[str, str]]:
    """(ticker, quarter) of quality-valid calls already in `earnings_call_sections` from any source (None = every stored ticker)."""
    scope = tickers if tickers is not None else context.store.distinct(Tables.earnings_call_sections, "ticker")
    valid, _ = stored_call_quarters(context, scope)
    return {(ticker, quarter) for ticker, quarters in valid.items() for quarter in quarters}


def ingest_earnings_calls(context: Context, tickers: list[str] | None = None, force: bool = False) -> int:
    """Parse cached transcript HTML into sections and upsert to `earnings_call_sections`
    (ticker, quarter, as_of, tag, text, url). Returns rows upserted.

    A (ticker, quarter) already stored is skipped without reading its HTML; a malformed transcript stores a
    null-producing marker so another source can retry it. `force=True` re-parses everything; `tickers`
    restricts to a subset (None = every cached transcript)."""
    cache = cache_dir(context, context.config.local.paths.call_transcripts)
    index = {(r["ticker"], r["quarter"]): r for r in _load_index(_index_path(context)).values()}
    keep = set(tickers) if tickers is not None else None
    existing = set() if force else _existing_section_keys(context, tickers)

    rows: list[dict] = []
    parsed = skipped = 0
    for html_path in tqdm(sorted(cache.glob("*/*.html")), "EC ingestion db"):
        ticker, quarter = html_path.parent.name, html_path.stem
        if keep is not None and ticker not in keep:
            continue
        if (ticker, quarter) in existing:  # already ingested -> skip (incremental)
            skipped += 1
            continue
        rec = index.get((ticker, quarter), {})
        sections = parse_transcript_sections(html_path.read_text(encoding="utf-8", errors="replace"))
        quality = assess_earnings_call_sections(sections)
        if not quality.valid:
            logger.warning("MF %s %s malformed (%s); storing null-producing marker for source retry.", ticker, quarter, quality.reason)
            for tag in EARNINGS_CALL_SCORED_TAGS:
                rows.append(
                    {
                        "ticker": ticker,
                        "quarter": quarter,
                        "tag": tag,
                        "as_of": rec.get("call_date"),
                        "url": rec.get("url"),
                        "text": quality.cleaned_sections.get(tag, ""),
                    }
                )
            parsed += 1
            existing.add((ticker, quarter))
            continue
        for tag, text in quality.cleaned_sections.items():
            if len(text) < 40:  # skip empty / stub sections
                continue
            rows.append({"ticker": ticker, "quarter": quarter, "tag": tag, "as_of": rec.get("call_date"), "url": rec.get("url"), "text": text})
        parsed += 1
        existing.add((ticker, quarter))  # de-dup within this run too

    if not rows:
        logger.info("MF ingest: nothing new — %d cached transcript(s) already ingested.", skipped)
        return 0
    df = pd.DataFrame(rows)
    saved = save_earnings_call_sections(context, df)
    logger.info(
        "MF ingest: +%d sections from %d NEW transcripts (%d cached skipped, %d tickers) -> '%s'",
        saved,
        parsed,
        skipped,
        df["ticker"].nunique(),
        Tables.earnings_call_sections,
    )
    return saved


def _invalidate_derived_calls(context: Context, missing: dict[str, list[str]]) -> int:
    """Drop cached features for calls the shared quality gate says need recovery."""
    calls = pd.DataFrame(
        [(ticker, quarter) for ticker, quarters in missing.items() for quarter in quarters],
        columns=["ticker", "quarter"],
    )
    return invalidate_earnings_call_derivatives(context, calls)


def download_earnings_calls(
    context: Context,
    tickers: list[str] | None = None,
    limit: int | None = None,
    recent_since: str = "2025-01-01",
    use_global_crawl: bool = False,
    mf_history_years: float = 2.0,
    use_roic: bool = True,
) -> None:
    """Download stage: cache the HF backbone, compute the recent gap once, then fill it from Roic (`use_roic`) and
    finally Motley Fool (quote pages, optional global crawl, HTML download), each source getting only the remainder.

    The HF download is non-fatal: it is deep history, and the gap falls back to its date floor without it.
    `tickers` restricts the subset; `limit` bounds the MF download."""

    try:
        download_hf_parquet(context)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "HF backbone download failed (%s: %s) -> continuing without it; ROIC and Fool still run and the recent gap falls back to the date floor.",
            type(exc).__name__,
            exc,
        )

    missing = missing_quarters_by_ticker(context, tickers=tickers, since=recent_since)
    logger.info("Recent gap: %d ticker(s) missing %d quarter(s) in total.", len(missing), sum(len(v) for v in missing.values()))
    invalidated = _invalidate_derived_calls(context, missing)
    if invalidated:
        logger.info("Invalidated %d stale derived rows for missing/malformed calls.", invalidated)

    if use_roic:
        roic = fetch_roic_transcripts(context, tickers=tickers, missing=missing, since=recent_since)
        missing = remaining_after(missing, roic.filled)  # fool only gets the remainder

    build_transcript_index_by_ticker(context, tickers=tickers, missing=missing, since=recent_since)

    if use_global_crawl:
        build_transcript_index(context, tickers=tickers, history_years=mf_history_years)

    download_transcripts(context, tickers=tickers, limit=limit)


def ingest_all_earnings_calls(context: Context, tickers: list[str] | None = None, force: bool = False) -> int:
    """Ingest stage for every downloaded source (cached HF parquet, then cached Motley Fool HTML) -> `earnings_call_sections`.

    Returns total rows upserted and records the run. The HF leg is non-fatal so recent quarters already on disk
    as Fool HTML still land."""
    saved = 0
    try:
        saved += ingest_hf_transcripts(context, tickers=tickers, force=force)  # cached parquet
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "HF backbone ingest failed (%s: %s) -> continuing to the Motley Fool ingest so the recent quarters already on disk still land.",
            type(exc).__name__,
            exc,
        )
    saved += ingest_earnings_calls(context, tickers=tickers, force=force)  # cached MF HTML
    record_run(context, Tables.earnings_call_sections, len(tickers) if tickers else 0, saved)
    return saved


def fetch_earnings_calls(
    context: Context,
    tickers: list[str] | None = None,
    limit: int | None = None,
    recent_since: str = "2025-01-01",
    use_global_crawl: bool = False,
    mf_history_years: float = 2.0,
) -> int:
    """Full transcript pipeline: `download_earnings_calls` then `ingest_all_earnings_calls` (Airflow runs them as separate tasks)."""
    download_earnings_calls(
        context, tickers=tickers, limit=limit, recent_since=recent_since, use_global_crawl=use_global_crawl, mf_history_years=mf_history_years
    )
    return ingest_all_earnings_calls(context, tickers=tickers)  # records the run itself
