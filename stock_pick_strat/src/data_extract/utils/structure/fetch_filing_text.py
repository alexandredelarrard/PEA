"""
fetch_filing_text.py  (src/data_extract/utils/structure/fetch_filing_text.py)
-----------------------------------------------------------------------------
10-K narrative sections — Item 1A (Risk Factors) + Item 7 (MD&A) — extracted to raw text in
`filing_risk_text`, for the downstream embedding/drift feature layer (YoY risk-factor additions,
MD&A tone drift — reusing the notes-embedding machinery).

Per ticker: list 10-Ks via the shared `list_filings` (per-ticker `since` -> DAG-fast incremental),
download each 10-K's primary HTML (cached to data/sec_filings_text/ BEFORE the DB), section-carve
Item 1A + Item 7 (`extract_item_sections`), and upsert one row per (ticker, accession, section).
Deduped by accession so a re-run never re-downloads a stored 10-K.

Section carving (the fiddly part, like the DEF 14A finder): each item's BODY is the LONGEST text
span between its start heading and the next section marker — TOC entries are skipped because the
next marker sits only a few chars away (span < FILING_TEXT_MIN_CHARS).
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from tqdm import tqdm

from src.constants.constants import (
    FILING_SECTION_MDA, FILING_SECTION_RISK, FILING_TEXT_CACHE_DIR, FILING_TEXT_FORMS,
    FILING_TEXT_MIN_CHARS, FILING_TEXT_TABLE,
)
from src.context import Context
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.edgar_fillings import list_filings
from src.data_extract.utils.common.sec_utils import load_cik_mapping, sec_get

logger = logging.getLogger(__name__)
_TABLE = FILING_TEXT_TABLE
_MAX_CHARS = 300_000                       # cap a stored section (risk factors can be enormous)

# --- Item markers (content-anchored; `N\b` never matches "NA" — no word boundary in "7a"/"1a").
# The apostrophe in "management's" is matched with \W (any non-word), NOT a literal quote: EDGAR HTML
# uses several encodings for it (U+2019, Win-1252 0x92, mojibake), so a fixed quote class misses some
# filers -> 0 MD&A. MD&A lives in DIFFERENT items per form: 10-K Item 7, 10-Q Item 2. ---
_SEP = r"[\.\:\)\s–—-]{0,8}"
# Risk Factors — 10-K Item 1A (Part I). Ends at the next Part I item: Item 1B (Unresolved Staff
# Comments), Item 1C (Cybersecurity — mandatory since Dec-2023; filers that OMIT 1B end here), or
# Item 2 (Properties). Without 1C/the title, a filer lacking Item 1B had NO end -> the span over-ran
# to end-of-document (the PTC bug).
_RISK_START = re.compile(rf"item{_SEP}1a\b{_SEP}risk\s+factors", re.I)
_RISK_END = re.compile(
    rf"item{_SEP}1b\b|item{_SEP}1c\b|unresolved\s+staff\s+comments|item{_SEP}2\b{_SEP}propert", re.I)
# MD&A start — 10-K "Item 7 + management", 10-Q "Item 2 + management" (stops before the apostrophe).
_MDA_START_10K = re.compile(rf"item{_SEP}7\b{_SEP}management", re.I)
_MDA_START_10Q = re.compile(rf"item{_SEP}2\b{_SEP}management", re.I)
# FALLBACK start (both forms): the standalone MD&A TITLE, for filers that print it WITHOUT the item
# prefix. Allows an optional inserted word — combined multi-registrant filers (e.g. Entergy) title it
# "Management's FINANCIAL Discussion and Analysis".
_MDA_START_ALT = re.compile(
    r"management\W{0,3}(?:s\W{1,3})?(?:financial\W{1,3})?discussion\W{1,3}and\W{1,3}analysis", re.I)
# MD&A end — PREFER the true next section (10-K: Item 7A / 10-Q: Item 3, both "Quantitative and
# Qualitative Disclosures") over the financial-statements item, so an intro cross-ref to the latter
# can't truncate the body. Fallbacks: 10-K Item 8 (Financial Statements); 10-Q Item 4 (Controls) /
# Part II.
_MDA_END_10K_PRI = re.compile(rf"item{_SEP}7a\b|quantitative\s+and\s+qualitative", re.I)
_MDA_END_10K_FALL = re.compile(rf"item{_SEP}8\b{_SEP}financial\s+statements", re.I)
_MDA_END_10Q_PRI = re.compile(rf"item{_SEP}3\b{_SEP}quantitative|quantitative\s+and\s+qualitative", re.I)
_MDA_END_10Q_FALL = re.compile(rf"item{_SEP}4\b{_SEP}controls|part{_SEP}ii\b", re.I)
# cross-reference cues right before an item marker -> a POINTER, not a real heading. START uses a
# STRICT set (only unambiguous pointers) — broad prepositions like "under"/"with" legitimately
# precede a real heading ("risks described under Item 1A. Risk Factors <body>"), so skipping on them
# drops real sections (the PTC bug). END uses a BROADER set so an intro cross-ref ("read in
# conjunction WITH Item 8, Financial Statements") can't truncate the body to a stub.
_XREF_START = re.compile(r"\b(see|refer|conjunction|pursuant|incorporat)\b\W{0,4}$", re.I)
_XREF_END = re.compile(r"\b(see|refer|conjunction|pursuant|incorporat|with|under|within)\b\W{0,4}$", re.I)


def _first_end(low: str, s: int, end_re: re.Pattern) -> int | None:
    """First end-marker start after `s` that is NOT a cross-reference (skip pointers), else None."""
    for m in end_re.finditer(low):
        x = m.start()
        if x > s and not _XREF_END.search(low[max(0, x - 25):x]):
            return x
    return None


def _best_span(text: str, low: str, start_re: re.Pattern, min_chars: int,
               end_primary: re.Pattern, end_fallback: re.Pattern | None = None) -> str | None:
    """The LONGEST body between a real start HEADING and the next real end HEADING. The start skips
    cross-references; the end prefers `end_primary` (the true next section) and only uses
    `end_fallback` when no primary end follows the start. None if no span reaches `min_chars`."""
    best = ""
    for m in start_re.finditer(low):
        if _XREF_START.search(low[max(0, m.start() - 25):m.start()]):
            continue                                   # a pointer to the section, not the heading
        s = m.end()
        e = _first_end(low, s, end_primary)
        if e is None and end_fallback is not None:
            e = _first_end(low, s, end_fallback)
        if e is None:
            e = len(text)
        span = text[s:e].strip()
        if len(span) >= min_chars and len(span) > len(best):
            best = span
    return best[:_MAX_CHARS] if best else None


def extract_item_sections(text: str, form: str) -> dict[str, str]:
    """{section: body_text} for the sections in a filing's plain text, FORM-AWARE:
      * 10-K -> Risk Factors (Item 1A) + MD&A (Item 7).
      * 10-Q -> MD&A (Item 2) only (Part II Item 1A risk factors are usually 'no material change').
    MD&A uses a two-tier strategy: the primary item anchor for the form, then — if that yields
    nothing (a filer that prints the MD&A title without the item prefix) — a FALLBACK on the
    standalone 'Management's Discussion and Analysis' title. The end prefers the true next section
    (Item 7A / Item 3, Quantitative & Qualitative) over the financial-statements item."""
    if not text or len(text) < FILING_TEXT_MIN_CHARS:
        return {}
    low = text.lower()
    is_10k = str(form).upper().startswith("10-K")
    mda_start = _MDA_START_10K if is_10k else _MDA_START_10Q
    end_pri = _MDA_END_10K_PRI if is_10k else _MDA_END_10Q_PRI
    end_fall = _MDA_END_10K_FALL if is_10k else _MDA_END_10Q_FALL

    out: dict[str, str] = {}
    if is_10k:                                         # substantive annual risk factors
        rf = _best_span(text, low, _RISK_START, FILING_TEXT_MIN_CHARS, _RISK_END)
        if rf:
            out[FILING_SECTION_RISK] = rf
    md = _best_span(text, low, mda_start, FILING_TEXT_MIN_CHARS, end_pri, end_fall)
    if md is None:                                     # fallback: title-only MD&A heading
        md = _best_span(text, low, _MDA_START_ALT, FILING_TEXT_MIN_CHARS, end_pri, end_fall)
    if md:
        out[FILING_SECTION_MDA] = md
    return out


def _seen(context: Context) -> tuple[set[str], dict[str, pd.Timestamp]]:
    try:
        df = context.store.load(_TABLE, columns=["ticker", "accession_number", "filed"])
    except Exception:
        return set(), {}
    if df is None or df.empty:
        return set(), {}
    seen = set(df["accession_number"].dropna().astype(str))
    d = df.copy()
    d["filed"] = pd.to_datetime(d["filed"], errors="coerce")
    last = d.dropna(subset=["filed"]).groupby("ticker")["filed"].max().to_dict()
    return seen, last


def fetch_filing_text(context: Context, tickers: list[str], years: int | None = None) -> pd.DataFrame:
    """Build/refresh the filing-text table (10-K Item 1A + Item 7, 10-Q Item 2 MD&A), one ticker at
    a time (upsert per ticker). Returns the full table."""
    years = int(years if years is not None else context.config.data_extract.years_history)
    cache_dir = context.paths["DATA_STORE"] / FILING_TEXT_CACHE_DIR
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    seen, last_by_ticker = _seen(context)
    total_rows, touched, no_sections = 0, 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="10-K/10-Q text"):
        ticker, cik, company = r["ticker"], r["cik"], r.get("company_name", "")
        try:
            filings = list_filings(cik, FILING_TEXT_FORMS, years, company,
                                   since=last_by_ticker.get(ticker))
        except Exception as e:                          # noqa: BLE001
            context.log.warning("%s: 10-K list failed (%s)", ticker, e)
            continue
        new = filings[~filings["accession_number"].astype(str).isin(seen)] if not filings.empty \
            else filings
        rows: list[dict] = []
        for _, f in new.iterrows():
            acc = str(f["accession_number"])
            html = _download_cached(context, cache_dir, cik, acc, f["doc_url"])
            if not html:
                continue
            sections = extract_item_sections(html_to_text(html), f.get("form", ""))
            if not sections:
                no_sections += 1
            for section, body in sections.items():
                rows.append({
                    "ticker": ticker, "cik": cik, "accession_number": acc,
                    "form": str(f.get("form", "")),
                    "filed": pd.Timestamp(f["filing_date"]).normalize(),
                    "period_of_report": f.get("period_of_report"),
                    "section": section, "text": body, "n_words": len(body.split()),
                })
            seen.add(acc)
        if rows:
            total_rows += context.store.save(_TABLE, pd.DataFrame(rows))
            touched += 1

    context.log.info("10-K/10-Q text: +%d section rows across %d tickers (%d filings yielded no "
                     "section) -> '%s'", total_rows, touched, no_sections, _TABLE)
    return context.store.load(_TABLE)


def _download_cached(context: Context, cache_dir, cik: str, accession: str, url: str) -> str | None:
    """Fetch a 10-K's primary HTML, caching the RAW file to disk before parsing (reproducible /
    offline re-parse). Returns the HTML text (None on failure)."""
    dest = cache_dir / str(cik) / f"{accession}.html"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest.read_text(encoding="utf-8", errors="replace")
    try:
        html = sec_get(url).text
    except Exception as e:                              # noqa: BLE001
        context.log.warning("10-K doc fetch failed %s (%s)", url, e)
        return None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(html, encoding="utf-8")
    except Exception:                                   # caching best-effort
        pass
    return html
