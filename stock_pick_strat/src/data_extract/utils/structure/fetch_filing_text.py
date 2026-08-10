"""
fetch_filing_text.py  (src/data_extract/utils/structure/fetch_filing_text.py)
-----------------------------------------------------------------------------
10-K narrative sections — Item 1A (Risk Factors) + Item 7 (MD&A) — and 10-Q MD&A
(Item 2), extracted to raw text in `filing_risk_text` via `edgartools` (mirrors
`fetch_8k_edgar.py` / `fetch_13d_edgar.py`), for the downstream embedding/drift
feature layer (YoY risk-factor additions, MD&A tone drift — reusing the
notes-embedding machinery).

Data Grain:
- One row per (ticker, accession_number, section).

Extraction strategy (PRIMARY + fallback, same pattern as `fetch_13d_edgar.py`'s
Item 3/4/5/6 narrative):
  * PRIMARY — edgartools' own `TenK`/`TenQ` structured section parser
    (`filing.obj()`): `.risk_factors` / `.management_discussion` for 10-K,
    `["Part I, Item 2"]` for 10-Q. Already handles cross-reference-index filings
    (GE-style), bold-paragraph headings and table-cell layouts.
  * FALLBACK — a hardened regex carve (`extract_item_sections`) over the
    filing's plain text (`filing.text()`), used only when the structured parse
    returns nothing (or a bare heading/cross-reference stub under
    `FILING_TEXT_MIN_CHARS`). Covers edge cases found empirically across the
    archive: apostrophe encoding variants (U+2019 / Win-1252 / mojibake),
    filers that print the MD&A title without the item prefix, filers missing
    Item 1B/1C.

Deduped by accession so a re-run never re-parses a stored 10-K/10-Q.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
from edgar import Company

from src.data_store.schema import Tables
from src.constants.constants import (FILING_SECTION_MDA, FILING_SECTION_RISK, FILING_TEXT_FORMS, FILING_TEXT_MIN_CHARS)
from src.context import Context
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import _configure_identity

logger = logging.getLogger(__name__)
_MAX_CHARS = 300_000                       # cap a stored section (risk factors can be enormous)
_COLS = ["ticker", "cik", "accession_number", "form", "filed", "period_of_report",
        "section", "text", "n_words"]

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
    FALLBACK path only (see module docstring) — called when edgartools' own structured
    parse (`.risk_factors` / `.management_discussion`) returns nothing or a stub. MD&A uses a
    two-tier strategy: the primary item anchor for the form, then — if that yields nothing (a
    filer that prints the MD&A title without the item prefix) — a FALLBACK on the standalone
    'Management's Discussion and Analysis' title. The end prefers the true next section
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


def _structured_sections(obj, form: str) -> dict[str, str]:
    """Section text from edgartools' own `TenK`/`TenQ` parser (PRIMARY path — see module
    docstring). A result under `FILING_TEXT_MIN_CHARS` is a bare heading/cross-reference stub,
    discarded so the caller falls through to the regex carve."""
    out: dict[str, str] = {}
    is_10k = str(form).upper().startswith("10-K")
    try:
        if is_10k:
            rf = obj.risk_factors
            if rf and len(rf) >= FILING_TEXT_MIN_CHARS:
                out[FILING_SECTION_RISK] = rf[:_MAX_CHARS]
            mda = obj.management_discussion
        else:
            mda = obj["Part I, Item 2"]
    except Exception:                                   # noqa: BLE001 -- best-effort only
        return out
    if mda and len(mda) >= FILING_TEXT_MIN_CHARS:
        out[FILING_SECTION_MDA] = mda[:_MAX_CHARS]
    return out


def _filing_sections(filing) -> dict[str, str]:
    """PRIMARY (structured `filing.obj()`) + FALLBACK (regex carve over `filing.text()` for
    whichever section the structured parse missed) — see module docstring."""
    form = filing.form
    needed = {FILING_SECTION_RISK, FILING_SECTION_MDA} if str(form).upper().startswith("10-K") \
        else {FILING_SECTION_MDA}
    try:
        obj = filing.obj()
    except Exception:                                   # noqa: BLE001 -- best-effort only
        obj = None
    sections = _structured_sections(obj, form) if obj is not None else {}
    missing = needed - sections.keys()
    if missing:
        try:
            text = filing.text()
        except Exception:                               # noqa: BLE001 -- best-effort only
            text = None
        if text:
            fallback = extract_item_sections(text, form)
            for k in missing:
                if k in fallback:
                    sections[k] = fallback[k]
    return sections


def _seen(context: Context) -> tuple[set[str], dict[str, pd.Timestamp]]:
    try:
        df = context.store.load(Tables.filing_risk_text, columns=["ticker", "accession_number", "filed"])
    except Exception:
        return set(), {}
    if df is None or df.empty:
        return set(), {}
    seen = set(df["accession_number"].dropna().astype(str))
    d = df.copy()
    d["filed"] = pd.to_datetime(d["filed"], errors="coerce")
    last = d.dropna(subset=["filed"]).groupby("ticker")["filed"].max().to_dict()
    return seen, last


def build_ticker_filing_text(ticker: str, cik: str, since: pd.Timestamp | None = None,
                             done_accessions: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Walks `Company(ticker).get_filings(form=FILING_TEXT_FORMS)`, skips accessions already in
    `done_accessions` or filed before `since`, and builds one row per (filing, section) via
    `_filing_sections`."""
    company = Company(ticker)
    filings = company.get_filings(form=list(FILING_TEXT_FORMS))
    sorted_filings = sorted(filings, key=lambda f: f.filing_date)
    if since is not None:
        sorted_filings = [f for f in sorted_filings if pd.Timestamp(f.filing_date) >= since]

    rows: list[dict] = []
    for f in sorted_filings:
        if f.accession_number in done_accessions:
            continue
        for section, body in _filing_sections(f).items():
            rows.append({
                "ticker": ticker, "cik": cik, "accession_number": f.accession_number,
                "form": str(f.form), "filed": pd.Timestamp(f.filing_date).normalize(),
                "period_of_report": f.period_of_report,
                "section": section, "text": body, "n_words": len(body.split()),
            })
    return pd.DataFrame(rows, columns=_COLS)


def fetch_filing_text(context: Context, tickers: list[str], years: int | None = None) -> pd.DataFrame:
    """Public entry point (mirrors `fetch_8k_edgar`'s conventions): per-ticker try/except so one
    bad ticker cannot abort the batch, incremental via `_seen` (dedup by accession + per-ticker
    resume cutoff), scoped by `years` (falls back to `data_extract.years_history`). Tickers are
    walked CONCURRENTLY on a thread pool (`run_per_ticker`) -- see parallel_fetch.py's module
    docstring."""
    _configure_identity()
    years = int(years if years is not None else context.config.data_extract.years_history)
    since = pd.Timestamp.today() - pd.DateOffset(years=years)
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    seen, last_by_ticker = _seen(context)

    def _worker(ticker: str, cik: str) -> tuple[int, bool]:
        try:
            ticker_since = max(since, last_by_ticker[ticker]) if ticker in last_by_ticker else since
            out = build_ticker_filing_text(ticker, cik, since=ticker_since, done_accessions=seen)
        except Exception as e:                          # noqa: BLE001
            context.log.warning("fetch_filing_text: %s failed (%s)", ticker, e)
            return 0, False
        if not out.empty:
            context.store.save(Tables.filing_risk_text, out)
        return len(out), True

    results = run_per_ticker(cik_map, _worker, desc="10-K/10-Q text (edgartools)")
    total_rows = sum(n for n, _ in results)
    failed = sum(1 for _, ok in results if not ok)

    context.log.info("fetch_filing_text: +%d section rows across %d/%d ticker(s) (%d failed) -> '%s'",
                     total_rows, len(results), len(cik_map), failed, Tables.filing_risk_text)
    record_run(context, Tables.filing_risk_text, len(cik_map), total_rows)
    return context.store.load(Tables.filing_risk_text)
