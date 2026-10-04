"""10-K Item 1A (Risk Factors) + Item 7 (MD&A) and 10-Q Item 2 (MD&A) as raw text.

Writes `sec_filing_text`, one row per (ticker, accession, section). Primary path: edgartools' typed
`TenK`/`TenQ` section parser; fallback: a regex carve over the filing text, used only for a section the
structured parse missed or returned as a sub-`FILING_TEXT_MIN_CHARS` stub. A transient SEC failure
raises and fails the filing; a parse failure falls back as above; a filing with no readable section
becomes an empty-filing marker.
"""

from __future__ import annotations

import re
from functools import partial

from src.data_extract.utils.common.edgar_driver import EdgarFetch, FilingStamp, parse_filing_rows
from src.data_extract.utils.common.item_carve import ITEM_SEP, CrossRefCues, carve_spans, item_heading
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, filing_obj, filing_text
from src.data_store.schema import Tables

FILING_TEXT_FORMS = ["10-K", "10-Q"]
FILING_SECTION_RISK = "risk_factors"  # 10-K Item 1A
FILING_SECTION_MDA = "mda"  # 10-K Item 7 / 10-Q Item 2
FILING_TEXT_MIN_CHARS = 1500  # below this a "section" is a TOC/cross-ref stub

_MAX_CHARS = 300_000  # cap a stored section (risk factors can be enormous)
_COLS = ["ticker", "cik", "accession_number", "form", "filed", "period_of_report", "section", "text", "n_words"]

# Item markers are content-anchored; the "management's" apostrophe is matched with \W because EDGAR encodes it several ways.
# Risk Factors (10-K Item 1A) ends at Item 1B, Item 1C (for filers that omit 1B) or Item 2 (Properties).
_RISK_START = item_heading("1a", r"risk\s+factors")
_RISK_END = re.compile(rf"item{ITEM_SEP}1b\b|item{ITEM_SEP}1c\b|unresolved\s+staff\s+comments|item{ITEM_SEP}2\b{ITEM_SEP}propert", re.I)
# MD&A start: 10-K Item 7, 10-Q Item 2, each followed by "management".
_MDA_START_10K = item_heading(7, "management")
_MDA_START_10Q = item_heading(2, "management")
# Fallback start: the standalone MD&A title without its item prefix, allowing an inserted "financial".
_MDA_START_ALT = re.compile(r"management\W{0,3}(?:s\W{1,3})?(?:financial\W{1,3})?discussion\W{1,3}and\W{1,3}analysis", re.I)
# MD&A end prefers the next section (10-K Item 7A / 10-Q Item 3); fallbacks are 10-K Item 8 and 10-Q Item 4 / Part II.
_MDA_END_10K_PRI = re.compile(rf"item{ITEM_SEP}7a\b|quantitative\s+and\s+qualitative", re.I)
_MDA_END_10K_FALL = item_heading(8, r"financial\s+statements")
_MDA_END_10Q_PRI = re.compile(rf"item{ITEM_SEP}3\b{ITEM_SEP}quantitative|quantitative\s+and\s+qualitative", re.I)
_MDA_END_10Q_FALL = re.compile(rf"item{ITEM_SEP}4\b{ITEM_SEP}controls|part{ITEM_SEP}ii\b", re.I)
# Cues that make an item marker a pointer, not a heading: strict at START ("under" can precede a real heading),
# broader at END so an intro cross-reference cannot truncate the body.
_CROSS_REFS = CrossRefCues(
    start=re.compile(r"\b(see|refer|conjunction|pursuant|incorporat)\b\W{0,4}$", re.I),
    end=re.compile(r"\b(see|refer|conjunction|pursuant|incorporat|with|under|within)\b\W{0,4}$", re.I),
)


def _best_span(text: str, start_re: re.Pattern, min_chars: int, end_primary: re.Pattern, end_fallback: re.Pattern | None = None) -> str | None:
    """The LONGEST body between a real start HEADING and the next real end HEADING (cross-references
    skipped at both ends; `end_fallback` only when no primary end follows the start), compared on
    whitespace-stripped bounds. None if no span reaches `min_chars`."""
    best_s = best_e = 0
    fallback = (end_fallback,) if end_fallback is not None else ()
    for s, e in carve_spans(text, start_re, (end_primary,), fallback_end_res=fallback, cross_refs=_CROSS_REFS):
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s > best_e - best_s:
            best_s, best_e = s, e
    return text[best_s:best_e][:_MAX_CHARS] if best_e - best_s >= min_chars else None


def extract_item_sections(text: str, form: str) -> dict[str, str]:
    """{section: body_text} carved from a filing's plain text (the fallback path).

    10-K -> Risk Factors (Item 1A) + MD&A (Item 7); 10-Q -> MD&A (Item 2) only. MD&A tries the
    form's item anchor first, then the standalone title."""
    if not text or len(text) < FILING_TEXT_MIN_CHARS:
        return {}
    is_10k = str(form).upper().startswith("10-K")
    mda_start = _MDA_START_10K if is_10k else _MDA_START_10Q
    end_pri = _MDA_END_10K_PRI if is_10k else _MDA_END_10Q_PRI
    end_fall = _MDA_END_10K_FALL if is_10k else _MDA_END_10Q_FALL

    out: dict[str, str] = {}
    if is_10k:  # substantive annual risk factors
        rf = _best_span(text, _RISK_START, FILING_TEXT_MIN_CHARS, _RISK_END)
        if rf:
            out[FILING_SECTION_RISK] = rf
    md = _best_span(text, mda_start, FILING_TEXT_MIN_CHARS, end_pri, end_fall)
    if md is None:  # fallback: title-only MD&A heading
        md = _best_span(text, _MDA_START_ALT, FILING_TEXT_MIN_CHARS, end_pri, end_fall)
    if md:
        out[FILING_SECTION_MDA] = md
    return out


def _structured_sections(obj, form: str) -> dict[str, str]:
    """Section text from edgartools' `TenK`/`TenQ` parser; a result under `FILING_TEXT_MIN_CHARS` is dropped as a stub."""
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
    except Exception:  # noqa: BLE001 -- best-effort only
        return out
    if mda and len(mda) >= FILING_TEXT_MIN_CHARS:
        out[FILING_SECTION_MDA] = mda[:_MAX_CHARS]
    return out


def _filing_sections(filing) -> dict[str, str]:
    """Structured sections from the parsed filing, with the regex carve filling whichever section it missed."""
    form = filing.form
    needed = {FILING_SECTION_RISK, FILING_SECTION_MDA} if str(form).upper().startswith("10-K") else {FILING_SECTION_MDA}
    try:
        obj = filing_obj(filing)
    except ParseFailureError:
        obj = None
    sections = _structured_sections(obj, form) if obj is not None else {}
    missing = needed - sections.keys()
    if not missing:
        return sections
    try:
        text = filing_text(filing)
    except TransientReadError:
        raise
    except Exception:  # noqa: BLE001 -- best-effort only
        text = None
    if not text:
        return sections
    fallback = extract_item_sections(text, form)
    for k in missing:
        if k in fallback:
            sections[k] = fallback[k]
    return sections


def _filing_rows(ticker: str, stamp: FilingStamp) -> list[dict]:
    """One row per extracted section of the filing; none when no section could be read."""
    filed = stamp.filed.normalize()
    return [
        {
            # The CIK that filed it: `FILING_TEXT_FORMS` is SPLIT, so each row's registrant is unambiguous.
            "ticker": ticker,
            "cik": stamp.cik,
            "accession_number": stamp.accession_number,
            "form": str(stamp.form),
            "filed": filed,
            "period_of_report": stamp.period_of_report,
            "section": section,
            "text": body,
            "n_words": len(body.split()),
        }
        for section, body in _filing_sections(stamp.filing).items()
    ]


FILING_TEXT_FETCH = EdgarFetch(
    desc="10-K/10-Q text (edgartools)",
    tables=(Tables.filing_risk_text,),
    forms=tuple(FILING_TEXT_FORMS),
    parse=partial(parse_filing_rows, table=Tables.filing_risk_text, columns=_COLS, row_fn=_filing_rows),
)
