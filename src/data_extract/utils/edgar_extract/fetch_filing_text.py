"""10-K Item 1 (Business), Item 1A (Risk Factors) and Item 7 (MD&A), and 10-Q Item 2 (MD&A), as raw text.

Writes `sec_filing_text`, one row per (ticker, accession, section). Primary path: edgartools' typed
`TenK`/`TenQ` section parser; fallback: a regex carve over the filing text, used only for a section the
structured parse missed or returned as a sub-`FILING_TEXT_MIN_CHARS` stub. A transient SEC failure
raises and fails the filing; a parse failure falls back as above. An expected section the filing lacks
(Items 1, 1A and 7 for an original 10-K, MD&A for any other form) is stored as an empty-text row, so a
filing is done once every expected section has a row, and one with a missing row is read again.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import partial
from typing import Any

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, filing_obj, filing_text
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, FilingStamp, parse_filing_rows, save_key
from src.data_extract.utils.edgar_extract.item_carve import ITEM_SEP, CrossRefCues, carve_spans, item_heading
from src.data_store.schema import Table, Tables

FILING_SECTION_BUSINESS = "business"  # 10-K Item 1
FILING_SECTION_RISK = "risk_factors"  # 10-K Item 1A
FILING_SECTION_MDA = "mda"  # 10-K Item 7 / 10-Q Item 2
FILING_TEXT_MIN_CHARS = 1500  # below this a "section" is a TOC/cross-ref stub
_TEN_K_SECTIONS = (FILING_SECTION_BUSINESS, FILING_SECTION_RISK, FILING_SECTION_MDA)

_MAX_CHARS = 300_000  # cap a stored section (risk factors can be enormous)

# Item markers are content-anchored; the "management's" apostrophe is matched with \W because EDGAR encodes it several ways.
# Business and Risk Factors match headings that start a line: a cross-reference ("described in Item 1A") sits inside a
# sentence. Business ("Item 1.", "Items 1 and 2.", "Description of Business") ends at the next Item 1A/1B/2/3 heading,
# whatever its caption (filers before 2005 have no Item 1A).
_LINE_SEP = rf"(?:{ITEM_SEP}\|?{ITEM_SEP})"  # "Item 1 | Business" (table-cell headings) besides "Item 1. -"
_BUSINESS_START = re.compile(
    rf"^[ \t]*items?{_LINE_SEP}1\b(?:{_LINE_SEP}(?:and|&){_LINE_SEP}(?:items?{_LINE_SEP})?2\b)?{_LINE_SEP}"
    r"(?:description\s+of\s+(?:the\s+)?)?business[^\n]*",
    re.I | re.M,
)
_BUSINESS_END = re.compile(rf"^[ \t]*item{_LINE_SEP}(?:1a|1b|2|3)\b", re.I | re.M)
# Risk Factors (10-K Item 1A) ends at Item 1B, Item 1C (for filers that omit 1B) or Item 2.
_RISK_START = re.compile(rf"^[ \t]*item{_LINE_SEP}1a\b{_LINE_SEP}risk\s+factors[^\n]*", re.I | re.M)
_RISK_END = re.compile(rf"^[ \t]*(?:item{_LINE_SEP}(?:1b|1c|2)\b|unresolved\s+staff\s+comments)", re.I | re.M)
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
    end=re.compile(r"\b(see|refer|conjunction|pursuant|incorporat|with|under|within|in)\b\W{0,4}$", re.I),
)


def _best_span(
    text: str,
    start_re: re.Pattern,
    min_chars: int,
    end_primary: re.Pattern,
    end_fallback: re.Pattern | None = None,
    *,
    require_end: bool = False,
) -> str | None:
    """The LONGEST body between a real start HEADING and the next real end HEADING (cross-references
    skipped at both ends; `end_fallback` only when no primary end follows the start), compared on
    whitespace-stripped bounds. `require_end` drops a span that runs to the end of the text. None if no
    span reaches `min_chars`."""
    best_s = best_e = 0
    fallback = (end_fallback,) if end_fallback is not None else ()
    for s, e in carve_spans(text, start_re, (end_primary,), fallback_end_res=fallback, cross_refs=_CROSS_REFS):
        if require_end and e == len(text):
            continue
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s > best_e - best_s:
            best_s, best_e = s, e
    return text[best_s:best_e][:_MAX_CHARS] if best_e - best_s >= min_chars else None


def extract_item_sections(text: str, form: str) -> dict[str, str]:
    """{section: body_text} carved from a filing's plain text (the fallback path).

    10-K -> Business (Item 1) + Risk Factors (Item 1A) + MD&A (Item 7); 10-Q -> MD&A (Item 2) only. MD&A tries the
    form's item anchor first, then the standalone title."""
    if not text or len(text) < FILING_TEXT_MIN_CHARS:
        return {}
    is_10k = str(form).upper().startswith("10-K")
    mda_start = _MDA_START_10K if is_10k else _MDA_START_10Q
    end_pri = _MDA_END_10K_PRI if is_10k else _MDA_END_10Q_PRI
    end_fall = _MDA_END_10K_FALL if is_10k else _MDA_END_10Q_FALL

    out: dict[str, str] = {}
    if is_10k:  # substantive annual business description and risk factors
        business = _best_span(text, _BUSINESS_START, FILING_TEXT_MIN_CHARS, _BUSINESS_END, require_end=True)
        if business:
            out[FILING_SECTION_BUSINESS] = business
        rf = _best_span(text, _RISK_START, FILING_TEXT_MIN_CHARS, _RISK_END, require_end=True)
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
            business = obj.business
            if business and len(business) >= FILING_TEXT_MIN_CHARS:
                out[FILING_SECTION_BUSINESS] = business[:_MAX_CHARS]
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


def _filing_sections(filing: Any) -> dict[str, str]:
    """Structured sections from the parsed filing, with the regex carve filling whichever section it missed. A 10-K's
    Business takes the carve whenever it finds one: edgartools' Item 1 can stop at a sub-heading inside the item."""
    form = filing.form
    is_10k = str(form).upper().startswith("10-K")
    needed = set(_TEN_K_SECTIONS) if is_10k else {FILING_SECTION_MDA}
    try:
        obj = filing_obj(filing)
    except ParseFailureError:
        obj = None
    sections = _structured_sections(obj, form) if obj is not None else {}
    missing = needed - (sections.keys() - {FILING_SECTION_BUSINESS})
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


def expected_sections(form: object) -> tuple[str, ...]:
    """The sections a filing of `form` must hold a row for: Items 1, 1A and 7 for an original 10-K, MD&A otherwise."""
    return _TEN_K_SECTIONS if str(form).upper() == "10-K" else (FILING_SECTION_MDA,)


def _filing_rows(ticker: str, stamp: FilingStamp) -> list[dict]:
    """One row per section read, plus an empty-text row (NULL `text`, 0 words) for each expected section the filing
    lacks: that row records the attempt, so the filing is not read again."""
    sections: dict[str, str | None] = {**dict.fromkeys(expected_sections(stamp.form)), **_filing_sections(stamp.filing)}
    stamped = stamp.columns(Tables.filing_risk_text)
    return [
        {
            # The table's registered forms use SPLIT, so each row's registrant is unambiguous.
            "ticker": ticker,
            **stamped,
            "section": section,
            "text": body,
            "n_words": len(body.split()) if body else 0,
        }
        for section, body in sections.items()
    ]


def filing_text_done(context: Context, keys: Sequence[str], runtime_floor: pd.Timestamp | None) -> dict[str, set[str]]:
    """Accessions holding a row, text or empty, for every section their form expects (`expected_sections`; an
    accession is a 10-K when any of its rows says so). An empty-filing marker is no section, so its filing is read
    once more."""
    rows = context.store.load(
        Tables.filing_risk_text, columns=["ticker", "accession_number", "form", "section"], where={"ticker": list(keys)}, optional=True
    )
    done: dict[str, set[str]] = {key: set() for key in keys}
    if rows is None or rows.empty:
        return done
    section = rows["section"].where(rows["section"].isin(_TEN_K_SECTIONS))
    per_filing = (
        rows.assign(ten_k=rows["form"].astype("string").str.upper().eq("10-K").fillna(False), section=section, mda=section.eq(FILING_SECTION_MDA))
        .groupby(["ticker", "accession_number"])
        .agg(ten_k=("ten_k", "any"), n_sections=("section", "nunique"), mda=("mda", "any"))
    )
    complete = per_filing["n_sections"].eq(len(_TEN_K_SECTIONS)).where(per_filing["ten_k"], per_filing["mda"])
    for ticker, accessions in per_filing[complete.astype(bool)].reset_index().groupby("ticker")["accession_number"]:
        done[str(ticker)] = set(accessions.astype(str))
    return done


def save_filing_text(context: Context, fetch: EdgarFetch, ticker: str, frames: dict[Table, list[pd.DataFrame]]) -> tuple[dict[Table, int], bool]:
    """`save_key` without the empty-text rows whose section is already stored: a re-read never blanks stored text."""
    table = Tables.filing_risk_text
    if not frames.get(table):
        return save_key(context, fetch, ticker, frames)
    df = pd.concat(frames[table], ignore_index=True)
    empty = df["text"].isna()
    stored = (
        context.store.load(
            table,
            columns=["accession_number", "section"],
            where={"ticker": ticker, "accession_number": sorted(df.loc[empty, "accession_number"].astype(str).unique())},
            optional=True,
        )
        if empty.any()
        else None
    )
    if stored is not None and not stored.empty:
        taken = pd.MultiIndex.from_frame(stored.astype(str))
        empty &= pd.MultiIndex.from_frame(df[["accession_number", "section"]].astype(str)).isin(taken)
        df = df[~empty]
    return save_key(context, fetch, ticker, {**frames, table: [df]})


FILING_TEXT_FETCH = EdgarFetch(
    tables=(Tables.filing_risk_text,),
    parse=partial(parse_filing_rows, table=Tables.filing_risk_text, row_fn=_filing_rows),
    done_accessions=filing_text_done,
    save=save_filing_text,
)
