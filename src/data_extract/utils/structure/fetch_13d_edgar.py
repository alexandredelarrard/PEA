"""
fetch_13d_edgar.py (src/data_extract/utils/structure/fetch_13d_edgar.py)
---------------------------------------------------------------------------
SC 13D/13D-A activist filings via edgartools (`filing.obj()`) into two tables:
`sec_13d`, one row per (ticker, accession, rp_seq) -- co-filers are kept by
0-based position because a reporting person often has no CIK -- and
`sec_13d_transactions`, one row per disclosed trade (Item 5(c) 60-day log, an
independent grain).

EDGAR has TWO 13D eras, and the split runs through everything below. At the
structured-XML mandate the form string itself changed -- "SC 13D" through
2024-12-16, "SCHEDULE 13D" from 2024-12-17 -- and `get_filings(form=...)` matches
EXACTLY, so `SEC_13D_FORMS` must list both pairs or the table simply stops
(measured: 461 filings across 91 tickers went missing that way).

Four properties the parsing depends on:
- `has_structured_data` means "this filing has XML", NOT "this filing is modern":
  it is False for essentially every pre-mandate 13D and True for essentially every
  one since. Pre-mandate it nulled edgartools' 0 defaults by accident; post-mandate
  it stops discriminating, so `_is_placeholder_numerics` carries that guard instead.
  Either way the table never claims a 0% stake the filer did not disclose.
- The structured `.items` narrative is only populated from XML, so pre-mandate
  Item 3/4/5/6 prose is regex-carved out of `filing.text()` by two anchor sets
  whose union reads filings neither reads alone (see `_extract_13d_item_sections`).
- Carved bodies are normalized for encoding and whitespace only: 42% of filings
  carry cp1252 bytes and 84% carry box-drawing rule runs, both of which wreck
  tokenization. No sentence is ever removed (see `_normalize_item_text`).
- The 5(c) trade log is either its own exhibit or a "Schedule I" appendix inside
  the main document, so every attachment is scanned for a "Trade Date" table and
  its columns role-mapped. `att.is_html()` is a METHOD -- calling it wrongly made
  binary attachments crash the carve and zero a filing's transactions.
"""

from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup

from src.constants.constants import SEC_13D_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import new_filings, run_edgar_fetch
from src.data_store.schema import Table, Tables
from src.utils.string import pad_cik

_COLS = ["ticker", "cik", "accession_number", "form", "filing_date", "rp_seq",
        "is_amendment", "amendment_number", "cusip", "issuer_name", "date_of_event",
        "has_structured_data", "reporting_person_name", "reporting_person_cik",
        "reporting_person_citizenship", "type_of_reporting_person",
        "reporting_person_comment", "is_group_member",
        "sole_voting_power", "shared_voting_power", "sole_dispositive_power",
        "shared_dispositive_power", "aggregate_amount", "percent_of_class",
        "item3_source_of_funds", "item4_purpose_of_transaction",
        "item5_interest_in_securities", "item6_contracts_understandings",
        "primary_document", "doc_url"]

_TRANSACTION_COLS = ["ticker", "cik", "accession_number", "filing_date", "trade_seq",
                     "reporting_person_name", "trade_date", "transaction_type",
                     "quantity", "price_per_share"]

# --- Item narrative fallback -------------------------------------------------- #
# 13D items don't have MD&A-style alternate titles (unlike 10-K/10-Q), so one caption
# keyword per item is the anchor; the same separator tolerance as
# fetch_filing_text.py's `_SEP` handles the "Item 4.", "Item 4:", "Item  4 -"
# formatting variance seen across filers/eras.
#
# TWO anchor sets exist because each reads filings the other cannot -- see
# `_extract_13d_item_sections` for the union rule that combines them. `_ITEM_ANCHORS`
# matches a caption ANYWHERE, which is the only thing that reads a filing rendered
# without newlines; the line-anchored set below is what recovers the headings whose
# captions the anywhere-matcher misses.
_SEP = r"[\.\:\)\s–—-]{0,8}"
_ITEM_ANCHORS: dict[int, re.Pattern] = {
    1: re.compile(rf"item{_SEP}1\b{_SEP}security\s+and\s+issuer", re.I),
    2: re.compile(rf"item{_SEP}2\b{_SEP}identity\s+and\s+background", re.I),
    3: re.compile(rf"item{_SEP}3\b{_SEP}source\s+and\s+amount", re.I),
    4: re.compile(rf"item{_SEP}4\b{_SEP}purpose\s+of\s+transaction", re.I),
    5: re.compile(rf"item{_SEP}5\b{_SEP}interest\s+in\s+securities", re.I),
    6: re.compile(rf"item{_SEP}6\b{_SEP}contracts", re.I),
    7: re.compile(rf"item{_SEP}7\b{_SEP}material\s+to\s+be\s+filed", re.I),
}
#: Caption keyword per item, widened where filers measurably diverge from the SEC's own
#: wording -- "Purpose of THE Transaction" alone accounted for most Item 4 misses.
_ITEM_CAPTIONS: dict[int, str] = {
    1: r"security\s+and\s+(?:the\s+)?issuer",
    2: r"identity\s+and\s+background",
    3: r"source\s+(?:and\s+amount|of\s+funds)",
    4: r"purpose\s+of\s+(?:the\s+)?transaction",
    5: r"interest\s+in\s+(?:the\s+)?securities",
    6: r"contracts",
    7: r"material\s+to\s+be\s+filed",
}
#: A heading STARTS A LINE. That single constraint rejects the mid-prose cross-references
#: ("...as described in Item 4 of Schedule 13D") that make a looser bare-number anchor
#: unusable, which in turn lets the caption become OPTIONAL: when the line ends right after
#: "Item N.", it is a captionless heading, not a cross-reference. The caption, when present,
#: is consumed to end of line so a body never starts mid-caption ("or Other Consideration...").
_ITEM_ANCHORS_LINE: dict[int, re.Pattern] = {
    n: re.compile(rf"^[ \t]*item{_SEP}{n}\b[\.\:\)]?[ \t]*(?:{cap}[^\n]*|$)", re.I | re.M)
    for n, cap in _ITEM_CAPTIONS.items()
}
#: Any captioned heading, anywhere -- used ONLY to detect that a carved body swallowed a
#: later item, never to carve.
_ITEM_HEADING_ANYWHERE: dict[int, re.Pattern] = {
    n: re.compile(rf"item{_SEP}{n}{_SEP}(?:{cap})", re.I)
    for n, cap in _ITEM_CAPTIONS.items()
}
_SIGNATURE_RE = re.compile(r"^\s*signature", re.I | re.M)
_ITEM_TEXT_FIELD = {3: "item3_source_of_funds", 4: "item4_purpose_of_transaction",
                    5: "item5_interest_in_securities", 6: "item6_contracts_understandings"}
_ITEM_TEXT_MIN_CHARS = 30   # below this, it's a heading with no body (item not amended this cycle)

#: The cp1252 0x80-0x9F block, decoded to the character the filer actually meant. These
#: bytes survive EDGAR's own encoding round-trip and arrive as raw C1 codepoints (a real PSA
#: filing stores \x93group\x94 for curly quotes). DERIVED rather than hand-typed because the
#: block's less common members are SEMANTIC, not punctuation: a KDP filing stores \x80 for the
#: EURO SIGN in "Investor paid \x80 52,544.78 in cash to Acorn", where dropping the byte would
#: silently change the currency of a disclosed consideration. Five of the 32 (0x81, 0x8D, 0x8F,
#: 0x90, 0x9D) are undefined in cp1252 and drop out of the comprehension.
_CP1252_C1_BLOCK = {
    chr(b): bytes([b]).decode("cp1252", "ignore") for b in range(0x80, 0xA0)
    if bytes([b]).decode("cp1252", "ignore")
}
#: Straightened on top of the decode: the quotes, dashes and ellipsis become ASCII, and the
#: zero-width / non-breaking characters that split a word into two tokens for no semantic
#: reason are dropped. Character-for-character substitutions only -- no sentence or phrase is
#: ever removed here.
#: Written as \u escapes, not literal glyphs: the characters this table exists to remove are
#: exactly the ones an editor or a lossy copy-paste would silently mangle in the source.
_CHAR_NORMALIZATION = _CP1252_C1_BLOCK | {
    "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x95": "-", "\x96": "-",
    "\x97": "-", "\x85": "...", "\xa0": " ", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00ad": "",
    "\u200b": "", "\ufeff": "",
}
#: Box-drawing / rule lines used as visual separators under a heading (U+2500-U+257F is the
#: Box Drawing block). Bounded to runs of 3+ so a hyphenated word ("non-transferable") and a
#: negative number are never touched.
_RULE_RUN_RE = re.compile(r"[\u2500-\u257f=_]{3,}|(?<![\w-])-{3,}(?![\w-])")


def _normalize_item_text(body: str) -> str:
    """Encoding and whitespace only. Deliberately NOT a content cleaner: stripping the legal
    boilerplate and the leaked cover-page rows was measured and moved the embedding similarity
    noise floor by 1.9-2.6%, which does not pay for the regex risk of deleting real prose."""
    if not body:
        return body
    for bad, good in _CHAR_NORMALIZATION.items():
        body = body.replace(bad, good)
    body = _RULE_RUN_RE.sub(" ", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" *\n[ \t]*", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _carve_with(text: str, anchors: dict[int, re.Pattern]) -> dict[str, str]:
    """Carve Item 3/4/5/6 bodies using ONE anchor set. Each body runs from its own
    heading to whichever comes first: a later item's heading (any of {item_no+1}..7)
    or the SIGNATURE block. A missing match is normal, not an error -- amendments
    routinely restate only SOME items, leaving the others (correctly) absent."""
    out: dict[str, str] = {}
    for item_no, field in _ITEM_TEXT_FIELD.items():
        m = anchors[item_no].search(text)
        if not m:
            continue
        start = m.end()
        end = len(text)
        for later_no in range(item_no + 1, 8):
            m2 = anchors[later_no].search(text, start)
            if m2 and m2.start() < end:
                end = m2.start()
        m3 = _SIGNATURE_RE.search(text, start)
        if m3 and m3.start() < end:
            end = m3.start()
        body = _normalize_item_text(text[start:end])
        if len(body) >= _ITEM_TEXT_MIN_CHARS:
            out[field] = body
    return out


def _swallowed_a_later_item(item_no: int, body: str) -> bool:
    return any(_ITEM_HEADING_ANYWHERE[later].search(body) for later in range(item_no + 1, 8))


def _extract_13d_item_sections(text: str) -> dict[str, str]:
    """Carve Item 3/4/5/6 bodies, preferring the line-anchored headings and falling back to
    the legacy anchors ONLY where line-anchoring found nothing AND the legacy body is not
    contaminated.

    Both halves are load-bearing. Line-anchoring is what fixes the three Item 4 misses
    ("Purpose of THE Transaction", a captionless bare `Item 4.`, and a caption padded past
    the 8-char separator budget), but it cannot match a filing rendered as ONE line with no
    newlines at all (HUBB 0001162044-13-001406 is such a filing) -- the legacy anchor is the
    only thing that reads those. The contamination test is what stops the fallback
    reintroducing the bug it exists to fix: a legacy item3 body that still contains Item 4's
    heading has swallowed Item 4 and is worse than no body at all.

    Measured over 182 originals (every SC 13D in the table -- an original must answer every
    item, so it is the ground-truth set) and 200 random amendments: item3 contamination
    3.8%/2.5% -> 0%/0%, item4 coverage 92.9% -> 98.9% on originals and 61.5% -> 70.0% on
    amendments, with ZERO fields regressing on either population. Amendment coverage stays
    well under 100% because Rule 13d-2(a) has an amendment restate only materially changed
    items -- carved / present-in-the-document is ~100%. That is not a deficiency to chase."""
    if not text:
        return {}
    line_sections = _carve_with(text, _ITEM_ANCHORS_LINE)
    legacy_sections = _carve_with(text, _ITEM_ANCHORS)
    out = dict(line_sections)
    for item_no, field in _ITEM_TEXT_FIELD.items():
        if field in out or field not in legacy_sections:
            continue
        body = legacy_sections[field]
        if not _swallowed_a_later_item(item_no, body):
            out[field] = body
    return out


# --- Item 5(c) 60-day transaction-log exhibit parser -------------------------- #
# The exhibit's HTML table renders currency columns as TWO cells ("$", "36.70")
# instead of one -- a common EDGAR legacy-table quirk. Consuming cells IN HEADER
# ORDER (rather than by fixed index) absorbs that quirk generically: whichever
# role the header assigns a cell to, a literal "$" is skipped and the next cell
# taken instead, regardless of which column it is or how many filers/exhibits
# use a different layout.
_TRADE_HEADER_CUE = re.compile(r"trade\s*date", re.I)
_ROLE_KEYWORDS = [
    ("reporting_person_name", ("name",)),
    ("trade_date", ("trade date",)),
    ("transaction_type", ("buy", "sell", "exercise")),
    ("quantity", ("shares", "quantity")),
    ("price_per_share", ("unit cost", "price", "cost")),
]
# Some filers (e.g. Elliott's 2024 SC 13D on LUV) drop the separate Buy/Sell
# column entirely and encode direction IN the quantity header instead --
# "Shares Purchased (Sold)": a plain number is a buy, a parenthesized one is a
# sell. Recognized separately since it maps to BOTH a quantity value AND a
# transaction_type (derived per-row from the parens, never from a header).
_SIGNED_QUANTITY_RE = re.compile(r"(purchased|acquired|bought).{0,20}\(\s*(sold|disposed)\s*\)", re.I)


def _header_roles(header_cells: list[str]) -> list[str | None]:
    used: set[str] = set()
    roles: list[str | None] = []
    for cell in header_cells:
        low = cell.lower()
        role = None
        if "quantity" not in used and _SIGNED_QUANTITY_RE.search(low):
            role = "quantity_signed"
            used.add("quantity")
        else:
            for r, keywords in _ROLE_KEYWORDS:
                if r not in used and any(k in low for k in keywords):
                    role = r
                    used.add(r)
                    break
        roles.append(role)
    return roles


def _row_values(cells: list[str], roles: list[str | None]) -> dict[str, str]:
    out: dict[str, str] = {}
    idx = 0
    for role in roles:
        if idx >= len(cells):
            break
        val = cells[idx]
        idx += 1
        if val == "$" and idx < len(cells):     # currency symbol split into its own cell
            val = cells[idx]
            idx += 1
        if not role or not val:
            continue
        if role == "quantity_signed":
            out["quantity"] = val
            out["transaction_type"] = "Sell" if val.strip().startswith("(") else "Buy"
        else:
            out[role] = val
    return out


_NUMERIC_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _clean_transaction_row(values: dict[str, str], filing_date: pd.Timestamp | None = None) -> dict:
    """Coerce the raw text cells into usable types.

    quantity/price: extract the leading numeric token and discard everything
    else (some exhibits print "760 Shares" instead of a bare number) -> float,
    NaN for "N/A"/unparseable (never 0 -- a real disclosed value is never
    silently claimed to be zero).

    trade_date: some exhibits print a bare "MM/DD" with NO YEAR (the year is
    implied by context). `pd.Timestamp` silently defaults a missing year to
    year 1 (`"0001-11-14"`), not the filing's year -- confirmed on a real BAC
    exhibit. When that default fires (year < 1900, never a legitimate SEC
    filing date) and `filing_date` is available, re-anchor the year to the
    filing's year, stepping back one year if that would still land AFTER the
    filing (the trade must precede the 60-day-lookback disclosure)."""
    out = dict(values)
    for field in ("quantity", "price_per_share"):
        raw = out.get(field)
        m = _NUMERIC_RE.search(str(raw).replace(",", "")) if raw else None
        out[field] = float(m.group().replace(",", "")) if m else float("nan")
    raw_date = out.get("trade_date")
    try:
        trade_date = pd.Timestamp(raw_date) if raw_date else None
    except (TypeError, ValueError):
        trade_date = None
    if trade_date is not None and trade_date.year < 1900 and filing_date is not None:
        trade_date = trade_date.replace(year=filing_date.year)
        if trade_date > filing_date:
            trade_date = trade_date.replace(year=trade_date.year - 1)
    out["trade_date"] = trade_date
    return out


def _extract_transaction_rows(filing, fallback_person: str | None,
                              filing_date: pd.Timestamp | None = None) -> list[dict]:
    """Scan every attachment's HTML tables for the Item 5(c) trading-data
    exhibit (identified by a "Trade Date" header cell, not by exhibit number --
    filers use EX-99.1, EX-99.2, etc. inconsistently) and role-map its rows.
    `fallback_person` fills `reporting_person_name` when the exhibit has no Name
    column (single-filer 13Ds usually omit it, since it would be redundant).
    `filing_date` anchors a bare "MM/DD" trade date with no year (see
    `_clean_transaction_row`)."""
    rows: list[dict] = []
    attachments = getattr(filing, "attachments", None) or []
    for att in attachments:
        # `is_html` is a METHOD, not a property -- `getattr(att, "is_html", False)`
        # (no call) previously fetched the always-truthy bound method itself, so
        # non-HTML attachments (GRAPHIC/.jpg letter images, routine on modern
        # activist letters) were never actually skipped. `.content` on one of
        # those returns raw bytes, which crashed the regex search below and, via
        # the caller's blanket except, silently zeroed out the WHOLE filing's
        # transaction rows -- the real cause of transaction coverage dying out
        # for any filing with an image attachment (i.e. most post-2020 ones).
        try:
            if not att.is_html():
                continue
        except Exception:                       # noqa: BLE001 -- best-effort only
            continue
        try:
            html = att.content
        except Exception:                       # noqa: BLE001 -- best-effort only
            continue
        if not isinstance(html, str) or not _TRADE_HEADER_CUE.search(html):
            continue
        soup = BeautifulSoup(html, "html.parser")
        for table in soup.find_all("table"):
            table_rows = table.find_all("tr")
            header_idx = None
            roles: list[str | None] = []
            for i, tr in enumerate(table_rows):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if any(_TRADE_HEADER_CUE.search(c) for c in cells):
                    header_idx = i
                    roles = _header_roles(cells)
                    break
            if header_idx is None:
                continue
            for tr in table_rows[header_idx + 1:]:
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if not cells:
                    continue
                values = _row_values(cells, roles)
                if "trade_date" not in values or "transaction_type" not in values:
                    continue                    # not a data row (e.g. a footnote line)
                values.setdefault("reporting_person_name", fallback_person)
                rows.append(_clean_transaction_row(values, filing_date))
    return rows


_RP_NUMERIC_ATTRS = ("sole_voting_power", "shared_voting_power", "sole_dispositive_power",
                     "shared_dispositive_power", "aggregate_amount", "percent_of_class")


def _is_placeholder_numerics(rp) -> bool:
    """A reporting person whose SIX numerics are all 0 while `commentContent` is set has not
    disclosed a zero position -- it has deferred the numbers to the Item 5 narrative ("Rows 7,
    8, 9, 10, 11, and 13: See Item 5"). Writing the literal 0 would make the table claim a 0%
    stake, which is the one thing this module's numeric handling exists to prevent. The
    all-zero AND comment-present conjunction matters: a genuine full disposal reports zeros
    with no comment, and a commented row with real numbers keeps them."""
    if not (getattr(rp, "comment", None) or "").strip():
        return False
    values = [getattr(rp, attr, None) for attr in _RP_NUMERIC_ATTRS]
    present = [v for v in values if v is not None]
    return bool(present) and all(v == 0 for v in present)


def _num_or_null(value, trust_value: bool) -> float:
    """A ReportingPerson numeric field is only meaningful when the caller has established
    the value is real rather than a class default (usually 0). Two independent things can
    make it a default, and `trust_value` is the AND of both: the filing's own parse found
    no structured data at all, or it did but this person deferred its numbers to Item 5
    (see `_is_placeholder_numerics`). Returns NaN (never None/Python-null) so the column
    stays float dtype even when every row in a batch is unknown -- an all-None object
    column gets inferred as SQL TEXT by `ensure_table`'s dtype mapping, which would corrupt
    a genuinely numeric field the first time a real value needs to share that column."""
    if not trust_value or value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _filing_rows(filing) -> list[dict]:
    """Extract structured 13D filing rows (one row per reporting person)."""
    obj = filing.obj()
    has_structured = bool(getattr(obj, "has_structured_data", False))
    issuer = getattr(obj, "issuer_info", None)
    security = getattr(obj, "security_info", None)
    items = getattr(obj, "items", None)

    # Date normalization
    raw_event_date = (
        getattr(obj, "date_of_event", None)
        or getattr(obj, "event_date", None)
        or None
    )
    event_date = pd.Timestamp(raw_event_date) if raw_event_date else None

    # Ticker fallback logic
    ticker = getattr(filing, "ticker", None) or (
        getattr(issuer, "ticker", None) if issuer else None
    )

    # `filing.document` renders as a rich TABLE, so str() on it stored an ASCII box
    # ("+-----+ | 1 p24-2469sc13d.htm ... |") in every row instead of a URL. Take the
    # attachment's own `url`, and fall back to composing the archives path.
    doc_attr = getattr(filing, "document", None)
    doc_url = getattr(doc_attr, "url", None) if doc_attr is not None else None
    if not doc_url:
        accession = str(getattr(filing, "accession_number", "") or "")
        primary = getattr(filing, "primary_document", None)
        cik_raw = str(getattr(filing, "cik", "") or "").lstrip("0")
        if accession and primary and cik_raw:
            doc_url = (f"https://www.sec.gov/Archives/edgar/data/{cik_raw}/"
                       f"{accession.replace('-', '')}/{primary}")
    doc_url = str(doc_url) if doc_url else None

    # Item 3/4/5/6 narrative: trust the structured XML parse when present, else fall
    # back to a text-section carve (see module docstring -- has_structured is
    # essentially always False for real filings today, so this fallback is the
    # path that actually fires in practice).
    if has_structured and items:
        item3_text = getattr(items, "item3_source_of_funds", None)
        item4_text = getattr(items, "item4_purpose_of_transaction", None)
        item5_parts = [
            getattr(items, "item5_number_of_shares", None),
            getattr(items, "item5_percentage_of_class", None),
            getattr(items, "item5_transactions", None),
            getattr(items, "item5_shareholders", None),
        ]
        item5_text = " | ".join(p for p in item5_parts if p) or None
        item6_text = getattr(items, "item6_contracts", None)
    else:
        try:
            raw_text = filing.text()
        except Exception:                           # noqa: BLE001 -- best-effort only
            raw_text = None
        sections = _extract_13d_item_sections(raw_text) if raw_text else {}
        item3_text = sections.get("item3_source_of_funds")
        item4_text = sections.get("item4_purpose_of_transaction")
        item5_text = sections.get("item5_interest_in_securities")
        item6_text = sections.get("item6_contracts_understandings")

    base = {
        "ticker": ticker,
        "cik": getattr(issuer, "cik", None) if issuer else None,
        "issuer_name": getattr(issuer, "name", None) if issuer else None,
        "accession_number": getattr(filing, "accession_number", None),
        "form": getattr(filing, "form", None),
        "filing_date": (
            pd.Timestamp(filing.filing_date)
            if getattr(filing, "filing_date", None)
            else None
        ),
        "date_of_event": event_date,
        "is_amendment": 1.0 if bool(getattr(obj, "is_amendment", False)) else 0.0,
        "amendment_number": _num_or_null(
            getattr(obj, "amendment_number", None), True
        ),
        "cusip": getattr(security, "cusip", None) if security else None,
        "has_structured_data": 1.0 if has_structured else 0.0,
        # Narrative & Item extraction
        "item3_source_of_funds": item3_text,
        "item4_purpose_of_transaction": item4_text,
        "item5_interest_in_securities": item5_text,
        "item6_contracts_understandings": item6_text,
        "primary_document": getattr(filing, "primary_document", None),
        "doc_url": doc_url,
    }

    persons = getattr(obj, "reporting_persons", None) or []

    # Fallback row if no reporting persons are parsed. The six numeric fields are NaN,
    # not None, for the reason `_num_or_null` documents: an all-None column would be
    # created as TEXT by `ensure_table` the first time this row seeds a cold table.
    if not persons:
        return [
            {
                **base,
                "rp_seq": 0,
                "reporting_person_name": None,
                "reporting_person_cik": None,
                "reporting_person_citizenship": None,
                "type_of_reporting_person": None,
                "reporting_person_comment": None,
                "is_group_member": None,
                "sole_voting_power": float("nan"),
                "shared_voting_power": float("nan"),
                "sole_dispositive_power": float("nan"),
                "shared_dispositive_power": float("nan"),
                "aggregate_amount": float("nan"),
                "percent_of_class": float("nan"),
            }
        ]

    rows = []
    for seq, rp in enumerate(persons):
        # `has_structured` alone stopped discriminating at the mandate: it was False for
        # every pre-2025 filing and so nulled the class defaults by accident, but it is True
        # for every filing since, so the placeholder test is what now carries the guard.
        trust_numerics = has_structured and not _is_placeholder_numerics(rp)
        rows.append(
            {
                **base,
                "rp_seq": seq,
                "reporting_person_name": getattr(rp, "name", None),
                "reporting_person_cik": None
                if getattr(rp, "no_cik", False)
                else getattr(rp, "cik", None),
                "reporting_person_citizenship": getattr(rp, "citizenship", None)
                or None,
                "type_of_reporting_person": getattr(
                    rp, "type_of_reporting_person", None
                )
                or None,
                "reporting_person_comment": getattr(rp, "comment", None) or None,
                "is_group_member": getattr(rp, "member_of_group", None),
                "sole_voting_power": _num_or_null(
                    getattr(rp, "sole_voting_power", None), trust_numerics
                ),
                "shared_voting_power": _num_or_null(
                    getattr(rp, "shared_voting_power", None), trust_numerics
                ),
                "sole_dispositive_power": _num_or_null(
                    getattr(rp, "sole_dispositive_power", None), trust_numerics
                ),
                "shared_dispositive_power": _num_or_null(
                    getattr(rp, "shared_dispositive_power", None), trust_numerics
                ),
                "aggregate_amount": _num_or_null(
                    getattr(rp, "aggregate_amount", None), trust_numerics
                ),
                "percent_of_class": _num_or_null(
                    getattr(rp, "percent_of_class", None), trust_numerics
                ),
            }
        )

    return rows


def build_ticker_13d_edgar(ticker: str, cik: str, *, since: pd.Timestamp | None = None,
                           done_accessions: frozenset[str] = frozenset(),
                           ) -> dict[Table, pd.DataFrame]:
    """One row per reporting person plus the filing's Item 5(c) trade log. A filing whose
    `.obj()` parse fails is skipped entirely -- unlike 8-K's item codes, a 13D without its
    parsed content is not independently useful. The trade log is still attempted, since the
    two parses read different parts of the filing.

    Issuer/filer guard: a ticker's 13D listing includes every filing where its CIK appears
    AT ALL -- as the targeted issuer, or merely as a FILER disclosing a stake in some
    unrelated issuer (routine for banks whose desks cross 5% in odd closed-end funds). Only
    filings whose issuer CIK matches the ticker's own are kept; otherwise every field would
    describe a different company. An unresolvable CIK on either side means "unknown", which
    must NOT reject -- hence the falsiness checks rather than an equality test alone."""
    ticker_cik = pad_cik(cik)
    rows: list[dict] = []
    txn_rows: list[dict] = []
    for filing in new_filings(ticker, SEC_13D_FORMS, since, done_accessions):
        try:
            filing_rows = _filing_rows(filing)
        except Exception:                               # noqa: BLE001 -- best-effort only
            filing_rows = []

        issuer_cik = pad_cik(filing_rows[0].get("cik")) if filing_rows else ""
        if ticker_cik and issuer_cik and issuer_cik != ticker_cik:
            continue                                    # ticker is a FILER here, not the issuer

        person_names = [r.get("reporting_person_name") for r in filing_rows
                        if r.get("reporting_person_name")]
        for r in filing_rows:
            r["ticker"] = ticker
            rows.append(r)

        try:
            fallback_person = person_names[0] if len(person_names) == 1 else None
            filing_date = (filing_rows[0].get("filing_date") if filing_rows
                           else pd.Timestamp(filing.filing_date))
            exhibit_rows = _extract_transaction_rows(filing, fallback_person, filing_date)
        except Exception:                               # noqa: BLE001 -- best-effort only
            exhibit_rows = []
            
        cik_val = filing_rows[0].get("cik") if filing_rows else cik
        for seq, tr in enumerate(exhibit_rows):
            tr.update(ticker=ticker, cik=cik_val, accession_number=filing.accession_number,
                      filing_date=filing_date, trade_seq=seq)
            txn_rows.append(tr)

    return {Tables.sec_13d: pd.DataFrame(rows, columns=_COLS),
            Tables.sec_13d_transactions: pd.DataFrame(txn_rows, columns=_TRANSACTION_COLS)}


def fetch_13d_edgar(context: Context, tickers: list[str], years_history: int,
                    full: bool = False) -> None:
    run_edgar_fetch(context, tickers, years_history,
                    tables=(Tables.sec_13d, Tables.sec_13d_transactions),
                    build=build_ticker_13d_edgar, desc="SC 13D (edgartools)", full=full)
