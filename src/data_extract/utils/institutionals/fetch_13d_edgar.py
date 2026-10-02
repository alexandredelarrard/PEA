"""
fetch_13d_edgar.py (src/data_extract/utils/institutionals/fetch_13d_edgar.py)
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
from collections.abc import Iterator
from typing import Any, cast

import pandas as pd
from bs4 import BeautifulSoup

from src.constants.constants import SEC_13D_FORMS
from src.data_extract.utils.common.edgar_driver import EdgarFetch, EdgarScope, FilingStamp
from src.data_extract.utils.common.item_carve import ITEM_SEP, carve_spans, item_heading
from src.data_extract.utils.institutionals.schedule_rows import SCHEDULE_NUMERIC_COLS, ScheduleSpec, kept_schedule_filings
from src.data_store.schema import Table, Tables

_COLS = [
    "ticker",
    "cik",
    "accession_number",
    "form",
    "filing_date",
    "rp_seq",
    "is_amendment",
    "amendment_number",
    "cusip",
    "issuer_name",
    "date_of_event",
    "has_structured_data",
    "reporting_person_name",
    "reporting_person_cik",
    "reporting_person_citizenship",
    "type_of_reporting_person",
    "reporting_person_comment",
    "is_group_member",
    "sole_voting_power",
    "shared_voting_power",
    "sole_dispositive_power",
    "shared_dispositive_power",
    "aggregate_amount",
    "percent_of_class",
    "item3_source_of_funds",
    "item4_purpose_of_transaction",
    "item5_interest_in_securities",
    "item6_contracts_understandings",
    "primary_document",
    "doc_url",
]

_TRANSACTION_COLS = [
    "ticker",
    "cik",
    "accession_number",
    "filing_date",
    "trade_seq",
    "reporting_person_name",
    "trade_date",
    "transaction_type",
    "quantity",
    "price_per_share",
]

# --- Item narrative fallback -------------------------------------------------- #
# 13D items have no MD&A-style alternate titles, so one caption keyword per item is the anchor.
# TWO anchor sets exist because each reads filings the other cannot -- see
# `_extract_13d_item_sections` for the union rule that combines them. `_ITEM_ANCHORS`
# matches a caption ANYWHERE, which is the only thing that reads a filing rendered
# without newlines; the line-anchored set below is what recovers the headings whose
# captions the anywhere-matcher misses.
_ITEM_ANCHORS: dict[int, re.Pattern] = {
    1: item_heading(1, r"security\s+and\s+issuer"),
    2: item_heading(2, r"identity\s+and\s+background"),
    3: item_heading(3, r"source\s+and\s+amount"),
    4: item_heading(4, r"purpose\s+of\s+transaction"),
    5: item_heading(5, r"interest\s+in\s+securities"),
    6: item_heading(6, r"contracts"),
    7: item_heading(7, r"material\s+to\s+be\s+filed"),
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
_ITEM_ANCHORS_LINE: dict[int, re.Pattern] = {n: item_heading(n, cap, line_anchored=True) for n, cap in _ITEM_CAPTIONS.items()}
#: Any captioned heading, anywhere -- used ONLY to detect that a carved body swallowed a
#: later item, never to carve.
_ITEM_HEADING_ANYWHERE: dict[int, re.Pattern] = {n: re.compile(rf"item{ITEM_SEP}{n}{ITEM_SEP}(?:{cap})", re.I) for n, cap in _ITEM_CAPTIONS.items()}
_SIGNATURE_RE = re.compile(r"^\s*signature", re.I | re.M)
_ITEM_TEXT_FIELD = {
    3: "item3_source_of_funds",
    4: "item4_purpose_of_transaction",
    5: "item5_interest_in_securities",
    6: "item6_contracts_understandings",
}
_ITEM_TEXT_MIN_CHARS = 30  # below this, it's a heading with no body (item not amended this cycle)

#: The cp1252 0x80-0x9F block, decoded to the character the filer actually meant. These
#: bytes survive EDGAR's own encoding round-trip and arrive as raw C1 codepoints (a real PSA
#: filing stores \x93group\x94 for curly quotes). DERIVED rather than hand-typed because the
#: block's less common members are SEMANTIC, not punctuation: a KDP filing stores \x80 for the
#: EURO SIGN in "Investor paid \x80 52,544.78 in cash to Acorn", where dropping the byte would
#: silently change the currency of a disclosed consideration. Five of the 32 (0x81, 0x8D, 0x8F,
#: 0x90, 0x9D) are undefined in cp1252 and drop out of the comprehension.
_CP1252_C1_BLOCK = {chr(b): bytes([b]).decode("cp1252", "ignore") for b in range(0x80, 0xA0) if bytes([b]).decode("cp1252", "ignore")}
#: Straightened on top of the decode: the quotes, dashes and ellipsis become ASCII, and the
#: zero-width / non-breaking characters that split a word into two tokens for no semantic
#: reason are dropped. Character-for-character substitutions only -- no sentence or phrase is
#: ever removed here.
#: Written as \u escapes, not literal glyphs: the characters this table exists to remove are
#: exactly the ones an editor or a lossy copy-paste would silently mangle in the source.
_CHAR_NORMALIZATION = _CP1252_C1_BLOCK | {
    "\x91": "'",
    "\x92": "'",
    "\x93": '"',
    "\x94": '"',
    "\x95": "-",
    "\x96": "-",
    "\x97": "-",
    "\x85": "...",
    "\xa0": " ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00ad": "",
    "\u200b": "",
    "\ufeff": "",
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
    """Carve Item 3/4/5/6 bodies using ONE anchor set. Each body runs from its own first
    heading to whichever comes first: a later item's heading (any of {item_no+1}..7)
    or the SIGNATURE block. A missing match is normal, not an error -- amendments
    routinely restate only SOME items, leaving the others (correctly) absent."""
    out: dict[str, str] = {}
    for item_no, field in _ITEM_TEXT_FIELD.items():
        spans = carve_spans(text, anchors[item_no], [anchors[later] for later in range(item_no + 1, 8)], stop_re=_SIGNATURE_RE)
        if not spans:
            continue
        start, end = spans[0]
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


def _cell_role(low: str, used: set[str]) -> str | None:
    """The first still-free role a lower-cased header cell names, recorded in `used`; None when
    it names none. A signed "Purchased (Sold)" quantity header claims the quantity role."""
    if "quantity" not in used and _SIGNED_QUANTITY_RE.search(low):
        used.add("quantity")
        return "quantity_signed"
    role = next((r for r, keywords in _ROLE_KEYWORDS if r not in used and any(k in low for k in keywords)), None)
    if role is not None:
        used.add(role)
    return role


def _header_roles(header_cells: list[str]) -> list[str | None]:
    """One role per header cell, each role assigned at most once, left to right."""
    used: set[str] = set()
    return [_cell_role(cell.lower(), used) for cell in header_cells]


def _row_values(cells: list[str], roles: list[str | None]) -> dict[str, str]:
    out: dict[str, str] = {}
    idx = 0
    for role in roles:
        if idx >= len(cells):
            break
        val = cells[idx]
        idx += 1
        if val == "$" and idx < len(cells):  # currency symbol split into its own cell
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
    out: dict[str, object] = dict(values)
    for field in ("quantity", "price_per_share"):
        raw = out.get(field)
        m = _NUMERIC_RE.search(str(raw).replace(",", "")) if raw else None
        out[field] = float(m.group().replace(",", "")) if m else float("nan")
    raw_date = out.get("trade_date")
    try:
        trade_date = pd.Timestamp(cast(Any, raw_date)) if raw_date else None
    except (TypeError, ValueError):
        trade_date = None
    if trade_date is not None and trade_date.year < 1900 and filing_date is not None:
        trade_date = trade_date.replace(year=filing_date.year)
        if trade_date > filing_date:
            trade_date = trade_date.replace(year=trade_date.year - 1)
    out["trade_date"] = trade_date
    return out


def _extract_transaction_rows(filing: Any, fallback_person: str | None, filing_date: pd.Timestamp | None = None) -> list[dict]:
    """Scan every attachment's HTML tables for the Item 5(c) trading-data
    exhibit (identified by a "Trade Date" header cell, not by exhibit number --
    filers use EX-99.1, EX-99.2, etc. inconsistently) and role-map its rows.
    `fallback_person` fills `reporting_person_name` when the exhibit has no Name
    column (single-filer 13Ds usually omit it, since it would be redundant).
    `filing_date` anchors a bare "MM/DD" trade date with no year (see
    `_clean_transaction_row`)."""
    return [
        row
        for html in _trade_cue_html(filing)
        for table in BeautifulSoup(html, "html.parser").find_all("table")
        for row in _table_trades(table, fallback_person, filing_date)
    ]


def _attachment_html(att: Any) -> str | None:
    """An attachment's HTML text, or None when it is not HTML or cannot be read. `is_html` is a
    METHOD: a non-HTML attachment (an image letter) returns bytes from `.content`, which must be
    skipped rather than fail the filing's whole trade log."""
    try:
        if not att.is_html():
            return None
        html = att.content
    except Exception:  # noqa: BLE001 -- best-effort only
        return None
    return html if isinstance(html, str) else None


def _trade_cue_html(filing: Any) -> Iterator[str]:
    """The HTML of each attachment that carries a "Trade Date" cue, in attachment order."""
    for att in getattr(filing, "attachments", None) or []:
        html = _attachment_html(att)
        if html is not None and _TRADE_HEADER_CUE.search(html):
            yield html


def _cells(tr: Any) -> list[str]:
    """A table row's non-empty cell texts."""
    return [text for text in (cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])) if text]


def _table_trades(table: Any, fallback_person: str | None, filing_date: pd.Timestamp | None) -> list[dict]:
    """The trades of one HTML table: rows after its first "Trade Date" header row, role-mapped by
    that header; [] when the table has no such header."""
    table_rows = table.find_all("tr")
    for idx, tr in enumerate(table_rows):
        cells = _cells(tr)
        if any(_TRADE_HEADER_CUE.search(c) for c in cells):
            return _data_rows(table_rows[idx + 1 :], _header_roles(cells), fallback_person, filing_date)
    return []


def _data_rows(table_rows: list[Any], roles: list[str | None], fallback_person: str | None, filing_date: pd.Timestamp | None) -> list[dict]:
    """Cleaned trades from the rows under a header; a row without a trade date and a direction
    (a footnote line) is skipped."""
    trades: list[dict] = []
    for tr in table_rows:
        values = _row_values(_cells(tr), roles)
        if "trade_date" not in values or "transaction_type" not in values:
            continue
        values.setdefault("reporting_person_name", cast(Any, fallback_person))
        trades.append(_clean_transaction_row(values, filing_date))
    return trades


def _is_placeholder_numerics(rp: Any) -> bool:
    """A reporting person whose SIX numerics are all 0 while `commentContent` is set has not
    disclosed a zero position -- it has deferred the numbers to the Item 5 narrative ("Rows 7,
    8, 9, 10, 11, and 13: See Item 5"). Writing the literal 0 would make the table claim a 0%
    stake, which is the one thing this module's numeric handling exists to prevent. The
    all-zero AND comment-present conjunction matters: a genuine full disposal reports zeros
    with no comment, and a commented row with real numbers keeps them."""
    if not (getattr(rp, "comment", None) or "").strip():
        return False
    values = [getattr(rp, attr, None) for attr in SCHEDULE_NUMERIC_COLS]
    present = [v for v in values if v is not None]
    return bool(present) and all(v == 0 for v in present)


def _item_texts(filing: Any, obj: Any, has_structured: bool) -> dict[str, Any]:
    """Item 3/4/5/6 narrative: the structured XML parse when the filing has one, else the bodies
    carved out of `filing.text()` (a text that cannot be read yields no items)."""
    items = getattr(obj, "items", None)
    if has_structured and items:
        item5_parts = [
            getattr(items, "item5_number_of_shares", None),
            getattr(items, "item5_percentage_of_class", None),
            getattr(items, "item5_transactions", None),
            getattr(items, "item5_shareholders", None),
        ]
        return {
            "item3_source_of_funds": getattr(items, "item3_source_of_funds", None),
            "item4_purpose_of_transaction": getattr(items, "item4_purpose_of_transaction", None),
            "item5_interest_in_securities": " | ".join(p for p in item5_parts if p) or None,
            "item6_contracts_understandings": getattr(items, "item6_contracts", None),
        }
    try:
        raw_text = filing.text()
    except Exception:  # noqa: BLE001 -- best-effort only
        raw_text = None
    sections = _extract_13d_item_sections(raw_text) if raw_text else {}
    return {field: sections.get(field) for field in _ITEM_TEXT_FIELD.values()}


def _trust_numerics(rp: Any, has_structured: bool) -> bool:
    """A 13D numeric is disclosed only in a structured filing whose reporting person did not defer
    its numbers to the Item 5 narrative (see `_is_placeholder_numerics`)."""
    return has_structured and not _is_placeholder_numerics(rp)


def _reporting_person_ciks(filing: Any, persons: list[Any]) -> list[str | None]:
    """Each reporting person's own parsed CIK, None when it asserts `no_cik`; no header backfill."""
    return [None if getattr(rp, "no_cik", False) else getattr(rp, "cik", None) for rp in persons]


def _event_date(obj: Any) -> pd.Timestamp | None:
    """`date_of_event`, else `event_date`, as a Timestamp; None when both are blank."""
    raw = getattr(obj, "date_of_event", None) or getattr(obj, "event_date", None) or None
    return pd.Timestamp(raw) if raw else None


def _filing_transactions(ticker: str, cik: str, stamp: FilingStamp, filing_rows: list[dict]) -> list[dict]:
    """The filing's Item 5(c) trades stamped with ticker, issuer CIK, accession, filing date and
    `trade_seq`. A sole named reporting person fills an exhibit that has no Name column."""
    names = [r.get("reporting_person_name") for r in filing_rows if r.get("reporting_person_name")]
    fallback_person = names[0] if len(names) == 1 else None
    try:
        trades = _extract_transaction_rows(stamp.filing, fallback_person, stamp.filed)
    except Exception as exc:  # noqa: BLE001 -- filing parser boundary
        raise RuntimeError(f"{SCHEDULE_13D.label} accession {stamp.accession_number} transaction exhibit could not be parsed") from exc
    issuer_cik = filing_rows[0].get("cik") if filing_rows else cik
    for seq, trade in enumerate(trades):
        trade.update(ticker=ticker, cik=issuer_cik, accession_number=stamp.accession_number, filing_date=stamp.filed, trade_seq=seq)
    return trades


SCHEDULE_13D = ScheduleSpec(
    label="SC 13D",
    forms=tuple(SEC_13D_FORMS),
    columns=tuple(_COLS),
    extra_base=_item_texts,
    trust_numerics=_trust_numerics,
    reporting_person_ciks=_reporting_person_ciks,
    event_date=_event_date,
    blank_to_none=False,
)


def build_ticker_13d_edgar(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None = None,
    done_accessions: frozenset[str] = frozenset(),
    scope: EdgarScope,
) -> dict[Table, pd.DataFrame]:
    """`ticker`'s new issuer-side 13D filings: one `sec_13d` row per reporting person plus each
    filing's Item 5(c) trades in `sec_13d_transactions`. A filing whose `.obj()` parse or trade
    exhibit fails raises, failing the ticker."""
    rows: list[dict] = []
    txn_rows: list[dict] = []
    walk = kept_schedule_filings(ticker, cik, since=since, done_accessions=done_accessions, scope=scope, spec=SCHEDULE_13D)
    for stamp, filing_rows in walk:
        rows.extend(filing_rows)
        txn_rows.extend(_filing_transactions(ticker, cik, stamp, filing_rows))
    return {Tables.sec_13d: pd.DataFrame(rows, columns=_COLS), Tables.sec_13d_transactions: pd.DataFrame(txn_rows, columns=_TRANSACTION_COLS)}


SEC_13D_FETCH = EdgarFetch(desc="SC 13D (edgartools)", tables=(Tables.sec_13d, Tables.sec_13d_transactions), build=build_ticker_13d_edgar)
