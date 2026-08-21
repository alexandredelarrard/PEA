"""
fetch_13d_edgar.py (src/data_extract/utils/structure/fetch_13d_edgar.py)
---------------------------------------------------------------------------
SC 13D/13D-A activist filings via edgartools (`filing.obj()`) into two tables:
`sec_13d`, one row per (ticker, accession, rp_seq) -- co-filers are kept by
0-based position because a reporting person often has no CIK -- and
`sec_13d_transactions`, one row per disclosed trade (Item 5(c) 60-day log, an
independent grain).

Three properties the parsing depends on:
- `has_structured_data` is False for essentially every real 13D, which makes
  edgartools default missing numerics to 0. They are forced to NULL instead, so
  the table never claims a false 0% stake.
- For the same reason the structured `.items` narrative is usually empty, so
  Item 3/4/5/6 prose is regex-carved out of `filing.text()`.
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
        "reporting_person_citizenship", "type_of_reporting_person", "is_group_member",
        "sole_voting_power", "shared_voting_power", "sole_dispositive_power",
        "shared_dispositive_power", "aggregate_amount", "percent_of_class",
        "item3_source_of_funds", "item4_purpose_of_transaction",
        "item5_interest_in_securities", "item6_contracts_understandings",
        "primary_document", "doc_url"]

_TRANSACTION_COLS = ["ticker", "cik", "accession_number", "filing_date", "trade_seq",
                     "reporting_person_name", "trade_date", "transaction_type",
                     "quantity", "price_per_share"]

# --- Item narrative fallback -------------------------------------------------- #
# 13D items don't have MD&A-style alternate titles (unlike 10-K/10-Q), so a single
# anchor per item -- caption keyword right after "Item N" -- is enough; the same
# separator tolerance as fetch_filing_text.py's `_SEP` handles the "Item 4.",
# "Item 4:", "Item  4 -" formatting variance seen across filers/eras.
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
_SIGNATURE_RE = re.compile(r"^\s*signature", re.I | re.M)
_ITEM_TEXT_FIELD = {3: "item3_source_of_funds", 4: "item4_purpose_of_transaction",
                    5: "item5_interest_in_securities", 6: "item6_contracts_understandings"}
_ITEM_TEXT_MIN_CHARS = 30   # below this, it's a heading with no body (item not amended this cycle)


def _extract_13d_item_sections(text: str) -> dict[str, str]:
    """Regex-carve Item 3/4/5/6 narrative bodies out of a 13D's raw text. Each
    body runs from its own heading to whichever comes first: the next item's
    heading (any of items {item_no+1}..7) or the SIGNATURE block. A missing
    match is normal, not an error -- amendments routinely restate only SOME
    items, leaving the others (correctly) absent from that accession's text."""
    if not text:
        return {}
    out: dict[str, str] = {}
    for item_no, field in _ITEM_TEXT_FIELD.items():
        m = _ITEM_ANCHORS[item_no].search(text)
        if not m:
            continue
        start = m.end()
        end = len(text)
        for later_no in range(item_no + 1, 8):
            m2 = _ITEM_ANCHORS[later_no].search(text, start)
            if m2 and m2.start() < end:
                end = m2.start()
        m3 = _SIGNATURE_RE.search(text, start)
        if m3 and m3.start() < end:
            end = m3.start()
        body = text[start:end].strip()
        if len(body) >= _ITEM_TEXT_MIN_CHARS:
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


def _num_or_null(value, has_structured_data: bool) -> float:
    """A ReportingPerson numeric field is only meaningful when the filing's own
    parse actually found structured data -- otherwise it is the class default
    (usually 0), not a real disclosed value. Returns NaN (never None/Python-null)
    so the column stays float dtype even when every row in a batch is unknown --
    an all-None object column gets inferred as SQL TEXT by `ensure_table`'s
    dtype mapping, which would corrupt a genuinely numeric field the first time
    a real (has_structured_data=True) value needs to share that column."""
    if not has_structured_data or value is None:
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
                "is_group_member": getattr(rp, "member_of_group", None),
                "sole_voting_power": _num_or_null(
                    getattr(rp, "sole_voting_power", None), has_structured
                ),
                "shared_voting_power": _num_or_null(
                    getattr(rp, "shared_voting_power", None), has_structured
                ),
                "sole_dispositive_power": _num_or_null(
                    getattr(rp, "sole_dispositive_power", None), has_structured
                ),
                "shared_dispositive_power": _num_or_null(
                    getattr(rp, "shared_dispositive_power", None), has_structured
                ),
                "aggregate_amount": _num_or_null(
                    getattr(rp, "aggregate_amount", None), has_structured
                ),
                "percent_of_class": _num_or_null(
                    getattr(rp, "percent_of_class", None), has_structured
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


def fetch_13d_edgar(context: Context, tickers: list[str], years_history: int) -> None:
    run_edgar_fetch(context, tickers, years_history,
                    tables=(Tables.sec_13d, Tables.sec_13d_transactions),
                    build=build_ticker_13d_edgar, desc="SC 13D (edgartools)")
