"""
fetch_13d_edgar.py (src/data_extract/utils/structure/fetch_13d_edgar.py)
---------------------------------------------------------------------------
Extracts SC 13D/13D-A filings into `sec_13d` using `edgartools` (`filing.obj()`).
Replaces legacy `fetch_13d.py` by capturing full reporting person metadata,
CUSIPs, amendment details, and Item 4 narrative text.

Data Grain:
- One row per (ticker, accession_number, rp_seq) in `sec_13d`.
- One row per disclosed trade in `sec_13d_transactions` (Item 5(c) 60-day log,
  see below) -- independent grain, no rp_seq relationship.
- Preserves multi-filer group submissions using 0-based position (`rp_seq`)
  since individual co-filers may lack a CIK (`no_cik=True`).

Key Guardrails:
- Numeric Null Guard: Unstructured 13Ds lack standard XML schemas, causing
  `edgartools` to default missing numeric fields to 0. If `has_structured_data`
  is False, force all numeric fields (ownership %, voting powers) to NULL
  to avoid publishing false "0% ownership" metrics. Only Name/CIK are trusted.
- Item Narrative Text Fallback: `edgartools`' structured `.items` (Item 3/4/5/6
  narrative) is populated ONLY from a filing's XML submission. Empirically --
  checked across 100+ real filings spanning 1994 to mid-2025, including several
  filed well after the SEC's Dec-2024 structured-data mandate -- `has_structured_data`
  is essentially always False for real SC 13D filings today, so relying on
  `.items` alone silently loses Item 4 (the highest-signal field: activist intent)
  for virtually the entire archive. The underlying `filing.text()` DOES reliably
  carry the Item 3/4/5/6 prose regardless of XML availability, so `_filing_rows`
  falls back to a regex section-carve (`_extract_13d_item_sections`, same anchor
  style as `fetch_filing_text.py`'s 10-K/10-Q carving) whenever the structured
  parse is empty.
- 60-Day Transaction Log: Item 5(c)'s trade-by-trade log is usually NOT inline
  narrative text -- it is filed as a separate exhibit (e.g. `EX-99.2`, a
  "TRADING DATA" table: reporting person, trade date, buy/sell, quantity, unit
  cost). `_extract_transaction_rows` scans every filing attachment's HTML tables
  for one with a "Trade Date" header and role-maps its columns generically
  (works regardless of exhibit number or exact column order/count).
"""

from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup
from edgar import Company

from src.constants.constants import SEC_13D_FORMS, SEC_13D_TABLE, SEC_13D_TRANSACTIONS_TABLE
from src.context import Context
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import _configure_identity

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


def _header_roles(header_cells: list[str]) -> list[str | None]:
    used: set[str] = set()
    roles: list[str | None] = []
    for cell in header_cells:
        low = cell.lower()
        role = None
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
        if role and val:
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
        if not getattr(att, "is_html", False):
            continue
        try:
            html = att.content
        except Exception:                       # noqa: BLE001 -- best-effort only
            continue
        if not html or not _TRADE_HEADER_CUE.search(html):
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


def _cik_num(value) -> int | None:
    """Normalize a CIK for comparison (zero-padding/type varies by source --
    '0000070858' from a filing header vs '70858' from the ticker universe)."""
    try:
        return int(str(value).lstrip("0") or "0")
    except (TypeError, ValueError):
        return None


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

    # Document URL resolution
    doc_attr = getattr(filing, "document", None)
    doc_url = str(doc_attr) if doc_attr else None

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

    # Fallback row if no reporting persons are parsed
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
                "sole_voting_power": None,
                "shared_voting_power": None,
                "sole_dispositive_power": None,
                "shared_dispositive_power": None,
                "aggregate_amount": None,
                "percent_of_class": None,
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
                          done_accessions: frozenset[str] = frozenset()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walks `Company(ticker).get_filings(form=SEC_13D_FORMS)`, skips accessions
    already in `done_accessions` or filed before `since`, and reads each filing's
    typed `Schedule13D` object -- one row per reporting person (see module
    docstring for the grain rationale). A filing whose `.obj()` parse fails is
    skipped entirely (no event-only fallback row): unlike 8-K's item codes, SC 13D
    without its parsed content is not independently useful metadata.

    Issuer/Filer Guard: `Company(ticker).get_filings(form=SEC_13D_FORMS)` returns
    EVERY SC 13D where `ticker`'s CIK appears at all -- as the subject company
    (an activist targeting it) OR merely as A FILER disclosing a >5% stake it
    holds in some UNRELATED issuer (routine for banks/asset managers whose
    trading desks cross 5% thresholds in odd micro-caps/closed-end funds; e.g.
    Bank of America is a routine 13D FILER on municipal bond funds it has no
    connection to as a company). A filing is only kept when the extracted
    issuer CIK matches `cik` (the ticker's own CIK, passed by the caller) --
    otherwise `ticker` is the filer, not the target, and every field (issuer
    name, trade prices/dates) would describe a different company entirely.

    Returns `(sec_13d_rows, transaction_rows)`: the per-reporting-person table and
    the independent Item 5(c) trade-log table (see `_extract_transaction_rows`).
    A filing's trade-log exhibit is attempted even when its own `_filing_rows`
    call fails, since the two parses draw on different parts of the filing."""

    company = Company(ticker)
    filings = company.get_filings(form=list(SEC_13D_FORMS))
    sorted_filings = sorted(filings, key=lambda f: f.filing_date)
    if since is not None:
        sorted_filings = [f for f in sorted_filings if pd.Timestamp(f.filing_date) >= since]

    ticker_cik = _cik_num(cik)
    rows: list[dict] = []
    txn_rows: list[dict] = []
    for filing in sorted_filings:
        if filing.accession_number in done_accessions:
            continue
        try:
            filing_rows = _filing_rows(filing)
        except Exception:                               # noqa: BLE001 -- best-effort only
            filing_rows = []

        issuer_cik = _cik_num(filing_rows[0].get("cik")) if filing_rows else None
        if ticker_cik is not None and issuer_cik is not None and issuer_cik != ticker_cik:
            continue                                    # ticker is a FILER here, not the issuer

        person_names = [r.get("reporting_person_name") for r in filing_rows
                        if r.get("reporting_person_name")]
        for r in filing_rows:
            r["ticker"] = ticker
            rows.append(r)

        try:
            fallback_person = person_names[0] if len(person_names) == 1 else None
            filing_date = filing_rows[0].get("filing_date") if filing_rows else pd.Timestamp(filing.filing_date)
            exhibit_rows = _extract_transaction_rows(filing, fallback_person, filing_date)
        except Exception:                               # noqa: BLE001 -- best-effort only
            exhibit_rows = []
        cik_val = filing_rows[0].get("cik") if filing_rows else cik
        for seq, tr in enumerate(exhibit_rows):
            tr.update(ticker=ticker, cik=cik_val, accession_number=filing.accession_number,
                      filing_date=filing_date, trade_seq=seq)
            txn_rows.append(tr)

    return (pd.DataFrame(rows, columns=_COLS),
            pd.DataFrame(txn_rows, columns=_TRANSACTION_COLS))


def fetch_13d_edgar(context: Context, tickers: list[str]) -> pd.DataFrame:
    """Public entry point (mirrors `fetch_8k_edgar`'s conventions): per-ticker
    try/except, incremental via `existing_filings` (dedup by accession ONLY --
    every ticker's FULL `years` window is re-listed every run, gap-filling instead
    of resuming from a max-date cutoff), scoped by `years` (falls back to
    `data_extract.years_history`). Tickers are walked CONCURRENTLY on a thread
    pool (`run_per_ticker`) -- see parallel_fetch.py's module docstring."""

    _configure_identity()

    years = int(context.config.data_extract.years_history)
    since = pd.Timestamp.today() - pd.DateOffset(years=years)
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    seen = existing_filings(context, SEC_13D_TABLE)

    def _worker(ticker: str, cik: str) -> tuple[int, int, bool]:
        try:
            out, txns = build_ticker_13d_edgar(ticker, cik, since=since, done_accessions=seen)
        except Exception as e:                          # noqa: BLE001
            context.log.warning("fetch_13d_edgar: %s failed (%s)", ticker, e)
            return 0, 0, False
        if not out.empty:
            context.store.save(SEC_13D_TABLE, out)
        if not txns.empty:
            context.store.save(SEC_13D_TRANSACTIONS_TABLE, txns)
        return len(out), len(txns), True

    results = run_per_ticker(cik_map, _worker, desc="SC 13D (edgartools)")
    total_rows = sum(n for n, _, _ in results)
    txn_total = sum(t for _, t, _ in results)
    failed = sum(1 for _, _, ok in results if not ok)

    context.log.info("fetch_13d_edgar: +%d rows (+%d transactions) across %d/%d ticker(s) "
                     "(%d failed) -> '%s'/'%s'", total_rows, txn_total,
                     len(results), len(cik_map), failed, SEC_13D_TABLE, SEC_13D_TRANSACTIONS_TABLE)
    return context.store.load(SEC_13D_TABLE)
