"""
fetch_13d_edgar.py  (src/data_extract/utils/structure/fetch_13d_edgar.py)
---------------------------------------------------------------------------
edgartools-based SC 13D extraction -> `sec_13d`. Replaces `fetch_13d.py`'s
submissions-JSON approach (event/date only -- filer name and % owned were
explicitly a "later best-effort doc-parse enhancement" the old file never built)
by reading each filing's typed `Schedule13D` object (`filing.obj()`), which
exposes `reporting_persons` (name, CIK, citizenship, voting/dispositive power,
percent_of_class), `issuer_info` / `security_info` (CUSIP), `is_amendment` /
`amendment_number`, and best-effort narrative `items` (purpose of transaction).

`has_structured_data` (a field ON `Schedule13D` itself) matters: confirmed
empirically against several real filings (GameStop 2005-2024, spanning old and
new EDGAR-native XML eras) that it is consistently False, and every ReportingPerson
numeric field (voting/dispositive power, aggregate_amount, percent_of_class)
DEFAULTS TO 0 in that case rather than reflecting a real parsed value -- SC 13D is a
narrative form with no XBRL-grade structured schema, unlike SC 13F's required XML
infotable. Publishing those zeros as if real would silently claim "0% ownership" for
an activist that may hold 15%. So every such numeric field is stored NULL, never a
default 0, whenever `has_structured_data` is False -- only the reporting person's
NAME/CIK (which the header always carries reliably) is trusted unconditionally.

Grain: one row per (ticker, accession_number, rp_seq) -- a single 13D can have
MULTIPLE co-filing reporting persons (e.g. a fund + its general partner), and
collapsing them into one row would silently drop all but one filer. `rp_seq` (the
person's 0-based position in the filing) is used as the sub-key rather than CIK,
since a reporting person without an assigned CIK (`no_cik=True`) is common.
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from src.constants.constants import SEC_13D_FORMS, SEC_13D_TABLE
from src.context import Context
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping
from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import _configure_identity

_TABLE = SEC_13D_TABLE
_COLS = ["ticker", "cik", "accession_number", "form", "filing_date", "rp_seq",
        "is_amendment", "amendment_number", "cusip", "issuer_name", "date_of_event",
        "has_structured_data", "reporting_person_name", "reporting_person_cik",
        "reporting_person_citizenship", "type_of_reporting_person",
        "sole_voting_power", "shared_voting_power", "sole_dispositive_power",
        "shared_dispositive_power", "aggregate_amount", "percent_of_class",
        "item4_purpose_of_transaction", "primary_document", "doc_url"]


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
    obj = filing.obj()
    has_structured = bool(getattr(obj, "has_structured_data", False))
    issuer = getattr(obj, "issuer_info", None)
    security = getattr(obj, "security_info", None)
    items = getattr(obj, "items", None)
    date_of_event = getattr(obj, "date_of_event", None) or getattr(obj, "event_date", None) or None

    base = {
        "ticker": None, "cik": getattr(issuer, "cik", None) if issuer else None,
        "accession_number": filing.accession_number, "form": filing.form,
        "filing_date": pd.Timestamp(filing.filing_date),
        "is_amendment": 1.0 if bool(getattr(obj, "is_amendment", False)) else 0.0,
        "amendment_number": _num_or_null(getattr(obj, "amendment_number", None), True),
        "cusip": getattr(security, "cusip", None) if security else None,
        "issuer_name": getattr(issuer, "name", None) if issuer else None,
        "date_of_event": date_of_event or None,
        "has_structured_data": 1.0 if has_structured else 0.0,
        "item4_purpose_of_transaction": getattr(items, "item4_purpose_of_transaction", None) if items else None,
        "primary_document": getattr(filing, "primary_document", None),
        "doc_url": getattr(filing, "document", None) and str(getattr(filing, "document")) or None,
    }
    persons = getattr(obj, "reporting_persons", None) or []
    if not persons:
        return [{**base, "rp_seq": 0, "reporting_person_name": None, "reporting_person_cik": None,
                "reporting_person_citizenship": None, "type_of_reporting_person": None,
                "sole_voting_power": None, "shared_voting_power": None,
                "sole_dispositive_power": None, "shared_dispositive_power": None,
                "aggregate_amount": None, "percent_of_class": None}]
    rows = []
    for seq, rp in enumerate(persons):
        rows.append({
            **base, "rp_seq": seq,
            "reporting_person_name": getattr(rp, "name", None),
            "reporting_person_cik": None if getattr(rp, "no_cik", False) else getattr(rp, "cik", None),
            "reporting_person_citizenship": getattr(rp, "citizenship", None) or None,
            "type_of_reporting_person": getattr(rp, "type_of_reporting_person", None) or None,
            "sole_voting_power": _num_or_null(getattr(rp, "sole_voting_power", None), has_structured),
            "shared_voting_power": _num_or_null(getattr(rp, "shared_voting_power", None), has_structured),
            "sole_dispositive_power": _num_or_null(getattr(rp, "sole_dispositive_power", None), has_structured),
            "shared_dispositive_power": _num_or_null(getattr(rp, "shared_dispositive_power", None), has_structured),
            "aggregate_amount": _num_or_null(getattr(rp, "aggregate_amount", None), has_structured),
            "percent_of_class": _num_or_null(getattr(rp, "percent_of_class", None), has_structured),
        })
    return rows


def build_ticker_13d_edgar(ticker: str, *, since: pd.Timestamp | None = None,
                          done_accessions: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Walks `Company(ticker).get_filings(form=SEC_13D_FORMS)`, skips accessions
    already in `done_accessions` or filed before `since`, and reads each filing's
    typed `Schedule13D` object -- one row per reporting person (see module
    docstring for the grain rationale). A filing whose `.obj()` parse fails is
    skipped entirely (no event-only fallback row): unlike 8-K's item codes, SC 13D
    without its parsed content is not independently useful metadata."""
    from edgar import Company

    company = Company(ticker)
    filings = company.get_filings(form=list(SEC_13D_FORMS))
    sorted_filings = sorted(filings, key=lambda f: f.filing_date)
    if since is not None:
        sorted_filings = [f for f in sorted_filings if pd.Timestamp(f.filing_date) >= since]

    rows: list[dict] = []
    for filing in sorted_filings:
        if filing.accession_number in done_accessions:
            continue
        try:
            filing_rows = _filing_rows(filing)
        except Exception:                               # noqa: BLE001 -- best-effort only
            continue
        for r in filing_rows:
            r["ticker"] = ticker
            rows.append(r)
    return pd.DataFrame(rows, columns=_COLS)


def fetch_13d_edgar(context: Context, tickers: list[str], years: int | None = None) -> pd.DataFrame:
    """Public entry point (mirrors `fetch_8k_edgar`'s conventions): per-ticker
    try/except, incremental via `existing_filings`, scoped by `years` (falls back
    to `data_extract.years_history`)."""
    _configure_identity()
    years = int(years if years is not None else context.config.data_extract.years_history)
    since = pd.Timestamp.today() - pd.DateOffset(years=years)
    cik_map = load_cik_mapping(context)
    cik_map = cik_map[cik_map["ticker"].isin(tickers)]

    seen, last_by_ticker = existing_filings(context, _TABLE)
    all_frames: list[pd.DataFrame] = []
    touched, failed = 0, 0
    for _, r in tqdm(cik_map.iterrows(), total=len(cik_map), desc="SC 13D (edgartools)"):
        ticker = r["ticker"]
        try:
            ticker_since = max(since, last_by_ticker[ticker]) if ticker in last_by_ticker else since
            out = build_ticker_13d_edgar(ticker, since=ticker_since, done_accessions=seen)
        except Exception as e:                          # noqa: BLE001
            context.log.warning("fetch_13d_edgar: %s failed (%s)", ticker, e)
            failed += 1
            continue
        if not out.empty:
            context.store.save(_TABLE, out)
            seen.update(out["accession_number"].astype(str))
            all_frames.append(out)
        touched += 1

    context.log.info("fetch_13d_edgar: +%d rows across %d/%d ticker(s) (%d failed) -> '%s'",
                     sum(len(f) for f in all_frames), touched, len(cik_map), failed, _TABLE)
    return context.store.load(_TABLE)
