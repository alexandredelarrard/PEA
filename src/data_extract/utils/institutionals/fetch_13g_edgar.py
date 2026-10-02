"""
fetch_13g_edgar.py (src/data_extract/utils/institutionals/fetch_13g_edgar.py)
-----------------------------------------------------------------------------
Schedule 13G / 13G/A -- the PASSIVE >5% beneficial-ownership channel -- via edgartools
(`filing.obj()` -> `Schedule13G`) into `sec_13g`, one row per reporting person, the same grain
and column names as `sec_13d` so the 13G->13D escalation join is a plain
(ticker, reporting_person_cik) union ordered by filing_date.
"""

from __future__ import annotations

import re

import pandas as pd

from src.constants.constants import SEC_13G_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import (
    EdgarScope,
    FilingStamp,
    new_schedule_filings,
    num_or_null,
    run_edgar_fetch,
)
from src.data_extract.utils.common.registrant import issuer_ciks
from src.data_store.schema import Table, Tables
from src.utils.string import pad_cik

# Mirrors `sec_13d`'s columns minus the four Item narratives (a 13G has no
# purpose-of-transaction item), plus `rule_designation`.
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
    "rule_designation",
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
    "primary_document",
    "doc_url",
]

_NUMERIC_COLS = [
    "sole_voting_power",
    "shared_voting_power",
    "sole_dispositive_power",
    "shared_dispositive_power",
    "aggregate_amount",
    "percent_of_class",
]

# The XML cover page dates as MM/DD/YYYY ('03/31/2026' on the ETN filing above). Parsed with an
# explicit format rather than letting pandas infer: `01/02/2026` is a legal cover-page date and
# an inferred parse can read it day-first, silently moving an event by ten months.
_EVENT_DATE_FORMAT = "%m/%d/%Y"

# Legal-form suffixes carry no identity: the 13G XML writes the manager's trading name
# ('Vanguard Capital Management') where the SGML header writes its registered one
# ('VANGUARD CAPITAL MANAGEMENT LLC'). Stripped from BOTH sides before matching.
_ENTITY_SUFFIXES = re.compile(
    r"\b(l\.?l\.?c|l\.?p|l\.?l\.?p|inc|incorporated|corp|corporation|co|company|ltd|limited|"
    r"plc|sa|nv|ag|gmbh|trust|holdings?|group|partners|management)\b",
    re.I,
)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def _norm_entity(name: str | None) -> str:
    """An entity name reduced to its matchable core: lower-cased, punctuation dropped, legal-form
    suffixes removed, whitespace collapsed. Returns '' for anything unusable, which callers must
    treat as "no match" rather than as a match against another empty name."""
    if not name:
        return ""
    text = _NON_ALNUM.sub(" ", str(name).lower())
    text = _ENTITY_SUFFIXES.sub(" ", text)
    return " ".join(text.split())


def _header_filers(filing) -> list[tuple[str, str]]:
    """`(cik, name)` for each filer in the SGML header, in header order. Free once `.obj()` has
    run -- both go through the same memoized `filing.sgml()`. Empty on any failure: a missing
    header must degrade the CIK backfill, never drop the filing."""
    try:
        header = filing.header
    except Exception:  # noqa: BLE001 -- best-effort only
        return []
    out: list[tuple[str, str]] = []
    for flr in getattr(header, "filers", None) or []:
        info = getattr(flr, "company_information", None)
        if info is not None and getattr(info, "cik", None):
            out.append((str(info.cik), str(getattr(info, "name", "") or "")))
    return out


def _reporting_person_cik(rp, seq: int, filers: list[tuple[str, str]], n_persons: int) -> str | None:
    """This reporting person's CIK, from the parsed object when it has one and from the SGML
    header's filer list when it does not (see the module docstring: the post-mandate 13G XML
    cover page has no CIK element at all).

    `no_cik` is the filer's own assertion that it HAS no CIK, and it wins over everything --
    inventing one from the header would attribute a stake to whichever entity happened to
    transmit the filing.

    Three fallbacks, narrowest first, so a wrong CIK is never preferred to a missing one:
    an exact normalized name match, then a unique prefix match either way round (the header's
    registered name is usually the XML trading name plus a legal suffix), then position -- and
    position only when the two lists are the same length, i.e. when the mapping is forced."""
    if getattr(rp, "no_cik", False):
        return None
    own = getattr(rp, "cik", None)
    if own:
        return str(own)
    if not filers:
        return None

    target = _norm_entity(getattr(rp, "name", None))
    if target:
        exact = [cik for cik, name in filers if _norm_entity(name) == target]
        if len(exact) == 1:
            return exact[0]
        prefix = [cik for cik, name in filers if (norm := _norm_entity(name)) and (norm.startswith(target) or target.startswith(norm))]
        if len(prefix) == 1:
            return prefix[0]

    if len(filers) == n_persons and seq < len(filers):
        return filers[seq][0]
    return None


def _event_date(raw) -> pd.Timestamp | None:
    """The cover-page event date, or None. Pre-mandate it is `''` (the class default), which must
    read as unknown rather than as an epoch."""
    if not raw:
        return None
    parsed = pd.to_datetime(raw, format=_EVENT_DATE_FORMAT, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(raw, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _filing_rows(stamp: FilingStamp) -> list[dict]:
    """One Schedule 13G -> one row per reporting person. Pure apart from `filing.obj()`."""
    filing = stamp.filing
    obj = filing.obj()
    has_structured = bool(getattr(obj, "has_structured_data", False))
    issuer = getattr(obj, "issuer_info", None)
    security = getattr(obj, "security_info", None)

    base = {
        "ticker": getattr(filing, "ticker", None) or (getattr(issuer, "ticker", None) if issuer else None),
        "cik": getattr(issuer, "cik", None) if issuer else None,
        "issuer_name": getattr(issuer, "name", None) if issuer else None,
        "accession_number": stamp.accession_number,
        "form": stamp.form,
        "filing_date": stamp.filed,
        "date_of_event": _event_date(getattr(obj, "date_of_event", None)),
        "is_amendment": 1.0 if bool(getattr(obj, "is_amendment", False)) else 0.0,
        "amendment_number": num_or_null(getattr(obj, "amendment_number", None), True),
        # '' is the pre-mandate class default for both, not a value
        "cusip": (getattr(security, "cusip", None) if security else None) or None,
        "has_structured_data": 1.0 if has_structured else 0.0,
        "rule_designation": getattr(obj, "rule_designation", None) or None,
        "primary_document": stamp.primary_document,
        "doc_url": stamp.doc_url,
    }

    persons = getattr(obj, "reporting_persons", None) or []

    # Fallback row when no reporting person parsed, exactly as 13D does. The numerics are NaN
    # rather than None so a batch in which every filing fails this way still seeds a float
    # column -- `ensure_table` infers an all-None object column as SQL TEXT.
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
                **{col: float("nan") for col in _NUMERIC_COLS},
            }
        ]

    filers = _header_filers(filing)
    rows = []
    for seq, rp in enumerate(persons):
        rows.append(
            {
                **base,
                "rp_seq": seq,
                "reporting_person_name": getattr(rp, "name", None) or None,
                "reporting_person_cik": _reporting_person_cik(rp, seq, filers, len(persons)),
                "reporting_person_citizenship": getattr(rp, "citizenship", None) or None,
                "type_of_reporting_person": getattr(rp, "type_of_reporting_person", None) or None,
                "reporting_person_comment": getattr(rp, "comment", None) or None,
                "is_group_member": getattr(rp, "member_of_group", None),
                **{col: num_or_null(getattr(rp, col, None), has_structured) for col in _NUMERIC_COLS},
            }
        )
    return rows


def build_ticker_13g_edgar(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None = None,
    done_accessions: frozenset[str] = frozenset(),
    scope: EdgarScope,
) -> dict[Table, pd.DataFrame]:
    """`ticker`'s new Schedule 13G filings as `sec_13g` rows.

    Issuer/filer guard: a ticker's 13G listing includes every filing where its CIK appears AT
    ALL -- as the subject issuer, or merely as the FILER disclosing a stake in some unrelated
    issuer. That is routine on 13G, where an S&P 500 asset manager or bank files hundreds
    against other companies; kept, every field would describe a different company. An
    unresolvable CIK on either side means "unknown" and must NOT reject -- hence the falsiness
    checks rather than an equality test alone."""
    ticker_ciks = issuer_ciks(ticker, cik, scope.registrants, scope.identity)
    rows: list[dict] = []
    for filing in new_schedule_filings(ticker, ticker_ciks, SEC_13G_FORMS, since, done_accessions):
        stamp = FilingStamp.of(filing, cik)
        try:
            filing_rows = _filing_rows(stamp)
        except Exception as exc:  # noqa: BLE001 -- filing parser boundary
            raise RuntimeError(f"SC 13G accession {stamp.accession_number} could not be parsed") from exc

        issuer_cik = pad_cik(filing_rows[0].get("cik")) if filing_rows else ""
        if ticker_ciks and issuer_cik and issuer_cik not in ticker_ciks:
            continue  # ticker is a FILER here, not the issuer

        for row in filing_rows:
            row["ticker"] = ticker
            rows.append(row)

    return {Tables.sec_13g: pd.DataFrame(rows, columns=_COLS)}


def fetch_13g_edgar(context: Context, tickers: list[str], years_history: int, full: bool = False) -> None:
    run_edgar_fetch(
        context,
        tickers,
        years_history,
        tables=(Tables.sec_13g,),
        build=build_ticker_13g_edgar,
        desc="SC 13G (edgartools)",
        full=full,
        require_complete=True,
        identity_aware=True,
    )
