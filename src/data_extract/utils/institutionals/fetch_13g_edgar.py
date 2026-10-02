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
from functools import partial
from typing import Any

import pandas as pd

from src.constants.constants import SEC_13G_FORMS
from src.data_extract.utils.common.edgar_driver import EdgarFetch
from src.data_extract.utils.institutionals.schedule_rows import ScheduleSpec, build_schedule_rows
from src.data_store.schema import Tables

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


def _header_filers(filing: Any) -> list[tuple[str, str]]:
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


def _reporting_person_cik(rp: Any, seq: int, filers: list[tuple[str, str]], n_persons: int) -> str | None:
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


def _reporting_person_ciks(filing: Any, persons: list[Any]) -> list[str | None]:
    """One CIK per reporting person, backfilled from the SGML header's filer list (read once)."""
    filers = _header_filers(filing)
    return [_reporting_person_cik(rp, seq, filers, len(persons)) for seq, rp in enumerate(persons)]


def _event_date(obj: Any) -> pd.Timestamp | None:
    """The cover-page event date, or None. Pre-mandate it is `''` (the class default), which must
    read as unknown rather than as an epoch."""
    raw = getattr(obj, "date_of_event", None)
    if not raw:
        return None
    parsed = pd.to_datetime(raw, format=_EVENT_DATE_FORMAT, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(raw, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _rule_designation(filing: Any, obj: Any, has_structured: bool) -> dict[str, Any]:
    """The 13G-only filing field: the Rule 13d-1 paragraph the filer relies on."""
    return {"rule_designation": getattr(obj, "rule_designation", None) or None}


def _trust_disclosed(rp: Any, has_structured: bool) -> bool:
    """A 13G numeric is a disclosed value exactly when the filing carried structured data."""
    return has_structured


SCHEDULE_13G = ScheduleSpec(
    label="SC 13G",
    forms=tuple(SEC_13G_FORMS),
    columns=tuple(_COLS),
    extra_base=_rule_designation,
    trust_numerics=_trust_disclosed,
    reporting_person_ciks=_reporting_person_ciks,
    event_date=_event_date,
    blank_to_none=True,
)

SEC_13G_FETCH = EdgarFetch(
    desc="SC 13G (edgartools)",
    tables=(Tables.sec_13g,),
    build=partial(build_schedule_rows, spec=SCHEDULE_13G, table=Tables.sec_13g),
)
