"""
fetch_13g_edgar.py (src/data_extract/utils/institutionals/fetch_13g_edgar.py)
-----------------------------------------------------------------------------
Schedule 13G / 13G/A (passive >5% ownership) via edgartools into `sec_13g`, one row per reporting
person with `sec_13d`'s grain and column names, built by the shared `schedule_rows` row builder.
A missing reporting-person CIK is backfilled from the SGML header filer list (read through `sec_io`).
"""

from __future__ import annotations

import re
from functools import partial
from typing import Any

import pandas as pd

from src.constants.constants import SEC_13G_FORMS
from src.data_extract.utils.common.edgar_driver import EdgarFetch
from src.data_extract.utils.common.sec_io import TransientReadError, filing_header
from src.data_extract.utils.institutionals.schedule_rows import ScheduleSpec, build_schedule_rows
from src.data_store.schema import Tables

# `sec_13d`'s columns minus the four Item narratives, plus `rule_designation`.
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

# Explicit cover-page format so an ambiguous date is never inferred day-first.
_EVENT_DATE_FORMAT = "%m/%d/%Y"

# Legal-form suffixes, stripped from both XML and header names before matching.
_ENTITY_SUFFIXES = re.compile(
    r"\b(l\.?l\.?c|l\.?p|l\.?l\.?p|inc|incorporated|corp|corporation|co|company|ltd|limited|"
    r"plc|sa|nv|ag|gmbh|trust|holdings?|group|partners|management)\b",
    re.I,
)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def _norm_entity(name: str | None) -> str:
    """An entity name lower-cased, without punctuation or legal suffixes; '' (never a match) when unusable."""
    if not name:
        return ""
    text = _NON_ALNUM.sub(" ", str(name).lower())
    text = _ENTITY_SUFFIXES.sub(" ", text)
    return " ".join(text.split())


def _header_filers(filing: Any) -> list[tuple[str, str]]:
    """`(cik, name)` per SGML header filer, in header order; empty when the header cannot be parsed
    (never drops the filing); a transient read raises."""
    try:
        header = filing_header(filing)
    except TransientReadError:
        raise
    except Exception:  # noqa: BLE001 -- best-effort only
        return []
    out: list[tuple[str, str]] = []
    for flr in getattr(header, "filers", None) or []:
        info = getattr(flr, "company_information", None)
        if info is not None and getattr(info, "cik", None):
            out.append((str(info.cik), str(getattr(info, "name", "") or "")))
    return out


def _reporting_person_cik(rp: Any, seq: int, filers: list[tuple[str, str]], n_persons: int) -> str | None:
    """The person's own CIK, else one backfilled from the header filers: a unique exact normalized
    name match, then a unique prefix match, then position only when both lists have equal length.
    `no_cik` wins over everything (None); a missing CIK is preferred to a wrong one."""
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
    """The cover-page event date (MM/DD/YYYY first, then inferred), or None when blank or unparseable."""
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
