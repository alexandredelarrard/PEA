"""
schedule_rows.py (src/data_extract/utils/institutionals/schedule_rows.py)
-------------------------------------------------------------------------
One Schedule 13D/13G row builder, the subject test and the issuer-guarded per-filing parse. A
`ScheduleSpec` carries what differs between the two forms (columns, form-specific filing fields,
numeric trust, reporting-person CIK, event-date parse, blank normalisation); each form module
declares one spec. The local index lists a schedule under every party, so a schedule on which the
ticker is only a filer is rejected (an empty-filing marker) before its document is parsed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data_extract.utils.common.edgar_driver import EdgarScope, FilingStamp, num_or_null
from src.data_extract.utils.common.frame_sanitize import finalise_frame
from src.data_extract.utils.common.parallel_fetch import PROGRAMMING_ERRORS
from src.data_extract.utils.common.registrant import header_subject_ciks
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, filing_obj
from src.data_store.schema import Table
from src.utils.string import pad_cik

SCHEDULE_NUMERIC_COLS = (
    "sole_voting_power",
    "shared_voting_power",
    "sole_dispositive_power",
    "shared_dispositive_power",
    "aggregate_amount",
    "percent_of_class",
)


@dataclass(frozen=True)
class ScheduleSpec:
    """The per-form rules of a schedule walk.

    `label` prefixes the parse-failure message. `extra_base(filing, obj, has_structured)` returns the
    form's own filing-level fields; `trust_numerics(rp, has_structured)` decides whether a reporting
    person's numerics are disclosed values; `reporting_person_ciks(filing, persons)` returns one CIK
    per person; `event_date(obj)` parses the cover-page event date. `blank_to_none` stores a blank
    CUSIP or reporting-person name as None instead of ''.
    """

    label: str
    forms: tuple[str, ...]
    columns: tuple[str, ...]
    extra_base: Callable[[Any, Any, bool], dict[str, Any]]
    trust_numerics: Callable[[Any, bool], bool]
    reporting_person_ciks: Callable[[Any, list[Any]], list[str | None]]
    event_date: Callable[[Any], pd.Timestamp | None]
    blank_to_none: bool


def _blank(value: Any, spec: ScheduleSpec) -> Any:
    return (value or None) if spec.blank_to_none else value


def _base_fields(stamp: FilingStamp, obj: Any, has_structured: bool, spec: ScheduleSpec) -> dict[str, Any]:
    """The filing-level fields shared by every row of one schedule."""
    issuer = getattr(obj, "issuer_info", None)
    security = getattr(obj, "security_info", None)
    return {
        "cik": getattr(issuer, "cik", None) if issuer else None,
        "issuer_name": getattr(issuer, "name", None) if issuer else None,
        "accession_number": stamp.accession_number,
        "form": stamp.form,
        "filing_date": stamp.filed,
        "date_of_event": spec.event_date(obj),
        "is_amendment": 1.0 if bool(getattr(obj, "is_amendment", False)) else 0.0,
        "amendment_number": num_or_null(getattr(obj, "amendment_number", None), True),
        "cusip": _blank(getattr(security, "cusip", None) if security else None, spec),
        "has_structured_data": 1.0 if has_structured else 0.0,
        "primary_document": stamp.primary_document,
        "doc_url": stamp.doc_url,
        **spec.extra_base(stamp.filing, obj, has_structured),
    }


def _fallback_row(base: dict[str, Any]) -> dict[str, Any]:
    """The single row of a schedule with no parsed reporting person. Numerics are NaN, not None, so
    a batch of these still seeds a float column (`ensure_table` infers all-None as TEXT)."""
    return {
        **base,
        "rp_seq": 0,
        "reporting_person_name": None,
        "reporting_person_cik": None,
        "reporting_person_citizenship": None,
        "type_of_reporting_person": None,
        "reporting_person_comment": None,
        "is_group_member": None,
        **{col: float("nan") for col in SCHEDULE_NUMERIC_COLS},
    }


def schedule_filing_rows(stamp: FilingStamp, spec: ScheduleSpec) -> list[dict[str, Any]]:
    """One schedule -> one row per reporting person (one fallback row when none parsed)."""
    obj = filing_obj(stamp.filing)
    has_structured = bool(getattr(obj, "has_structured_data", False))
    base = _base_fields(stamp, obj, has_structured, spec)
    persons = getattr(obj, "reporting_persons", None) or []
    if not persons:
        return [_fallback_row(base)]
    ciks = spec.reporting_person_ciks(stamp.filing, persons)
    rows = []
    for seq, (rp, rp_cik) in enumerate(zip(persons, ciks, strict=True)):
        trusted = spec.trust_numerics(rp, has_structured)
        rows.append(
            {
                **base,
                "rp_seq": seq,
                "reporting_person_name": _blank(getattr(rp, "name", None), spec),
                "reporting_person_cik": rp_cik,
                "reporting_person_citizenship": getattr(rp, "citizenship", None) or None,
                "type_of_reporting_person": getattr(rp, "type_of_reporting_person", None) or None,
                "reporting_person_comment": getattr(rp, "comment", None) or None,
                "is_group_member": getattr(rp, "member_of_group", None),
                **{col: num_or_null(getattr(rp, col, None), trusted) for col in SCHEDULE_NUMERIC_COLS},
            }
        )
    return rows


def schedule_is_subject(ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope) -> bool:
    """False when the SGML header names subject companies and none is a CIK that is the ticker's company on the filing date.

    A header without subject companies defers to the issuer guard of `schedule_ticker_rows`; an
    unreadable header (an SEC error page) raises `TransientReadError`."""
    subjects = header_subject_ciks(stamp.filing)
    return not subjects or not subjects.isdisjoint(scope.filing_scope(ticker, cik).ciks_on(stamp.filed))


def schedule_ticker_rows(ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope, spec: ScheduleSpec) -> list[dict[str, Any]]:
    """One schedule's rows stamped with `ticker`; [] when the parsed issuer CIK is not the ticker's company on the filing date.

    An unresolvable CIK on either side means unknown and does not reject. A parse failure raises
    `ParseFailureError` naming the accession; a transient SEC failure raises `TransientReadError`."""
    try:
        rows = schedule_filing_rows(stamp, spec)
    except (TransientReadError, *PROGRAMMING_ERRORS):
        raise
    except Exception as exc:  # noqa: BLE001 -- filing parser boundary
        raise ParseFailureError(f"{spec.label} accession {stamp.accession_number} could not be parsed") from exc
    ticker_ciks = scope.filing_scope(ticker, cik).ciks_on(stamp.filed)
    issuer_cik = pad_cik(rows[0].get("cik")) if rows else ""
    if ticker_ciks and issuer_cik and issuer_cik not in ticker_ciks:
        return []
    for row in rows:
        row["ticker"] = ticker
    return rows


def parse_schedule(ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope, *, spec: ScheduleSpec, table: Table) -> dict[Table, pd.DataFrame]:
    """A schedule fetch's `parse`: one `table` frame finalised by `finalise_frame`."""
    return {table: finalise_frame(table, schedule_ticker_rows(ticker, cik, stamp, scope, spec), columns=spec.columns)}
