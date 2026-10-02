"""`FilingStamp`: the one filing-level stamp every EDGAR fetcher writes, and its
`period_of_report` guard that must never abort a walk.

edgartools' own `Filing.period_of_report` property can raise `TypeError` from inside itself on
old submissions whose homepage metadata does not parse. `TypeError` is in `PROGRAMMING_ERRORS`,
which `run_per_ticker` re-raises deliberately, so one unparseable filing would abort a whole
multi-ticker run unless the read is guarded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.common.parallel_fetch import PROGRAMMING_ERRORS


class _Raising:
    """A filing whose `period_of_report` raises, exactly as edgartools' does."""

    accession_number = "0000950129-04-000328"
    form = "8-K"
    filing_date = pd.Timestamp("2004-01-30").date()
    cik = 808362

    @property
    def period_of_report(self) -> Any:
        broken: Any = None
        _, _, period = broken  # what attachments.py:1170 really does
        return period


class _Counting:
    """A filing that counts how often its `period_of_report` property is read."""

    accession_number = "0000320193-24-000123"
    form = "10-K"
    filing_date = pd.Timestamp("2024-11-01").date()
    cik = 320193
    primary_document = "aapl-20240928.htm"

    def __init__(self) -> None:
        self.reads = 0

    @property
    def period_of_report(self) -> str:
        self.reads += 1
        return "2024-09-28"


def _filing(**overrides: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "accession_number": "0001193125-25-000001",
        "form": "SC 13D",
        "filing_date": "2025-01-15",
        "cik": "0001326380",
        "primary_document": "sc13d.htm",
        "document": None,
    }
    return SimpleNamespace(**{**base, **overrides})


def test_the_raising_property_is_a_programming_error_and_would_abort_the_run() -> None:
    """The premise. If this ever stops holding, the guard below is no longer load-bearing."""
    with pytest.raises(PROGRAMMING_ERRORS):
        _ = _Raising().period_of_report
    print("\n=== SANITY: the premise ===")
    print("  filing.period_of_report raises TypeError, which run_per_ticker re-raises by design.")


def test_getattr_with_a_default_does_not_guard_it() -> None:
    """`getattr`'s default only answers AttributeError -- a raising property passes straight through it."""
    with pytest.raises(TypeError):
        getattr(_Raising(), "period_of_report", None)
    print("\n=== SANITY: why getattr is not enough ===")
    print("  getattr(..., None) re-raises the TypeError; only try/except swallows it.")


def test_the_stamp_guard_returns_none_instead() -> None:
    stamp = FilingStamp.of(_Raising(), "0000808362")
    assert stamp.period_of_report is None
    assert stamp.cik == "0000808362"
    print("\n=== SANITY: the guard ===")
    print("  FilingStamp.period_of_report -> None, so the walk keeps its other tickers.")


def test_the_stamp_guard_is_transparent_when_the_property_works() -> None:
    assert FilingStamp.of(_filing(period_of_report="2024-12-31"), "").period_of_report == "2024-12-31"
    assert FilingStamp.of(_filing(), "").period_of_report is None  # absent attribute, not an error
    print("\n=== SANITY: transparent guard ===")
    print("  a working property passes through unchanged; an absent one reads as None.")


def test_period_of_report_is_read_once_per_filing() -> None:
    filing = _Counting()
    stamp = FilingStamp.of(filing, "")
    assert filing.reads == 0  # lazy: building the stamp does not touch the property
    values = {stamp.period_of_report for _ in range(5)}
    assert values == {"2024-09-28"}
    assert filing.reads == 1
    print("\n=== SANITY: one read ===")
    print(f"  5 stamp reads -> {filing.reads} property access; a fetcher stamping hundreds of rows pays once.")


def test_filer_cik_wins_and_roster_cik_is_only_the_fallback() -> None:
    assert FilingStamp.of(_filing(cik=34088), "0002115436").cik == "0000034088"
    assert FilingStamp.of(_filing(cik=None), "2115436").cik == "0002115436"
    assert FilingStamp.of(SimpleNamespace(accession_number="a", form="8-K", filing_date="2024-01-02"), "320193").cik == "0000320193"
    print("\n=== SANITY: filer CIK ===")
    print("  the filing's own CIK is stored (padded); the roster CIK fills in only when the filing exposes none.")


def test_amendment_flag_and_raw_filing_date() -> None:
    original = FilingStamp.of(_filing(form="SC 13D", filing_date="2025-01-15"), "")
    amended = FilingStamp.of(_filing(form="sc 13d/a"), "")
    assert (original.is_amendment, amended.is_amendment) == (False, True)
    assert original.filed == pd.Timestamp("2025-01-15")
    assert FilingStamp.of(_filing(filing_date=pd.Timestamp("2025-01-15 16:30")), "").filed.hour == 16  # not normalised
    print("\n=== SANITY: amendment + date ===")
    print("  '/A' (any case) marks an amendment; `filed` is the raw Timestamp, callers normalise when their table stores a date.")


def test_doc_url_prefers_the_attachment_then_the_archives_path() -> None:
    attached = FilingStamp.of(_filing(document=SimpleNamespace(url="https://www.sec.gov/Archives/x/y.htm")), "")
    assert attached.doc_url == "https://www.sec.gov/Archives/x/y.htm"

    class _BoxedDocument:
        url = None

        def __str__(self) -> str:
            return "+------+\n| 1 sc13d.htm |\n+------+"

    composed = FilingStamp.of(_filing(document=_BoxedDocument()), "")
    assert composed.doc_url == "https://www.sec.gov/Archives/edgar/data/1326380/000119312525000001/sc13d.htm"
    assert FilingStamp.of(_filing(primary_document=None), "").doc_url is None  # no primary document -> no URL
    assert FilingStamp.of(_filing(cik=None), "").doc_url is None  # no CIK at all -> no URL
    print("\n=== SANITY: doc_url ===")
    print("  attachment url verbatim, else the archives path from cik/accession/primary document, else None (never a rendered table).")
