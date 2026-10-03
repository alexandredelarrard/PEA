"""Characterization of the insider owner-inclusive Atom paging policy.

Pins the request URLs, the stop on a page whose oldest entry predates `since`, the stop of one
form family on a failed page (other families continue) and the `since=None` URL. Offline.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from src.data_extract.utils.institutionals import fetch_insider_edgar as module

URL_PREFIX = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001494877"


def _entry(accession: str, date: str, form: str) -> str:
    return (
        "<entry><content>"
        f"<accession-number>{accession}</accession-number><filing-date>{date}</filing-date>"
        f"<filing-type>{form}</filing-type></content></entry>"
    )


def _page(entries: list[str]) -> str:
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{"".join(entries)}</feed>'


def _install(monkeypatch: pytest.MonkeyPatch, pages: dict[tuple[str, int], Any], requested: list[str]) -> None:
    """Serve `pages[(family, start)]`; an Exception value is raised, a missing key is an empty feed."""

    def download(url: str) -> str:
        requested.append(url)
        query = parse_qs(urlparse(url).query)
        page = pages.get((query["type"][0], int(query["start"][0])), _page([]))
        if isinstance(page, Exception):
            raise page
        return page

    monkeypatch.setattr("edgar.httprequests.download_text", download)


def _full_page(prefix: str, first_day: str, form: str = "4") -> str:
    days = pd.date_range(end=first_day, periods=100, freq="D")[::-1]
    return _page([_entry(f"{prefix}-{index:03d}", day.strftime("%Y-%m-%d"), form) for index, day in enumerate(days)])


def test_family_stops_when_the_oldest_entry_predates_since(monkeypatch):
    requested: list[str] = []
    pages = {("4", 0): _full_page("p0", "2026-06-30"), ("4", 100): _full_page("p1", "2026-03-22")}
    _install(monkeypatch, pages, requested)

    filings = module.ownership_filings(
        "DLR", "1494877", since=pd.Timestamp("2026-03-01"), through=pd.Timestamp("2026-06-30"), done_accessions=frozenset({"p0-000"})
    )

    assert requested == [
        f"{URL_PREFIX}&type=3&datea=20260301&dateb=20260630&owner=include&start=0&count=100&output=atom",
        f"{URL_PREFIX}&type=4&datea=20260301&dateb=20260630&owner=include&start=0&count=100&output=atom",
        f"{URL_PREFIX}&type=4&datea=20260301&dateb=20260630&owner=include&start=100&count=100&output=atom",
        f"{URL_PREFIX}&type=5&datea=20260301&dateb=20260630&owner=include&start=0&count=100&output=atom",
    ]
    accessions = [filing.accession_number for filing in filings]
    assert len(accessions) == 121
    assert accessions[0] == "p1-021" and accessions[-1] == "p0-001"
    assert all(filing.cik == 1494877 and filing.company == "DLR" for filing in filings)
    print("\n=== SANITY: insider Atom paging ===")
    print("  page 0 full and inside the window -> page 1 requested; page 1 reaches before since -> family stops")
    print(f"  {len(accessions)} filings kept (since filter + done accession); padded CIK in the URL, int CIK on the Filing")


def test_full_page_reaching_before_since_does_not_request_the_next_page(monkeypatch):
    requested: list[str] = []
    pages = {("4", 0): _full_page("p0", "2026-04-30")}
    _install(monkeypatch, pages, requested)

    filings = module.ownership_filings(
        "DLR", "1494877", since=pd.Timestamp("2026-04-01"), through=pd.Timestamp("2026-06-30"), done_accessions=frozenset()
    )

    assert [parse_qs(urlparse(url).query)["start"][0] for url in requested] == ["0", "0", "0"]
    assert len(filings) == 30
    print("\n=== SANITY: oldest entry before since -> no page 1 ===")


def test_failed_page_stops_only_that_family_and_keeps_earlier_pages(monkeypatch, caplog):
    requested: list[str] = []
    pages = {
        ("4", 0): _full_page("p0", "2026-06-30"),
        ("4", 100): RuntimeError("503 Service Unavailable"),
        ("5", 0): _page([_entry("five", "2026-05-05", "5")]),
    }
    _install(monkeypatch, pages, requested)

    with caplog.at_level(logging.WARNING, logger=module.__name__):
        filings = module.ownership_filings("DLR", "1494877", since=None, through=pd.Timestamp("2026-06-30"), done_accessions=frozenset())

    assert requested[0] == f"{URL_PREFIX}&type=3&datea=&dateb=20260630&owner=include&start=0&count=100&output=atom"
    assert [parse_qs(urlparse(url).query)["start"][0] for url in requested] == ["0", "0", "100", "0"]
    assert len(filings) == 101
    assert "five" in {filing.accession_number for filing in filings}
    assert [record.getMessage() for record in caplog.records] == [
        "ownership filing search failed for DLR form 4 at offset 100: RuntimeError('503 Service Unavailable')"
    ]
    print("\n=== SANITY: failed Atom page ===")
    print("  family 4 stops at offset 100 with a warning; its page 0 and family 5 are kept (silent truncation is the known TODO)")
