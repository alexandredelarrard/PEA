"""Characterization of the registrant filing walks and the owner-inclusive schedule search.

Pins CIK walk order, accession dedup, log text, the margin owner rule and the schedule window
partition/bisect result against one unsplit window. Offline: stub `Company`, stub `Filing` and a
fake SEC Atom server.
"""

from __future__ import annotations

import logging
import types
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from src.data_extract.utils.common.registrant import (
    ScheduleDiscoveryIncompleteError,
    resolve_registrant_filings,
    resolve_schedule_subject_filings,
)
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity, filing, patch_company

LOGGER = "src.data_extract.utils.common.registrant"
SUBJECT_CIK = "0001364742"

#: XOM's register windows plus one event-only CIK (99).
_XOM = dated_identity(
    [
        ("XOM", "0000034088", "cik_window", SENTINEL, "2026-07-01"),
        ("XOM", "0002115436", "cik_window", "2026-07-01", None),
        ("XOM", "0000000099", "cik_event", SENTINEL, None),
    ],
    {"XOM": "0002115436"},
)


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == LOGGER and record.levelno == level]


# --------------------------------------------------------------------------- #
# (a) UNION over event CIKs                                                    #
# --------------------------------------------------------------------------- #
def test_union_walks_every_event_cik_in_order_first_writer_wins_and_logs_contributions(monkeypatch, caplog):
    built = patch_company(
        monkeypatch,
        {
            34088: [filing("b", "2026-02-01"), filing("c", "2010-01-01"), filing("done", "2026-03-01")],
            2115436: [filing("a", "2026-08-01"), filing("b", "2026-02-01"), filing("old", "1999-01-01")],
            99: [filing("d", "2026-09-01"), filing("e", "2005-05-05")],
        },
    )
    stats: dict[str, int] = {}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(
            _XOM.filing_scope("XOM"), ["8-K", "8-K/A"], since=pd.Timestamp("2000-01-01"), done_accessions=frozenset({"done"}), stats=stats
        )

    assert [f.accession_number for f in out] == ["e", "c", "b", "a", "d"]
    assert built == [99, 34088, 2115436]
    assert stats == {"skipped_existing": 1, "foreign_skipped": 0}
    assert _messages(caplog, logging.INFO) == ["XOM: 8-K,8-K/A listed by union across 2 from 0000000099, 2 from 0000034088, 1 from 0002115436"]
    print("\n=== SANITY: UNION over event CIKs ===")
    print(f"  order {[f.accession_number for f in out]}; CIKs walked {built}; first writer wins; 'done' skipped once")


def test_a_single_cik_walk_applies_since_and_done_before_sort_and_is_silent(monkeypatch, caplog):
    identity = dated_identity([("AAPL", "0000320193", "cik_window", SENTINEL, None)], {"AAPL": "0000320193"})
    built = patch_company(
        monkeypatch, {320193: [filing("c", "2020-03-01"), filing("x", "2010-01-01"), filing("d", "2019-01-01"), filing("a", "2018-01-01")]}
    )
    stats: dict[str, int] = {}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(
            identity.filing_scope("AAPL"), ["8-K"], since=pd.Timestamp("2015-01-01"), done_accessions=frozenset({"d", "x"}), stats=stats
        )
    assert [f.accession_number for f in out] == ["a", "c"]
    assert built == [320193]
    assert stats == {"skipped_existing": 2, "foreign_skipped": 0}
    assert _messages(caplog, logging.INFO) == []
    print("\n=== SANITY: single-CIK walk -> since + done, sorted, silent ===")


# --------------------------------------------------------------------------- #
# (b) SPLIT over CIK windows                                                   #
# --------------------------------------------------------------------------- #
def test_split_walks_windows_only_and_a_dead_window_cik_warns(monkeypatch, caplog):
    built = patch_company(
        monkeypatch,
        {
            34088: [filing("pre", "2026-03-01"), filing("margin", "2026-07-15"), filing("late", "2026-09-01")],
            99: [filing("event-only", "2026-03-01")],
        },
        dead=frozenset({2115436}),
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(_XOM.filing_scope("XOM"), ["10-K"], since=None, done_accessions=frozenset())
    assert [f.accession_number for f in out] == ["pre", "margin"]
    assert built == [34088, 2115436]
    assert _messages(caplog, logging.WARNING) == ["XOM: CIK 0002115436 could not be resolved"]
    print("\n=== SANITY: SPLIT walks the windows only ===")
    print("  event-only CIK 99 never listed; predecessor keeps its 31-day margin filing; dead successor CIK warns")


def test_split_accession_admitted_by_two_windows_goes_to_its_date_owner(monkeypatch):
    shared = filing("joint", "2026-07-10")
    patch_company(monkeypatch, {34088: [shared], 2115436: [shared, filing("suc", "2026-08-01")]})
    out = resolve_registrant_filings(_XOM.filing_scope("XOM"), ["10-Q"], since=None, done_accessions=frozenset())
    assert [f.accession_number for f in out] == ["joint", "suc"]
    print("\n=== SANITY: an accession in both widened windows is listed once ===")


# --------------------------------------------------------------------------- #
# (c) schedule windows                                                          #
# --------------------------------------------------------------------------- #
class _FakeFiling:
    def __init__(self, *, cik: int, company: str, form: str, filing_date: str, accession_no: str) -> None:
        self.cik = cik
        self.company = company
        self.form = form
        self.filing_date = filing_date
        self.accession_number = accession_no
        subject = SUBJECT_CIK if not accession_no.startswith("other") else "0000320193"
        self.header = types.SimpleNamespace(subject_companies=[types.SimpleNamespace(company_information=types.SimpleNamespace(cik=subject))])


def _schedule_book() -> list[tuple[str, str, str, bool]]:
    """(accession, date, form, has_file_number): dense 2021, sparse elsewhere, 2020-01-01..2023-12-31."""
    book: list[tuple[str, str, str, bool]] = []
    for index, day in enumerate(pd.date_range("2021-01-01", "2021-12-31", freq="D")):
        form = "SC 13G/A" if index % 3 else "SC 13G"
        book.append((f"dense-{index:03d}", day.strftime("%Y-%m-%d"), form, index % 4 != 0))
    for index, day in enumerate(pd.date_range("2020-01-05", "2023-12-25", freq="21D")):
        book.append((f"sparse-{index:03d}", day.strftime("%Y-%m-%d"), "SC 13G", index % 5 != 0))
    book.append(("other-1", "2022-03-03", "SC 13G", True))
    book.append(("form-424B2", "2022-03-04", "424B2", True))
    return book


def _atom_server(book: list[tuple[str, str, str, bool]], requested: list[str]) -> Any:
    def download(url: str) -> str:
        requested.append(url)
        query = parse_qs(urlparse(url).query)
        start, count = int(query["start"][0]), int(query["count"][0])
        lo, hi = query["datea"][0], query["dateb"][0]
        rows = sorted((row for row in book if lo <= row[1].replace("-", "") <= hi), key=lambda row: row[1], reverse=True)
        parts = [
            "<entry><content>"
            f"<accession-number>{accession}</accession-number><filing-date>{date}</filing-date>"
            f"<filing-type>{form}</filing-type>{'<file-number>005-1</file-number>' if numbered else ''}"
            "</content></entry>"
            for accession, date, form, numbered in rows[start : start + count]
        ]
        return f'<feed xmlns="http://www.w3.org/2005/Atom">{"".join(parts)}</feed>'

    return download


def _run_schedule(monkeypatch, caplog, safe_offset: int) -> tuple[list[str], list[str], list[str]]:
    requested: list[str] = []
    monkeypatch.setattr("edgar.Filing", _FakeFiling)
    monkeypatch.setattr("edgar.httprequests.download_text", _atom_server(_schedule_book(), requested))
    monkeypatch.setattr("src.data_extract.utils.common.registrant.SCHEDULE_ATOM_SAFE_OFFSET", safe_offset)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        filings = resolve_schedule_subject_filings(
            "BLK",
            frozenset({SUBJECT_CIK}),
            ["SC 13G", "SC 13G/A"],
            since=pd.Timestamp("2020-01-01"),
            through=pd.Timestamp("2023-12-31"),
            done_accessions=frozenset({"dense-001"}),
        )
    return [cast(Any, filing).accession_number for filing in filings], requested, _messages(caplog, logging.INFO)


def test_schedule_partition_and_bisect_match_one_big_window(monkeypatch, caplog):
    big, big_requests, big_logs = _run_schedule(monkeypatch, caplog, safe_offset=10_000)
    split, split_requests, split_logs = _run_schedule(monkeypatch, caplog, safe_offset=100)

    assert big == split
    assert len(big) == 328
    assert big_requests[0] == (
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001364742&type=SC+13G"
        "&datea=20200101&dateb=20231231&owner=include&start=0&count=100&output=atom"
    )
    assert len(big_requests) == 5
    assert [
        parse_qs(urlparse(url).query)["datea"][0] + ".." + parse_qs(urlparse(url).query)["dateb"][0] + "@" + parse_qs(urlparse(url).query)["start"][0]
        for url in split_requests
    ] == [
        "20200101..20231231@0",
        "20200101..20201231@0",
        "20210101..20211231@0",
        "20210101..20210702@0",
        "20210101..20210402@0",
        "20210403..20210702@0",
        "20210703..20211231@0",
        "20210703..20211001@0",
        "20211002..20211231@0",
        "20220101..20221231@0",
        "20230101..20231231@0",
    ]
    assert split_logs[0] == (
        "BLK: schedule search reached offset 100 for subject CIK 0001364742, form SC 13G; "
        "partitioning 2020-01-01..2023-12-31 into complete one-year windows"
    )
    assert split_logs[1] == (
        "BLK: schedule search reached offset 100 for subject CIK 0001364742, form SC 13G; bisecting 2021-01-01..2021-12-31 into complete date windows"
    )
    assert big_logs[-1] == (
        "BLK: subject-first schedules -- 5 page(s), 106 owner-side row(s) excluded from Atom metadata, "
        "329 candidate(s), 328 subject match(es), 0 unknown header(s), 328 retained for full parsing"
    )
    assert split_logs[-1] == (
        "BLK: subject-first schedules -- 11 page(s), 201 owner-side row(s) excluded from Atom metadata, "
        "329 candidate(s), 328 subject match(es), 0 unknown header(s), 328 retained for full parsing"
    )
    print("\n=== SANITY: schedule window split ===")
    print(f"  one window: {len(big_requests)} pages; split: {len(split_requests)} pages; same {len(big)} accessions")
    print("  owner rows on discarded pre-split pages stay counted (106 -> 201); page order pinned")


def test_schedule_unsplittable_day_is_incomplete(monkeypatch):
    book = [(f"same-{index:03d}", "2022-05-05", "SC 13G", True) for index in range(150)]
    requested: list[str] = []
    monkeypatch.setattr("edgar.Filing", _FakeFiling)
    monkeypatch.setattr("edgar.httprequests.download_text", _atom_server(book, requested))
    monkeypatch.setattr("src.data_extract.utils.common.registrant.SCHEDULE_ATOM_SAFE_OFFSET", 100)
    with pytest.raises(ScheduleDiscoveryIncompleteError, match="inside the unsplittable date 2022-05-05 for subject CIK 0001364742, form SC 13G"):
        resolve_schedule_subject_filings(
            "BLK",
            frozenset({SUBJECT_CIK}),
            ["SC 13G"],
            since=pd.Timestamp("2022-05-05"),
            through=pd.Timestamp("2022-05-05"),
            done_accessions=frozenset(),
        )
    print("\n=== SANITY: one day over the safe offset -> hard incomplete ===")


def test_schedule_requires_a_finite_start(monkeypatch):
    monkeypatch.setattr("edgar.httprequests.download_text", lambda url: pytest.fail("no request without a start date"))
    with pytest.raises(ScheduleDiscoveryIncompleteError, match="requires a finite start date"):
        resolve_schedule_subject_filings("BLK", frozenset({SUBJECT_CIK}), ["SC 13G"], since=None, done_accessions=frozenset())
    print("\n=== SANITY: no start date -> hard incomplete before any request ===")


def test_schedule_empty_payload_is_incomplete(monkeypatch):
    monkeypatch.setattr("edgar.httprequests.download_text", lambda url: None)
    with pytest.raises(ScheduleDiscoveryIncompleteError) as raised:
        resolve_schedule_subject_filings(
            "BLK",
            frozenset({SUBJECT_CIK}),
            ["SC 13G"],
            since=pd.Timestamp("2022-01-01"),
            through=pd.Timestamp("2022-02-01"),
            done_accessions=frozenset(),
        )
    assert (
        str(raised.value)
        == "BLK: schedule search failed for subject CIK 0001364742, form SC 13G, offset 0: ValueError('SEC Atom response was empty')"
    )
    print("\n=== SANITY: empty Atom payload -> hard incomplete with the pinned message ===")
