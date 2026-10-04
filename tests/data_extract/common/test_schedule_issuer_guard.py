"""The issuer/filer guard on 13D/13G accepts EVERY CIK of the ticker's filing scope, not just the roster's.

The 13G build (`schedule_rows.kept_schedule_filings`) keeps a schedule only when the issuer CIK read
off the filing is one of the ticker's scope CIKs. A single-CIK comparison against a multi-CIK listing
rejects exactly the pre-boundary schedules the lineage exists to recover -- measured as `sec-13d`
storing +0 rows after resolving MDT 28, BLK 50, VTRS 16+6 and ICE 10 predecessor filings.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from src.data_extract.utils.common.edgar_driver import EdgarScope
from src.data_extract.utils.common.registrant import (
    SCHEDULE_SUBJECT_CAP,
    ScheduleDiscoveryIncompleteError,
    filter_schedule_subject_filings,
    resolve_schedule_subject_filings,
)
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity

#: VTRS as the register holds it: Mylan Inc -> Mylan N.V. -> Viatris.
_VTRS = EdgarScope(
    dated_identity(
        [
            ("VTRS", "0000069499", "cik_window", SENTINEL, "2015-05-01"),
            ("VTRS", "0001623613", "cik_window", "2015-05-01", "2020-11-07"),
            ("VTRS", "0001792044", "cik_window", "2020-11-07", None),
        ],
        {"VTRS": "0001792044", "AAPL": "0000320193"},
    )
)


def _subjects(ticker: str, cik: str) -> frozenset[str]:
    return frozenset(_VTRS.filing_scope(ticker, cik).event_ciks)


def test_every_scope_cik_identifies_the_ticker_as_issuer():
    got = _subjects("VTRS", "0001792044")
    assert got == {"0000069499", "0001623613", "0001792044"}
    print("\n=== SANITY: the widened guard ===")
    print(f"  VTRS accepts {len(got)} issuer CIKs, one per window: {sorted(got)}")


def test_a_predecessor_schedule_is_not_rejected():
    """A 2012 schedule about Mylan Inc carries issuer CIK 0000069499, not the roster's 0001792044."""
    assert "0000069499" in _subjects("VTRS", "0001792044")
    print("\n=== SANITY: a predecessor issuer CIK is in the subject set ===")


def test_an_unrelated_issuer_is_still_rejected():
    """A schedule VTRS FILED about Apple keeps Apple's issuer CIK and must still be dropped."""
    assert "0000320193" not in _subjects("VTRS", "0001792044")
    print("\n=== SANITY: an unrelated issuer stays outside ===")


def test_a_single_cik_ticker_keeps_exactly_its_roster_cik():
    assert _subjects("AAPL", "0000320193") == {"0000320193"}
    assert EdgarScope().filing_scope("AAPL", "320193").event_ciks == ("0000320193",)
    print("\n=== SANITY: a single-CIK scope (with or without identity) is its roster CIK ===")


def _candidate(accession: str, subject_cik: str):
    return SimpleNamespace(
        accession_number=accession,
        header=SimpleNamespace(subject_companies=[SimpleNamespace(company_information=SimpleNamespace(cik=subject_cik))]),
    )


def test_subject_filter_finds_wanted_filing_beyond_the_old_2000_cap():
    """The broad holder book may be huge; the cap applies only after issuer filtering."""
    unrelated = [_candidate(f"other-{index}", "0000320193") for index in range(2_500)]
    wanted = _candidate("0001306550-23-008694", "0001364742")
    kept, stats = filter_schedule_subject_filings(
        [*unrelated, wanted],
        frozenset({"0001364742", "0002012383"}),
    )
    assert [cast(Any, filing).accession_number for filing in kept] == ["0001306550-23-008694"]
    assert stats == {"candidates": 2_501, "subject_matches": 1, "unknown_headers": 0}

    print("\n=== SANITY: schedule subject-first filtering ===")
    print("  wanted BLK accession at position 2,501 -> retained once; 2,500 filer-side rows dropped")
    print("  OK: broad-list size no longer causes an issuer-side history skip")


def test_post_filter_cap_aborts_as_incomplete_instead_of_returning_partial_data():
    candidates = [_candidate(f"wanted-{index}", "0001364742") for index in range(SCHEDULE_SUBJECT_CAP + 1)]
    with pytest.raises(ScheduleDiscoveryIncompleteError, match="after header filtering"):
        filter_schedule_subject_filings(candidates, frozenset({"0001364742"}))

    print("\n=== SANITY: post-filter safety cap ===")
    print(f"  {SCHEDULE_SUBJECT_CAP + 1} issuer matches -> hard incomplete failure")
    print("  OK: a limit can never masquerade as a successful empty result")


def test_owner_inclusive_search_filters_header_before_full_object(monkeypatch):
    feed = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><content><accession-number>0001306550-23-008694</accession-number>
      <filing-date>2023-02-14</filing-date><filing-type>SC 13G/A</filing-type>
      <file-number>005-82091</file-number></content></entry>
      <entry><content><accession-number>unrelated</accession-number>
      <filing-date>2023-02-15</filing-date><filing-type>SC 13G/A</filing-type></content></entry>
    </feed>"""

    class Filing:
        def __init__(self, *, cik, company, form, filing_date, accession_no):
            del cik, company
            self.form = form
            self.filing_date = filing_date
            self.accession_number = accession_no
            subject = "0001364742" if accession_no == "0001306550-23-008694" else "0000320193"
            self.header = SimpleNamespace(subject_companies=[SimpleNamespace(company_information=SimpleNamespace(cik=subject))])

        def obj(self):
            raise AssertionError("subject filtering must happen before obj()")

    monkeypatch.setattr("edgar.Filing", Filing)
    monkeypatch.setattr("edgar.httprequests.download_text", lambda url: feed)
    filings = resolve_schedule_subject_filings(
        "BLK",
        frozenset({"0001364742"}),
        ["SC 13G", "SC 13G/A"],
        since=pd.Timestamp("2022-01-01"),
        through=pd.Timestamp("2023-12-31"),
        done_accessions=frozenset(),
    )
    assert [cast(Any, filing).accession_number for filing in filings] == ["0001306550-23-008694"]


def test_failed_owner_inclusive_page_is_a_hard_incomplete_outcome(monkeypatch):
    monkeypatch.setattr(
        "edgar.httprequests.download_text",
        lambda url: (_ for _ in ()).throw(TimeoutError("SEC timeout")),
    )
    with pytest.raises(ScheduleDiscoveryIncompleteError, match="offset 0"):
        resolve_schedule_subject_filings(
            "BLK",
            frozenset({"0001364742"}),
            ["SC 13G", "SC 13G/A"],
            since=pd.Timestamp("2022-01-01"),
            through=pd.Timestamp("2023-12-31"),
            done_accessions=frozenset(),
        )


def test_transient_owner_inclusive_page_is_retried_before_success(monkeypatch):
    feed = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><content><accession-number>0001306550-23-008694</accession-number>
      <filing-date>2023-02-14</filing-date><filing-type>SC 13G/A</filing-type>
      <file-number>005-82091</file-number></content></entry>
    </feed>"""
    attempts = 0

    class Filing:
        def __init__(self, *, cik, company, form, filing_date, accession_no):
            del cik, company
            self.form = form
            self.filing_date = filing_date
            self.accession_number = accession_no
            self.header = SimpleNamespace(subject_companies=[SimpleNamespace(company_information=SimpleNamespace(cik="0001364742"))])

    def transient_then_success(url):
        del url
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("503 Service Unavailable")
        return feed

    monkeypatch.setattr("edgar.Filing", Filing)
    monkeypatch.setattr("edgar.httprequests.download_text", transient_then_success)
    monkeypatch.setattr("src.data_extract.utils.common.rate_limit.time.sleep", lambda seconds: None)
    filings = resolve_schedule_subject_filings(
        "BLK",
        frozenset({"0001364742"}),
        ["SC 13G", "SC 13G/A"],
        since=pd.Timestamp("2022-01-01"),
        through=pd.Timestamp("2023-12-31"),
        done_accessions=frozenset(),
    )

    assert attempts == 3
    assert [cast(Any, filing).accession_number for filing in filings] == ["0001306550-23-008694"]
    print("\n=== SANITY: transient SEC page retry ===")
    print("  two HTTP 503 failures -> third attempt succeeded")
    print("  OK: the complete BLK issuer result is retained without advancing a partial run")


def test_deep_owner_book_is_bisected_by_date_without_requesting_unsafe_offset(monkeypatch):
    def feed(entries: list[tuple[str, str]]) -> str:
        parts: list[str] = []
        for accession, date in entries:
            file_number = "<file-number>005-82091</file-number>" if accession == "0001306550-23-008694" else ""
            parts.append(
                "<entry><content>"
                f"<accession-number>{accession}</accession-number>"
                f"<filing-date>{date}</filing-date>"
                "<filing-type>SC 13G/A</filing-type>"
                f"{file_number}</content></entry>"
            )
        return f'<feed xmlns="http://www.w3.org/2005/Atom">{"".join(parts)}</feed>'

    requested: list[dict[str, list[str]]] = []

    def download(url):
        query = parse_qs(urlparse(url).query)
        requested.append(query)
        if query["datea"] == ["20220101"] and query["dateb"] == ["20231231"]:
            return feed([(f"broad-{index}", "2023-01-01") for index in range(100)])
        if query["datea"] == ["20220101"]:
            return feed([("0001306550-23-008694", "2022-06-30")])
        return feed([])

    class Filing:
        def __init__(self, *, cik, company, form, filing_date, accession_no):
            del cik, company
            self.form = form
            self.filing_date = filing_date
            self.accession_number = accession_no
            self.header = SimpleNamespace(subject_companies=[SimpleNamespace(company_information=SimpleNamespace(cik="0001364742"))])

    monkeypatch.setattr("edgar.Filing", Filing)
    monkeypatch.setattr("edgar.httprequests.download_text", download)
    monkeypatch.setattr("src.data_extract.utils.common.registrant.SCHEDULE_ATOM_SAFE_OFFSET", 100)
    filings = resolve_schedule_subject_filings(
        "BLK",
        frozenset({"0001364742"}),
        ["SC 13G", "SC 13G/A"],
        since=pd.Timestamp("2022-01-01"),
        through=pd.Timestamp("2023-12-31"),
        done_accessions=frozenset(),
    )

    assert [cast(Any, filing).accession_number for filing in filings] == ["0001306550-23-008694"]
    assert len(requested) == 3
    assert {query["start"][0] for query in requested} == {"0"}
    print("\n=== SANITY: SEC deep-pagination split ===")
    print("  full range filled the safe page budget -> two complete date windows queried")
    print("  OK: wanted issuer filing recovered without requesting the failing deep offset")
