"""The issuer/filer guard on 13D/13G must accept EVERY segment CIK, not just the roster's.

Regression test for a defect the register introduced rather than exposed. `build_ticker_13g_edgar`
keeps a schedule only when the issuer CIK read off the filing matches the ticker's own. Before the
register that was a single-CIK test against a single-CIK listing, and consistent. Widening the
listing to every segment without widening the comparison rejects exactly the pre-boundary
schedules the register exists to recover -- measured as `sec-13d` storing +0 rows after resolving
MDT 28, BLK 50, VTRS 16+6 and ICE 10 predecessor filings.
"""

from __future__ import annotations

from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from src.data_extract.utils.common.registrant import (
    SCHEDULE_SUBJECT_CAP,
    Registrant,
    ScheduleDiscoveryIncompleteError,
    Segment,
    filter_schedule_subject_filings,
    issuer_ciks,
    resolve_schedule_subject_filings,
)


def _reg() -> dict[str, Registrant]:
    """VTRS as the register holds it: a three-segment chain."""
    segs = (
        Segment(cik="0000069499", valid_from=None, valid_to=pd.Timestamp("2015-05-01"), evidence="Mylan Inc"),
        Segment(cik="0001623613", valid_from=pd.Timestamp("2015-05-01"), valid_to=pd.Timestamp("2020-11-07"), evidence="Mylan N.V."),
        Segment(cik="0001792044", valid_from=pd.Timestamp("2020-11-07"), valid_to=None, evidence="Viatris"),
    )
    return {"VTRS": Registrant(ticker="VTRS", kind="reorganisation", segments=segs)}


def test_every_segment_cik_identifies_the_ticker_as_issuer():
    got = issuer_ciks("VTRS", "0001792044", _reg())
    assert got == {"0000069499", "0001623613", "0001792044"}
    print("\n=== SANITY: the widened guard ===")
    print(f"  VTRS accepts {len(got)} issuer CIKs, one per segment: {sorted(got)}")


def test_a_predecessor_schedule_is_no_longer_rejected():
    """The exact rejection that produced +0. A 2012 schedule about Mylan Inc carries issuer CIK
    0000069499; the old test compared it to the roster's 0001792044 and dropped it."""
    accepted = issuer_ciks("VTRS", "0001792044", _reg())
    predecessor_issuer = "0000069499"
    assert predecessor_issuer != "0001792044"  # the old single-CIK test
    assert predecessor_issuer in accepted  # the new one


def test_an_unrelated_issuer_is_still_rejected():
    """The guard must not become a pass-through: a schedule VTRS FILED about Apple keeps
    Apple's issuer CIK and must still be dropped."""
    assert "0000320193" not in issuer_ciks("VTRS", "0001792044", _reg())


def test_a_ticker_with_no_register_entry_keeps_exactly_its_roster_cik():
    assert issuer_ciks("AAPL", "0000320193", _reg()) == {"0000320193"}


def test_the_roster_cik_is_kept_even_when_it_is_not_a_segment():
    """XOM's roster CIK was the holdco while the register named the predecessor. Dropping the
    roster CIK would have discarded rows the pipeline already resolved correctly."""
    reg = _reg()
    assert issuer_ciks("VTRS", "0009999999", reg) == {"0000069499", "0001623613", "0001792044", "0009999999"}


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
    assert [filing.accession_number for filing in kept] == ["0001306550-23-008694"]
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
    assert [filing.accession_number for filing in filings] == ["0001306550-23-008694"]


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
    assert [filing.accession_number for filing in filings] == ["0001306550-23-008694"]
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

    assert [filing.accession_number for filing in filings] == ["0001306550-23-008694"]
    assert len(requested) == 3
    assert {query["start"][0] for query in requested} == {"0"}
    print("\n=== SANITY: SEC deep-pagination split ===")
    print("  full range filled the safe page budget -> two complete date windows queried")
    print("  OK: wanted issuer filing recovered without requesting the failing deep offset")
