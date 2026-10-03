"""Characterization of the registrant filing walks and the owner-inclusive schedule search.

Pins provenance order, accession dedup, log text, the SPLIT duplicate-accession warning and the
schedule window partition/bisect result against one unsplit window. Offline: stub `Company`,
stub `Filing` and a fake SEC Atom server.
"""

from __future__ import annotations

import logging
import types
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.common.registrant import (
    Registrant,
    ScheduleDiscoveryIncompleteError,
    Segment,
    resolve_registrant_filings,
    resolve_schedule_subject_filings,
)

LOGGER = "src.data_extract.utils.common.registrant"
BOUNDARY = pd.Timestamp("2026-07-01")
SUBJECT_CIK = "0001364742"


def _filing(accession: str, filing_date: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(accession_number=accession, filing_date=filing_date)


def _patch_company(monkeypatch: pytest.MonkeyPatch, by_key: dict[Any, list], dead: frozenset = frozenset()) -> list[Any]:
    """`Company(x)` -> listing for `x`; keys in `dead` raise. Returns the construction log."""
    built: list[Any] = []

    def company(key: Any) -> types.SimpleNamespace:
        built.append(key)
        if key in dead:
            raise ValueError(f"no such company {key}")
        return types.SimpleNamespace(get_filings=lambda form: by_key.get(key, []))

    monkeypatch.setattr("edgar.Company", company)
    return built


def _identity(ticker: str, roster_cik: str, entity_by_cik: dict[str, str], *tenure: tuple[str, str]) -> Identity:
    return build_identity(
        lineage=pd.DataFrame([{"cik": cik, "entity_id": entity, "source": "fixture"} for cik, entity in entity_by_cik.items()]),
        tenure=pd.DataFrame(
            [
                {"symbol": symbol, "issuer_cik": cik, "valid_from": pd.Timestamp("2000-01-01"), "valid_to": None, "n_filings": 1, "source": "form345"}
                for symbol, cik in tenure
            ]
        ),
        roster=pd.DataFrame([{"ticker": ticker, "cik": roster_cik}]),
    )


def _xom_register() -> dict[str, Registrant]:
    segments = (
        Segment(cik="0000034088", valid_from=None, valid_to=BOUNDARY, evidence="fixture"),
        Segment(cik="0002115436", valid_from=BOUNDARY, valid_to=None, evidence="fixture"),
    )
    return {"XOM": Registrant(ticker="XOM", kind="reorganisation", segments=segments)}


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == LOGGER and record.levelno == level]


# --------------------------------------------------------------------------- #
# (a) register UNION + identity CIK                                            #
# --------------------------------------------------------------------------- #
def test_register_union_with_identity_cik_dedups_first_writer_and_logs_labels(monkeypatch, caplog):
    built = _patch_company(
        monkeypatch,
        {
            "XOM": [_filing("a", "2026-08-01"), _filing("b", "2026-02-01"), _filing("done", "2026-03-01")],
            34088: [_filing("b", "2026-02-01"), _filing("c", "2010-01-01"), _filing("done", "2026-03-01")],
            2115436: [_filing("a", "2026-08-01"), _filing("d", "2026-09-01"), _filing("old", "1999-01-01")],
            99: [_filing("d", "2026-09-01"), _filing("e", "2005-05-05")],
        },
    )
    identity = _identity(
        "XOM",
        "0002115436",
        {"0000034088": "E-XOM", "0002115436": "E-XOM", "0000000099": "E-XOM"},
        ("XOM", "0000034088"),
        ("XOM", "0002115436"),
    )
    stats: dict[str, int] = {}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(
            "XOM",
            ["8-K", "8-K/A"],
            since=pd.Timestamp("2000-01-01"),
            done_accessions=frozenset({"done"}),
            registrants=_xom_register(),
            identity=identity,
            stats=stats,
        )

    assert [f.accession_number for f in out] == ["e", "c", "b", "a", "d"]
    assert built == ["XOM", 34088, 2115436, 99]
    assert stats == {"skipped_existing": 1}
    assert _messages(caplog, logging.INFO) == [
        "XOM: 2 from ticker, 1 from 0000034088, 1 from 0002115436, 1 from 0000000099 across the 0000034088 -> 0002115436 boundary (8-K,8-K/A)"
    ]
    print("\n=== SANITY: register UNION + identity CIK ===")
    print(f"  order {[f.accession_number for f in out]}; first writer wins; 'done' skipped once; log label 'ticker' then CIKs")


def test_register_union_with_only_ticker_contributions_is_silent(monkeypatch, caplog):
    _patch_company(monkeypatch, {"XOM": [_filing("a", "2026-08-01")], 34088: [_filing("a", "2026-08-01")]})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("XOM", ["8-K"], since=None, done_accessions=frozenset(), registrants=_xom_register())
    assert [f.accession_number for f in out] == ["a"]
    assert _messages(caplog, logging.INFO) == []
    print("\n=== SANITY: register UNION, nothing from a segment -> no boundary log ===")


def test_register_entry_ignores_identity_aliases(monkeypatch, caplog):
    built = _patch_company(monkeypatch, {"XOM": [_filing("a", "2026-08-01")], "OLDX": [_filing("alias", "2001-01-01")]})
    identity = _identity("XOM", "0002115436", {"0000034088": "E-XOM", "0002115436": "E-XOM"}, ("XOM", "0002115436"), ("OLDX", "0002115436"))
    out = resolve_registrant_filings("XOM", ["8-K"], since=None, done_accessions=frozenset(), registrants=_xom_register(), identity=identity)
    assert [f.accession_number for f in out] == ["a"]
    assert "OLDX" not in built
    print("\n=== SANITY: a register entry makes aliases irrelevant ===")


# --------------------------------------------------------------------------- #
# (b) no register entry: aliases + additive CIK                                #
# --------------------------------------------------------------------------- #
def test_no_entry_aliases_and_additive_cik_log_identity_scope(monkeypatch, caplog):
    built = _patch_company(
        monkeypatch,
        {
            "ZBH": [_filing("a", "2025-01-01")],
            "ZMH": [_filing("a", "2025-01-01"), _filing("b", "2004-01-01")],
            58766: [_filing("c", "1999-01-01"), _filing("b", "2004-01-01")],
        },
    )
    identity = _identity("ZBH", "0001136869", {"0001136869": "E-ZBH", "0000058766": "E-ZBH"}, ("ZBH", "0001136869"), ("ZMH", "0001136869"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("ZBH", ["8-K"], since=None, done_accessions=frozenset(), registrants={}, identity=identity)

    assert [f.accession_number for f in out] == ["c", "b", "a"]
    assert built == ["ZBH", "ZMH", 58766]
    assert _messages(caplog, logging.INFO) == ["ZBH: identity scope added filings (1 from ZBH, 1 from ZMH, 1 from 0000058766)"]
    print("\n=== SANITY: no entry, alias + additive CIK ===")
    print("  ticker first, alias second, identity CIK third; each accession counted once")


def test_no_entry_dead_alias_and_dead_cik_warn_and_keep_the_walk(monkeypatch, caplog):
    _patch_company(monkeypatch, {"ZBH": [_filing("a", "2025-01-01")]}, dead=frozenset({"ZMH", 58766}))
    identity = _identity("ZBH", "0001136869", {"0001136869": "E-ZBH", "0000058766": "E-ZBH"}, ("ZBH", "0001136869"), ("ZMH", "0001136869"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("ZBH", ["8-K"], since=None, done_accessions=frozenset(), registrants={}, identity=identity)
    assert [f.accession_number for f in out] == ["a"]
    assert _messages(caplog, logging.WARNING) == [
        "ZBH: historical alias ZMH could not be resolved",
        "ZBH: register CIK 0000058766 could not be resolved",
    ]
    assert _messages(caplog, logging.INFO) == []
    print("\n=== SANITY: dead alias / CIK cost only their own filings ===")


def test_no_entry_split_form_walks_aliases_but_not_identity_ciks(monkeypatch, caplog):
    built = _patch_company(monkeypatch, {"ZBH": [_filing("a", "2025-01-01")], "ZMH": [_filing("b", "2004-01-01")]})
    identity = _identity("ZBH", "0001136869", {"0001136869": "E-ZBH"}, ("ZBH", "0001136869"), ("ZMH", "0001136869"))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("ZBH", ["10-K"], since=None, done_accessions=frozenset(), registrants={}, identity=identity)
    assert [f.accession_number for f in out] == ["b", "a"]
    assert built == ["ZBH", "ZMH"]
    assert _messages(caplog, logging.INFO) == ["ZBH: identity scope added filings (1 from ZBH, 1 from ZMH)"]
    print("\n=== SANITY: SPLIT form, no entry -> alias walked ===")


def test_plain_walk_applies_since_and_done_before_sort(monkeypatch, caplog):
    built = _patch_company(
        monkeypatch, {"AAPL": [_filing("c", "2020-03-01"), _filing("x", "2010-01-01"), _filing("d", "2019-01-01"), _filing("a", "2018-01-01")]}
    )
    stats: dict[str, int] = {}
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings(
            "AAPL", ["8-K"], since=pd.Timestamp("2015-01-01"), done_accessions=frozenset({"d", "x"}), registrants={}, stats=stats
        )
    assert [f.accession_number for f in out] == ["a", "c"]
    assert built == ["AAPL"]
    assert stats == {"skipped_existing": 2}
    assert caplog.records == [] or _messages(caplog, logging.INFO) == []
    print("\n=== SANITY: plain walk -> since + done, sorted, silent ===")


# --------------------------------------------------------------------------- #
# (c) SPLIT                                                                    #
# --------------------------------------------------------------------------- #
def test_split_duplicate_accession_warns_and_keeps_the_first_segment(monkeypatch, caplog):
    overlapping = {
        "T": Registrant(
            ticker="T",
            kind="reorganisation",
            segments=(
                Segment(cik="0000000001", valid_from=None, valid_to=pd.Timestamp("2021-01-01"), evidence="fixture"),
                Segment(cik="0000000002", valid_from=pd.Timestamp("2020-01-01"), valid_to=pd.Timestamp("2022-01-01"), evidence="fixture"),
                Segment(cik="0000000003", valid_from=pd.Timestamp("2022-01-01"), valid_to=None, evidence="fixture"),
            ),
        )
    }
    built = _patch_company(
        monkeypatch,
        {
            1: [_filing("dup", "2020-06-01"), _filing("old", "2019-01-01"), _filing("late1", "2023-01-01")],
            2: [_filing("dup", "2020-06-01"), _filing("mid", "2021-06-01")],
        },
        dead=frozenset({3}),
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = resolve_registrant_filings("T", ["10-K"], since=None, done_accessions=frozenset(), registrants=overlapping)

    assert [f.accession_number for f in out] == ["old", "dup", "mid"]
    assert built == [1, 2, 3]
    assert _messages(caplog, logging.WARNING) == [
        "T: accession dup kept by BOTH segment 0000000001 and 0000000002 -- the dated split makes that impossible, so the register's boundary is wrong",
        "T: register CIK 0000000003 could not be resolved",
    ]
    print("\n=== SANITY: SPLIT duplicate accession ===")
    print("  overlap -> warning, first segment keeps it; dead segment CIK warns with the ticker")


# --------------------------------------------------------------------------- #
# (d) schedule windows                                                          #
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
