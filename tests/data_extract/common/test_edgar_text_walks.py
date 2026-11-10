"""The `sec-employees-text` walk: the shared driver stores each 10-K family filing's chosen document as candidate
windows plus their merged union, from which the unchanged greedy rebuilds today's employee excerpt at any budget.
The `sec-def14a-text` walk stores each proxy's untruncated section payloads, marks a foreign-subject proxy once and
gives a proxy with no payload one visible `_none` row; SEC (listing, documents, headers) is stubbed.

Offline: a real SQLite `DataStore`, a seeded local EDGAR index and filing stand-ins whose text is three real 10-K
documents (ITW 2003 10-K/A, DUK 1999 10-K/A, MRSH 1997 10-K) under fixture filing dates inside the window.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import requests

from src.data_extract.utils.common.sec_io import TransientReadError
from src.data_extract.utils.common.workforce import merge_spans, workforce_number_spans
from src.data_extract.utils.edgar_extract import edgar_driver, edgar_fillings
from src.data_extract.utils.edgar_extract import fetch_def14a_text as def14a_text
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarRun, run_edgar_fetch
from src.data_extract.utils.edgar_extract.edgar_fillings import archive_url
from src.data_extract.utils.edgar_extract.fetch_def14a_text import DEF14A_TEXT_FETCH, NO_PAYLOAD
from src.data_extract.utils.edgar_extract.fetch_employees_text import EMPLOYEES_TEXT_FETCH, candidate_windows
from src.data_extract.utils.edgar_extract.html_text import html_to_text
from src.data_extract.utils.fundamentals.fundamentals_employees import _GAP
from src.data_extract.utils.structure.def14a.carve import prepare_def14a_payloads
from src.data_store.schema import Tables
from src.utils.string import pad_cik
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity
from tests.data_extract.edgar_fixtures import TenK, fake_context, patch_index_filings, roster_run, seed_index, sent_excerpt

#: Real 10-K document texts (the base code's chosen document), keyed `<ticker>_<accession>`.
REAL: dict[str, str] = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "employees_text" / "real_10k_texts.json").read_text("utf-8"))
AS_OF = pd.Timestamp("2026-09-30")
LIMITS = (3_000, 9_000, 20_000, 40_000, 60_000)
TABLE = Tables.sec_employees_text


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The driver's retry-round waits, recorded instead of slept."""
    recorded: list[float] = []
    monkeypatch.setattr(edgar_driver, "_sleep", recorded.append)
    return recorded


def old_excerpt(text: str, limit: int) -> str:
    """The base code's `employee_excerpt`: the same greedy over the windows of the whole document text."""
    separator = f"\n\n{_GAP}\n\n"
    chosen: list[tuple[int, int]] = []
    for window in candidate_windows(text, workforce_number_spans(text)):
        trial = merge_spans([*chosen, window])
        if sum(e - s for s, e in trial) + len(separator) * (len(trial) - 1) <= limit:
            chosen = trial
    return separator.join(text[start:end] for start, end in chosen)


def _accession(i: int, cik: int = 1) -> str:
    return f"{cik:010d}-2{i}-000001"


def _walk(ctx: Any, tickers: list[str], **kwargs: Any) -> edgar_driver.FetchSummary:
    return run_edgar_fetch(ctx, EMPLOYEES_TEXT_FETCH, roster_run(ctx, tickers, as_of=AS_OF, **kwargs), max_workers=1)


def _stored(store: Any, *, markers: bool = False) -> pd.DataFrame:
    df = store.load(TABLE, markers=markers, optional=True)
    return pd.DataFrame(columns=list(TABLE.retrieval_columns)) if df is None else df


def test_stored_windows_rebuild_the_old_excerpt_on_real_10k_text(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    texts = list(REAL.values())
    filings = {_accession(i): TenK(_accession(i), f"202{i}-03-01", text) for i, text in enumerate(texts, start=1)}
    seed_index(ctx, [(1, "AAA Inc", "10-K", f.filing_date, a) for a, f in filings.items()])
    patch_index_filings(monkeypatch, filings)

    _walk(ctx, ["AAA"])
    df = _stored(sqlite_store)

    diffs = 0
    for row in df.itertuples(index=False):
        filing, pieces = filings[row.accession_number], json.loads(row.pieces)
        assert row.source_document == "primary" and int(row.n_chars) == len(filing.body) and filing._sgml is None
        assert [[s, filing.body[s : s + len(t)]] for s, t in pieces] == pieces, "every piece is the document text at its offset"
        for limit in LIMITS:
            diffs += sent_excerpt(row, limit) != old_excerpt(filing.body, limit)
    sizes = sorted(sum(len(t) for _, t in json.loads(p)) for p in df["pieces"])
    assert diffs == 0 and len(df) == len(REAL) == 3
    print("\n=== SANITY CHECK: stored windows -> old excerpt ===")
    print(
        f"  {len(df)} real 10-K documents x {len(LIMITS)} limits: {diffs} diff(s) between the excerpt employee_task sends from stored rows and the old full-text greedy;"
    )
    print(f"  pieces chars {sizes} of documents {sorted(len(t) for t in texts)}; each filing's cached submission dropped after its read.")


def test_empty_text_is_one_marker_stored_once(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    blank = TenK(_accession(1), "2021-03-01", "   ")
    seed_index(ctx, [(1, "AAA Inc", "10-K", blank.filing_date, blank.accession_number)])
    patch_index_filings(monkeypatch, {blank.accession_number: blank})

    first = _walk(ctx, ["AAA"])
    second = _walk(ctx, ["AAA"])

    markers = _stored(sqlite_store, markers=True)
    assert first.work.size == 1 and first.outcomes["AAA"].markers == 1 and second.work.size == 0
    assert (
        _stored(sqlite_store).empty
        and markers["source_document"].tolist() == ["_empty"]
        and markers.reindex(columns=["pieces"])["pieces"].isna().all()
    )
    print("\n=== SANITY CHECK: empty text -> marker ===")
    print(f"  a filing with no usable text is stored once as source_document=_empty (hidden by load); the next run lists {second.work.size}.")


def test_transient_read_stores_nothing_and_lists_again(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    flaky = TenK(_accession(1), "2021-03-01", "As of December 31, 2020, we had 41,000 employees.", fail=10)
    seed_index(ctx, [(1, "AAA Inc", "10-K", flaky.filing_date, flaky.accession_number)])
    patch_index_filings(monkeypatch, {flaky.accession_number: flaky})

    first = _walk(ctx, ["AAA"])
    stored_after_failure = len(_stored(sqlite_store, markers=True))
    flaky.fail = 0
    second = _walk(ctx, ["AAA"])

    assert stored_after_failure == 0 and first.missing == {"AAA": [flaky.accession_number]}
    assert second.work.size == 1 and _stored(sqlite_store)["accession_number"].tolist() == [flaky.accession_number]
    print("\n=== SANITY CHECK: transient read ===")
    print(f"  an SEC 503 saves no row and no marker ({stored_after_failure} stored); the next run lists it again and stores it.")


def test_cold_fill_then_nothing_new(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"], max_documents_per_run=2)
    body = "As of December 31, 2020, we had 41,000 employees."
    filings = {_accession(i): TenK(_accession(i), f"202{i}-03-01", body) for i in range(1, 5)}
    seed_index(ctx, [(1, "AAA Inc", "10-K", f.filing_date, a) for a, f in filings.items()])
    patch_index_filings(monkeypatch, filings)

    capped = _walk(ctx, ["AAA"])
    after_capped = set(_stored(sqlite_store)["accession_number"])
    uncapped = _walk(ctx, ["AAA"], no_cap=True)
    again = _walk(ctx, ["AAA"])

    assert capped.work.uncapped == 4 and capped.work.size == 2 and after_capped == {_accession(3), _accession(4)}
    assert uncapped.work.size == 2 and set(_stored(sqlite_store)["accession_number"]) == set(filings) and again.work.size == 0
    print("\n=== SANITY CHECK: cold fill (AC-004) ===")
    print(f"  cap 2: the 2 newest of 4 stored; --no-cap filled the other {uncapped.work.size}; the next run lists {again.work.size}.")


def test_multi_window_ticker_keeps_only_its_own_windows(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    body = "As of December 31, 2020, we had 41,000 employees."
    old, new = 9999991, 1
    filings = {
        "old-in": TenK(_accession(9, old), "2019-03-01", body, cik=old),
        "old-after": TenK(_accession(1, old), "2021-03-01", body, cik=old),
        "new-before": TenK(_accession(8, new), "2018-03-01", body, cik=new),
        "new-in": TenK(_accession(2, new), "2022-03-01", body, cik=new),
    }
    seed_index(ctx, [(f.cik, "Fixture Co", "10-K", f.filing_date, f.accession_number) for f in filings.values()])
    patch_index_filings(monkeypatch, {f.accession_number: f for f in filings.values()})
    identity = dated_identity(
        [("AAA", "0009999991", "cik_window", SENTINEL, "2020-01-01"), ("AAA", "0000000001", "cik_window", "2020-01-01", None)],
        {"AAA": "0000000001"},
    )
    monkeypatch.setattr(edgar_driver, "load_identity", lambda context: identity)

    run_edgar_fetch(ctx, EMPLOYEES_TEXT_FETCH, EdgarRun.load(ctx, ["AAA"], 15, as_of=AS_OF), max_workers=1)

    df = _stored(sqlite_store, markers=True).sort_values("filing_date")
    assert df["accession_number"].tolist() == [filings["old-in"].accession_number, filings["new-in"].accession_number]
    assert df["cik"].tolist() == ["0009999991", "0000000001"]
    print("\n=== SANITY CHECK: multi-window ticker (AC-006) ===")
    print("  AAA = CIK 9999991 until 2020, CIK 1 after: only the two in-window 10-Ks stored, each stamped with its own filer CIK;")
    print("  the old CIK's later 10-K and the new CIK's earlier one are not read.")


def test_a_filing_shared_by_two_tickers_is_stored_for_each(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"], ciks=["1", "1"])
    shared = TenK(_accession(1), "2021-03-01", "As of December 31, 2020, we had 41,000 employees.")
    seed_index(ctx, [(1, "AAA Inc", "10-K", shared.filing_date, shared.accession_number)])
    patch_index_filings(monkeypatch, {shared.accession_number: shared})

    _walk(ctx, ["AAA"])
    second = _walk(ctx, ["AAA", "BBB"])

    df = _stored(sqlite_store)
    assert second.work.size == 1 and sorted(df["ticker"]) == ["AAA", "BBB"] and set(df["accession_number"]) == {shared.accession_number}
    print("\n=== SANITY CHECK: one 10-K, two tickers (per-key done set) ===")
    print(f"  AAA stored the shared 10-K first; adding BBB lists it {second.work.size} more time and stores it for BBB too.")


def test_parse_drops_the_submission_and_stamps_the_listed_period(monkeypatch: pytest.MonkeyPatch) -> None:
    filing = TenK("acc-1", "2021-03-01", "We had 41,000 employees.", report="2020-12-31")
    stamp = edgar_driver.FilingStamp.of(filing, "0000000001")
    assert EMPLOYEES_TEXT_FETCH.parse is not None

    frames = EMPLOYEES_TEXT_FETCH.parse("AAA", "0000000001", stamp, edgar_driver.EdgarScope())

    row = frames[TABLE].iloc[0]
    assert filing._sgml is None and row["period_of_report"] == "2020-12-31" and row["form"] == "10-K"
    assert json.loads(row["windows"])[0] == [0, 24] and json.loads(row["pieces"]) == [[0, "We had 41,000 employees."]]
    print("\n=== SANITY CHECK: one parsed row ===")
    print(f"  windows {row['windows']}, pieces {row['pieces']}, period_of_report {row['period_of_report']}; submission dropped (L4).")


# --- `sec-def14a-text` --------------------------------------------------------------------------------------------

#: A proxy whose carve fills the governance section only (the board-election prose).
PROXY = """<html><body><h1>Election of Directors</h1>
<p>Jane Example, age 54, Director since 2018. Jane Example is Chief Executive Officer.</p>
</body></html>"""
BLANK_PROXY = "<html><body><p>Notice of annual meeting.</p></body></html>"
DEF14A = Tables.sec_def14a_text
FOREIGN = "0001437107"


class ProxyWorld:
    """SEC stand-ins for the DEF 14A walk: the submissions listing, the document reads and the SGML subject headers.

    `listing[cik]` holds `(accession, primaryDocument)`; a `listing_error` raises from the listing;
    `html[url]` (else `PROXY`) is the document, a `BaseException` value raises; `subjects[accession]` is the header's
    subject CIKs, a `BaseException` raises."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, listing: dict[str, list[tuple[str, str]]]) -> None:
        self.listing = listing
        self.listing_error: BaseException | None = None
        self.html: dict[str, object] = {}
        self.subjects: dict[str, object] = {}
        self.listed: list[str] = []
        self.urls: list[str] = []
        self.headers: list[str] = []
        monkeypatch.setattr(def14a_text, "list_filings", self.list_filings)
        monkeypatch.setattr(def14a_text, "sec_get", self.sec_get)
        monkeypatch.setattr(def14a_text, "header_subject_ciks", self.header)

    def list_filings(self, context: Any, cik: str, forms: Any, since: pd.Timestamp) -> pd.DataFrame:
        self.listed.append(pad_cik(cik))
        if self.listing_error is not None:
            raise self.listing_error
        rows = [{"accession_number": a, "primary_document": p} for a, p in self.listing.get(pad_cik(cik), [])]
        return pd.DataFrame(rows, columns=["accession_number", "primary_document"])

    def sec_get(self, context: Any, url: str) -> SimpleNamespace:
        self.urls.append(url)
        body = self.html.get(url, PROXY)
        if isinstance(body, BaseException):
            raise body
        return SimpleNamespace(text=body)

    def header(self, filing: Any) -> frozenset[str]:
        self.headers.append(str(filing.accession_no))
        subjects = self.subjects.get(str(filing.accession_no), frozenset())
        if isinstance(subjects, BaseException):
            raise subjects
        return cast(frozenset[str], subjects)


def _proxy_walk(ctx: Any, run: EdgarRun) -> edgar_driver.FetchSummary:
    return run_edgar_fetch(ctx, DEF14A_TEXT_FETCH, run, max_workers=1)


def _proxies(store: Any, *, markers: bool = False) -> pd.DataFrame:
    df = store.load(DEF14A, markers=markers, optional=True)
    return pd.DataFrame(columns=list(DEF14A.retrieval_columns)) if df is None else df


def _two_window_run(ctx: Any, monkeypatch: pytest.MonkeyPatch) -> EdgarRun:
    """AAA = CIK 9999991 until 2020, CIK 1 after: a multi-window scope, so its proxies' subject headers are checked."""
    identity = dated_identity(
        [("AAA", "0009999991", "cik_window", SENTINEL, "2020-01-01"), ("AAA", "0000000001", "cik_window", "2020-01-01", None)],
        {"AAA": "0000000001"},
    )
    monkeypatch.setattr(edgar_driver, "load_identity", lambda context: identity)
    return EdgarRun.load(ctx, ["AAA"], 15, as_of=AS_OF)


def test_rejected_foreign_subject_is_one_marker_never_read_again(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    own, foreign = _accession(1), _accession(2)
    seed_index(ctx, [(1, "AAA Inc", "DEF 14A", "2022-03-01", own), (1, "AAA Inc", "DEFC14A", "2023-03-01", foreign)])
    world = ProxyWorld(monkeypatch, {"0000000001": [(own, "proxy.htm"), (foreign, "contest.htm")]})
    world.subjects = {own: frozenset({"0000000001"}), foreign: frozenset({FOREIGN})}

    first = _proxy_walk(ctx, _two_window_run(ctx, monkeypatch))
    second = _proxy_walk(ctx, _two_window_run(ctx, monkeypatch))

    rows, markers = _proxies(sqlite_store), _proxies(sqlite_store, markers=True)
    marker = markers[markers["section"].eq("_empty")]
    assert first.work.size == 2 and first.outcomes["AAA"].markers == 1 and second.work.size == 0
    assert marker["accession_number"].tolist() == [foreign] and set(rows["accession_number"]) == {own}
    assert sorted(world.headers) == sorted([own, foreign]) and not any("contest.htm" in url for url in world.urls)
    assert world.listed == ["0000000001"]
    print("\n=== SANITY CHECK: DEF 14A subject guard (AC-005/006) ===")
    print(f"  multi-window AAA: the proxy whose subject is {FOREIGN} is stored once as section=_empty (hidden by load), its document")
    print(f"  never read; the own proxy stored {len(rows)} section row(s); the next run lists {second.work.size} and reads no header again.")


def test_transient_header_error_stores_nothing_and_lists_again(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accession = _accession(1)
    seed_index(ctx, [(1, "AAA Inc", "DEF 14A", "2022-03-01", accession)])
    world = ProxyWorld(monkeypatch, {"0000000001": [(accession, "proxy.htm")]})
    world.subjects = {accession: TransientReadError("header: HTTP 503")}

    first = _proxy_walk(ctx, _two_window_run(ctx, monkeypatch))
    stored_after_failure, read_after_failure = len(_proxies(sqlite_store, markers=True)), len(world.urls)
    world.subjects = {accession: frozenset({"0000000001"})}
    second = _proxy_walk(ctx, _two_window_run(ctx, monkeypatch))

    assert stored_after_failure == 0 and read_after_failure == 0 and first.missing == {"AAA": [accession]}
    assert second.work.size == 1 and _proxies(sqlite_store)["accession_number"].unique().tolist() == [accession]
    print("\n=== SANITY CHECK: transient subject header (M3) ===")
    print(f"  an SEC 503 on the header saves no row and no marker ({stored_after_failure} stored) and reads no document;")
    print("  the next run lists the proxy again, reads the header and stores it.")


def test_unreadable_header_accepts_the_proxy(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accession = _accession(1)
    seed_index(ctx, [(1, "AAA Inc", "DEF 14A", "2022-03-01", accession)])
    world = ProxyWorld(monkeypatch, {"0000000001": [(accession, "proxy.htm")]})
    world.subjects = {accession: ValueError("no SGML header")}

    summary = _proxy_walk(ctx, _two_window_run(ctx, monkeypatch))

    rows = _proxies(sqlite_store, markers=True)
    assert summary.outcomes["AAA"].markers == 0 and rows["section"].tolist() == ["governance"] and world.headers == [accession]
    print("\n=== SANITY CHECK: non-transient header error (D-10) ===")
    print("  a header that cannot be parsed (not an SEC error page) accepts the proxy, as today: its governance payload is stored.")


def test_walked_proxy_with_no_payload_is_one_visible_none_row(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accession = _accession(1)
    seed_index(ctx, [(1, "AAA Inc", "DEF 14A", "2022-03-01", accession)])
    world = ProxyWorld(monkeypatch, {"0000000001": [(accession, "notice.htm")]})
    world.html = {archive_url("1", accession, "notice.htm"): BLANK_PROXY}

    first = _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))
    second = _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))

    rows = _proxies(sqlite_store)
    assert first.outcomes["AAA"].markers == 0 and second.work.size == 0
    assert rows[["section", "payload", "n_chars"]].to_dict("records") == [{"section": NO_PAYLOAD, "payload": "", "n_chars": 0}]
    print("\n=== SANITY CHECK: walked proxy, no payload (H2) ===")
    print(f"  every section carved empty: one VISIBLE row section={NO_PAYLOAD} (empty payload, n_chars 0), not a marker;")
    print(f"  the next run lists {second.work.size}.")


def test_def14a_cold_fill_then_nothing_new(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"], max_documents_per_run=2)
    accessions = [_accession(i) for i in range(1, 5)]
    seed_index(ctx, [(1, "AAA Inc", "DEF 14A", f"202{i}-03-01", a) for i, a in enumerate(accessions, start=1)])
    world = ProxyWorld(monkeypatch, {"0000000001": [(a, f"p{i}.htm") for i, a in enumerate(accessions, start=1)]})

    capped = _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))
    after_capped = set(_proxies(sqlite_store)["accession_number"])
    uncapped = _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF, no_cap=True))
    again = _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))

    assert capped.work.uncapped == 4 and capped.work.size == 2 and after_capped == set(accessions[2:])
    assert uncapped.work.size == 2 and set(_proxies(sqlite_store)["accession_number"]) == set(accessions) and again.work.size == 0
    assert world.listed == ["0000000001"] * 2, "one submissions read per run with pending proxies, none when nothing is pending"
    print("\n=== SANITY CHECK: DEF 14A cold fill (AC-004) ===")
    print(f"  cap 2: the 2 newest of 4 proxies stored; --no-cap filled the other {uncapped.work.size}; the next run lists {again.work.size};")
    print(f"  submissions reads per run: 1, 1, 0 ({len(world.listed)} in all).")


def test_failed_listing_stores_nothing_and_lists_again(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accession = _accession(1)
    seed_index(ctx, [(1, "AAA Inc", "DEF 14A", "2022-03-01", accession)])
    world = ProxyWorld(monkeypatch, {"0000000001": [(accession, "proxy.htm")]})
    world.listing_error = requests.HTTPError("403 Client Error: Forbidden")

    first = _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))
    stored_after_failure, read_after_failure = len(_proxies(sqlite_store, markers=True)), list(world.urls)
    world.listing_error = KeyError("filings")
    with pytest.raises(KeyError):
        _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))
    world.listing_error = None
    second = _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))

    assert stored_after_failure == 0 and read_after_failure == [] and first.missing == {"AAA": [accession]}
    assert second.work.size == 1 and _proxies(sqlite_store)["primary_document"].tolist() == ["proxy.htm"]
    print("\n=== SANITY CHECK: failed submissions listing (R-C02) ===")
    print(f"  a 403 listing stores nothing ({stored_after_failure} rows) and reads no document; a KeyError propagates;")
    print("  the next run lists the proxy again and stores it from its primary document, not the .txt.")


def test_listing_date_before_the_index_date_reads_the_primary_document(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accession = _accession(1)
    seed_index(ctx, [(1, "AAA Inc", "DEF 14A", "2022-03-01", accession)])
    real_listing = def14a_text.list_filings
    world = ProxyWorld(monkeypatch, {})
    monkeypatch.setattr(def14a_text, "list_filings", real_listing)
    recent = {
        "accessionNumber": [accession],
        "form": ["DEF 14A"],
        "filingDate": ["2022-02-26"],
        "primaryDocument": ["proxy.htm"],
        "reportDate": ["2022-04-20"],
    }
    monkeypatch.setattr(
        edgar_fillings, "sec_get", lambda context, url, **k: SimpleNamespace(json=lambda: {"filings": {"recent": recent, "files": []}})
    )

    _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))

    assert world.urls == [archive_url("1", accession, "proxy.htm")]
    assert _proxies(sqlite_store)["primary_document"].tolist() == ["proxy.htm"]
    print("\n=== SANITY CHECK: SEC filingDate 3 days before the index date (R2-04) ===")
    print("  the production listing keeps the proxy, so its primary document is read and stored, not the <acc>.txt submission.")


def test_a_proxy_shared_by_two_tickers_is_stored_for_each(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"], ciks=["1", "1"])
    accession = _accession(1)
    seed_index(ctx, [(1, "AAA Inc", "DEF 14A", "2022-03-01", accession)])
    ProxyWorld(monkeypatch, {"0000000001": [(accession, "proxy.htm")]})

    _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))
    second = _proxy_walk(ctx, roster_run(ctx, ["AAA", "BBB"], as_of=AS_OF))

    df = _proxies(sqlite_store)
    assert second.work.size == 1 and sorted(df["ticker"]) == ["AAA", "BBB"] and set(df["accession_number"]) == {accession}
    print("\n=== SANITY CHECK: one proxy, two tickers (per-key done set) ===")
    print(f"  AAA stored the shared proxy first; adding BBB lists it {second.work.size} more time and stores it for BBB too.")


def test_single_window_ticker_reads_no_header(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    accession = _accession(1)
    seed_index(ctx, [(1, "AAA Inc", "DEF 14A", "2022-03-01", accession)])
    world = ProxyWorld(monkeypatch, {"0000000001": [(accession, "proxy.htm")]})
    world.subjects = {accession: frozenset({FOREIGN})}

    _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))

    assert world.headers == [] and _proxies(sqlite_store)["section"].tolist() == ["governance"]
    print("\n=== SANITY CHECK: single-window ticker ===")
    print("  a one-CIK scope stores its proxy with no SGML header read (no guard), even when the header would name another subject.")


def test_parse_reads_the_listed_document(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    old, gone, contest = _accession(1), _accession(2), _accession(3)
    seed_index(
        ctx,
        [
            (1, "AAA Inc", "DEF 14A", "2021-03-01", old),
            (1, "AAA Inc", "DEF 14A", "2022-03-01", gone),
            (1, "AAA Inc", "DEFC14A", "2023-03-01", contest),
        ],
    )
    world = ProxyWorld(monkeypatch, {"0000000001": [(old, ""), (gone, "gone.htm"), (contest, "c.htm")]})
    world.html = {archive_url("1", gone, "gone.htm"): ValueError("HTTP 404")}

    _proxy_walk(ctx, roster_run(ctx, ["AAA"], as_of=AS_OF))

    rows = _proxies(sqlite_store).set_index("accession_number")
    expected = prepare_def14a_payloads(PROXY, html_to_text(PROXY))["governance"]
    assert sorted(world.urls) == sorted(
        [
            archive_url("1", old, f"{old}.txt"),
            archive_url("1", gone, "gone.htm"),
            archive_url("1", gone, f"{gone}.txt"),
            archive_url("1", contest, "c.htm"),
        ]
    )
    assert rows.loc[old, "primary_document"] == "" and rows.loc[contest, "primary_document"] == "c.htm"
    assert (rows["payload"] == expected).all() and (rows["n_chars"] == len(expected)).all() and "period" not in rows.columns
    assert world.listed == ["0000000001"]
    print("\n=== SANITY CHECK: one listing, today's document rule ===")
    print("  one submissions read for the CIK; no primaryDocument -> <acc>.txt; a 404 primary -> the .txt fallback; no period")
    print(f"  column; the stored payload equals the carve's untruncated payload ({len(expected)} chars).")
