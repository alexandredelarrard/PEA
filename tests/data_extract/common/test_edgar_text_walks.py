"""The `sec-employees-text` walk: the shared driver stores each 10-K family filing's chosen document as candidate
windows plus their merged union, from which the unchanged greedy rebuilds today's employee excerpt at any budget.

Offline: a real SQLite `DataStore`, a seeded local EDGAR index and filing stand-ins whose text is three real 10-K
documents (ITW 2003 10-K/A, DUK 1999 10-K/A, MRSH 1997 10-K) under fixture filing dates inside the window.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.sec_io import TransientReadError
from src.data_extract.utils.common.workforce import merge_spans
from src.data_extract.utils.edgar_extract import edgar_driver
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarRun, run_edgar_fetch
from src.data_extract.utils.edgar_extract.fetch_employees_text import EMPLOYEES_TEXT_FETCH
from src.data_extract.utils.fundamentals.fundamentals_employees import _GAP, employee_excerpt
from src.data_store.schema import Tables
from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity
from tests.data_extract.edgar_fixtures import fake_context, patch_index_filings, roster_run, seed_index

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


class TenK:
    """A 10-K stand-in: `text()` returns `body` (no HTML), `fail` transient reads raise first."""

    def __init__(self, accession: str, cik: int, filed: str, body: str, *, report: str | None = "2020-12-31", fail: int = 0) -> None:
        self.accession_number, self.cik, self.filing_date, self.form = accession, cik, filed, "10-K"
        self.company, self.period_of_report, self.primary_document = "Fixture Co", report, None
        self.attachments: list[object] = []
        self._sgml: object | None = object()
        self.body, self.fail = body, fail

    def html(self) -> str:
        if self.fail > 0:
            self.fail -= 1
            raise TransientReadError(f"{self.accession_number}: HTTP 503")
        return ""

    def text(self) -> str:
        return self.body

    def full_text_submission(self) -> str:
        return ""


def rebuild(windows: list[list[int]], pieces: list[list[Any]], limit: int) -> str:
    """Today's greedy over the stored windows, each chosen span sliced from the stored piece that holds it."""
    separator = f"\n\n{_GAP}\n\n"
    chosen: list[tuple[int, int]] = []
    for start, end in windows:
        trial = merge_spans([*chosen, (start, end)])
        if sum(e - s for s, e in trial) + len(separator) * (len(trial) - 1) <= limit:
            chosen = trial
    texts = []
    for start, end in chosen:
        offset, text = next((offset, text) for offset, text in pieces if offset <= start and end <= offset + len(text))
        texts.append(text[start - offset : end - offset])
    return separator.join(texts)


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
    filings = {_accession(i): TenK(_accession(i), 1, f"202{i}-03-01", text) for i, text in enumerate(texts, start=1)}
    seed_index(ctx, [(1, "AAA Inc", "10-K", f.filing_date, a) for a, f in filings.items()])
    patch_index_filings(monkeypatch, filings)

    _walk(ctx, ["AAA"])
    df = _stored(sqlite_store).set_index("accession_number")

    diffs = 0
    for accession, filing in filings.items():
        row = df.loc[accession]
        windows, pieces = json.loads(row["windows"]), json.loads(row["pieces"])
        assert row["source_document"] == "primary" and int(row["n_chars"]) == len(filing.body) and filing._sgml is None
        assert [[s, filing.body[s : s + len(t)]] for s, t in pieces] == pieces, "every piece is the document text at its offset"
        for limit in LIMITS:
            diffs += rebuild(windows, pieces, limit) != employee_excerpt(filing.body, limit)
    sizes = sorted(sum(len(t) for _, t in json.loads(p)) for p in df["pieces"])
    assert diffs == 0 and len(df) == len(REAL) == 3
    print("\n=== SANITY CHECK: stored windows -> old excerpt ===")
    print(f"  {len(df)} real 10-K documents x {len(LIMITS)} limits: {diffs} diff(s) between the greedy over stored rows and employee_excerpt;")
    print(f"  pieces chars {sizes} of documents {sorted(len(t) for t in texts)}; each filing's cached submission dropped after its read.")


def test_empty_text_is_one_marker_stored_once(tmp_path: Path, sqlite_store: Any, waits: list[float], monkeypatch) -> None:
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    blank = TenK(_accession(1), 1, "2021-03-01", "   ")
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
    flaky = TenK(_accession(1), 1, "2021-03-01", "As of December 31, 2020, we had 41,000 employees.", fail=10)
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
    filings = {_accession(i): TenK(_accession(i), 1, f"202{i}-03-01", body) for i in range(1, 5)}
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
        "old-in": TenK(_accession(9, old), old, "2019-03-01", body),
        "old-after": TenK(_accession(1, old), old, "2021-03-01", body),
        "new-before": TenK(_accession(8, new), new, "2018-03-01", body),
        "new-in": TenK(_accession(2, new), new, "2022-03-01", body),
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


def test_parse_drops_the_submission_and_stamps_the_listed_period(monkeypatch: pytest.MonkeyPatch) -> None:
    filing = TenK("acc-1", 1, "2021-03-01", "We had 41,000 employees.", report="2020-12-31")
    stamp = edgar_driver.FilingStamp.of(filing, "0000000001")
    assert EMPLOYEES_TEXT_FETCH.parse is not None

    frames = EMPLOYEES_TEXT_FETCH.parse("AAA", "0000000001", stamp, edgar_driver.EdgarScope())

    row = frames[TABLE].iloc[0]
    assert filing._sgml is None and row["period_of_report"] == "2020-12-31" and row["form"] == "10-K"
    assert json.loads(row["windows"])[0] == [0, 24] and json.loads(row["pieces"]) == [[0, "We had 41,000 employees."]]
    print("\n=== SANITY CHECK: one parsed row ===")
    print(f"  windows {row['windows']}, pieces {row['pieces']}, period_of_report {row['period_of_report']}; submission dropped (L4).")
