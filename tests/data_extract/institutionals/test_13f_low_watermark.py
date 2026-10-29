"""
test_13f_low_watermark.py (tests/data_extract/institutionals/test_13f_low_watermark.py)
---------------------------------------------------------------------------------------
The nightly 13F walk (M4) on a real SQLite `DataStore` with fake filings (no network): the resume
window is `max(sec13f_hr.filing_date)` minus the overlap to the run date. Successful books save
in batches while failed reads alone get retry rounds after the first pass. M4 case 2 (a new ticker) is the data-set backfill, in
`test_13f_backfill_zip.py`.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pytest

from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_store.schema import Resume, Tables
from src.data_store.store import DataStore
from src.validate.checks.institutionals import check_institutionals
from tests.data_extract.edgar_fixtures import fake_context
from tests.data_extract.institutionals.test_13f_one_walk import CMAP, ROSTER, _FakeFiling, _line, _seed_roster

_AS_OF = pd.Timestamp("2026-09-30")
_UNIVERSE = ["AAPL", "MSFT"]
_AAPL = [_line("037833100", "APPLE INC", 1_000.0, 10)]


def _patch(monkeypatch: pytest.MonkeyPatch, filings: list[_FakeFiling]) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], list[float]]:
    """`_listing` over the fakes, behind the edgartools `Filings` surface `fetch_13f` uses: each walk
    records its filing-date window and lists only the fakes filed inside it. Returns (windows, waits)."""
    by_accession = {f.accession_number: f for f in filings}
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    waits: list[float] = []

    class _Filings:
        def __init__(self, data: pa.Table) -> None:
            self.data = data

        def __len__(self) -> int:
            return self.data.num_rows

        def __iter__(self) -> Any:
            return (by_accession[a] for a in self.data.column("accession_number").to_pylist())

    def _listing(forms: list[str], since: pd.Timestamp, until: pd.Timestamp) -> _Filings:
        windows.append((since, until))
        inside = [f for f in filings if since <= pd.Timestamp(f.filing_date) <= until]
        return _Filings(
            pa.table(
                {
                    "form": [f.form for f in inside],
                    "accession_number": [f.accession_number for f in inside],
                    "filing_date": [date.fromisoformat(f.filing_date) for f in inside],
                }
            )
        )

    monkeypatch.setattr(f13, "Filings", _Filings)
    monkeypatch.setattr(f13, "_listing", _listing)
    monkeypatch.setattr(f13, "build_cusip_ticker_map", lambda context, cusips, **kwargs: CMAP)
    monkeypatch.setattr(f13, "_sleep", waits.append)
    return windows, waits


def _ctx(tmp_path: Any, store: Any) -> Any:
    ctx = fake_context(tmp_path, store, _UNIVERSE, redundant_ticks=[])
    _seed_roster(store, [ROSTER])
    return ctx


def _filing(n: int, filed: str, **kwargs: Any) -> _FakeFiling:
    return _FakeFiling(f"{n:010d}", filed, "2026-06-30", list(_AAPL), accession=f"{n:010d}-26-{n:06d}", **kwargs)


def _saved_ciks(store: Any) -> list[str]:
    df = store.load(Tables.sec13f_hr, columns=["cik"], optional=True)
    return [] if df is None else sorted(df["cik"].unique())


def _run(ctx: Any, as_of: pd.Timestamp = _AS_OF, **kwargs: Any) -> int:
    return f13.fetch_13f(
        ctx,
        tickers=kwargs.pop("tickers", _UNIVERSE),
        years_history=kwargs.pop("years_history", 15),
        save_every=kwargs.pop("save_every", 1),
        as_of=as_of,
        **kwargs,
    )


def _stored_watermark(store: Any) -> pd.Timestamp:
    return pd.Timestamp(store.max_date(Tables.sec13f_hr, "filing_date")).normalize()


def _seed_hr(store: Any, cik: str, filed: str) -> None:
    store.save(
        Tables.sec13f_hr,
        (
            pd.DataFrame(
                [
                    {
                        "cik": cik,
                        "period": pd.Timestamp("2026-06-30"),
                        "ticker": "AAPL",
                        "cusip": "037833100",
                        "filing_date": pd.Timestamp(filed),
                        "value_usd": 1.0,
                    }
                ]
            )
        ).assign(accession_number="_legacy"),
    )


# --------------------------------------------------------------------------- #
# Batch persistence and retry rounds                                         #
# --------------------------------------------------------------------------- #
def test_first_pass_keeps_saving_every_100_after_failures(tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    transient = _filing(1, "2026-09-28", fail_reads=1)
    malformed = _filing(2, "2026-09-28", unparseable=True)
    filings = [transient, malformed, *[_filing(n, "2026-09-29") for n in range(3, 204)]]
    ctx = _ctx(tmp_path, filing_store)
    _patch(monkeypatch, filings)
    saver = f13._Saver(ctx, set(_UNIVERSE), set(), save_every=100)
    read_filing = f13.read_filing_vintage
    saved_during_walk: list[int] = []

    def observed_read(stamp: f13.FilingStamp) -> Any:
        if stamp.cik in {"0000000103", "0000000203"}:
            saved_during_walk.append(len(_saved_ciks(filing_store)))
            assert len(saver.pending) == 0
        return read_filing(stamp)

    monkeypatch.setattr(f13, "read_filing_vintage", observed_read)
    failures = f13._first_pass(filings, saver)

    assert saved_during_walk == [100, 200]
    assert [read.stamp.accession_number for read in failures] == [transient.accession_number, malformed.accession_number]
    assert all(isinstance(read.result, f13.ReadFailure) for read in failures)
    assert len(_saved_ciks(filing_store)) == 201 and saver.pending == []
    print(
        "SANITY: failures before the first batch do not block saves: 100/200 books committed during parsing, final 1 flushed; only 2 failures queued."
    )


def test_retry_saves_recovered_filings_while_another_failure_persists(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    persistent = _filing(1, "2026-09-27", fail_reads=99)
    recovered = _filing(2, "2026-09-28", fail_reads=1)
    later = _filing(3, "2026-09-29")
    ctx = _ctx(tmp_path, filing_store)
    _, waits = _patch(monkeypatch, [persistent, recovered, later])
    saved_before_rounds: list[list[str]] = []

    def observe_wait(wait: float) -> None:
        waits.append(wait)
        saved_before_rounds.append(_saved_ciks(filing_store))

    monkeypatch.setattr(f13, "_sleep", observe_wait)
    assert _run(ctx) == 1

    assert saved_before_rounds == [["0000000003"], ["0000000002", "0000000003"], ["0000000002", "0000000003"]]
    assert waits == [60.0, 120.0, 240.0]
    assert [len(filing.reads) for filing in (persistent, recovered, later)] == [4, 2, 1]
    assert _saved_ciks(filing_store) == ["0000000002", "0000000003"]
    validation = check_institutionals(
        ctx,
        ciks=[filing.cik for filing in (persistent, recovered, later)],
        start_period="2026-06-30",
        end_period="2026-06-30",
        filed_through="2026-10-01",
        trading_index=pd.bdate_range("2026-06-01", "2026-10-01"),
        table=Tables.sec13f_hr,
    )
    assert validation.status == "pass", validation.summary()
    assert validation.metrics["accessions"] == 2 and validation.metrics["holding_rows"] == 2
    print(
        "SANITY: later book is saved before retries; a recovered book is saved after round 1 while another accession keeps failing; successes are not reread."
    )
    print("SANITY: read-only source validator passes both saved accessions; the exhausted failure has no completed book.")


def test_filings_after_a_recent_failure_save_and_the_failure_retries_next_run(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    first, failing, after = _filing(1, "2026-09-27"), _filing(2, "2026-09-28", fail_reads=4), _filing(3, "2026-09-29")
    ctx = _ctx(tmp_path, filing_store)
    _seed_hr(filing_store, "0000000009", "2026-09-26")
    windows, waits = _patch(monkeypatch, [first, failing, after])

    with caplog.at_level(logging.WARNING):
        assert _run(ctx) == 1
    night1 = _saved_ciks(filing_store)
    watermark1 = _stored_watermark(filing_store)
    _run(ctx, as_of=_AS_OF + pd.Timedelta(days=1))  # the failure is over (4 reads used up)

    assert night1 == ["0000000001", "0000000003", "0000000009"]
    assert watermark1 == pd.Timestamp(after.filing_date)
    assert len(failing.reads) == 5 and waits == [60.0, 120.0, 240.0]  # pass + 3 rounds, then the next night
    assert windows[1][0] <= pd.Timestamp(failing.filing_date)  # the next window includes the failed filing
    assert _saved_ciks(filing_store) == ["0000000001", "0000000002", "0000000003", "0000000009"]
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and failing.accession_number in r.getMessage()]
    assert len(errors) == 1 and "skipped" in errors[0], errors
    assert not any("held back" in r.getMessage() for r in caplog.records)
    print("\n=== SANITY: 13F recent failure ===")
    print(f"  night 1: the 2026-09-28 filing failed after 3 rounds {waits}; saved {night1}, watermark {watermark1:%Y-%m-%d};")
    print(f"  night 2 window starts {windows[1][0]:%Y-%m-%d} <= 2026-09-28 and retries the failed accession only. Validated.")


def test_a_failure_recovered_in_a_round_saves_every_filing(tmp_path, filing_store, monkeypatch):
    filings = [_filing(1, "2026-09-27"), _filing(2, "2026-09-28", fail_reads=1), _filing(3, "2026-09-29")]
    ctx = _ctx(tmp_path, filing_store)
    _, waits = _patch(monkeypatch, filings)

    _run(ctx)

    assert _saved_ciks(filing_store) == ["0000000001", "0000000002", "0000000003"]
    assert waits == [60.0]
    print("\n=== SANITY: 13F retry round ===")
    print(f"  one transient failure recovered in round 1 (waits {waits}); all 3 filings saved in order. Validated.")


def test_a_failure_older_than_the_overlap_is_skipped_with_an_error(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    old, later = _filing(1, "2026-09-15", fail_reads=99), _filing(2, "2026-09-29")
    ctx = _ctx(tmp_path, filing_store)
    _seed_hr(filing_store, "0000000009", "2026-09-18")  # window from 2026-09-11; the overlap edge is 2026-09-23
    _patch(monkeypatch, [old, later])

    with caplog.at_level(logging.WARNING):
        assert _run(ctx) == 1

    assert _saved_ciks(filing_store) == ["0000000002", "0000000009"]
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and old.accession_number in errors[0] and "skipped" in errors[0], errors
    print("\n=== SANITY: 13F failure older than the overlap ===")
    print(f"  2026-09-15 still fails at as_of 2026-09-30 -> ERROR {errors[0][:80]!r}; the 2026-09-29 filing saved. Validated.")


def test_a_deterministic_failure_retries_without_blocking_other_filings(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    broken, later = _filing(1, "2026-09-28", unparseable=True), _filing(2, "2026-09-29")
    ctx = _ctx(tmp_path, filing_store)
    _, waits = _patch(monkeypatch, [broken, later])

    with caplog.at_level(logging.WARNING):
        assert _run(ctx) == 1

    assert _saved_ciks(filing_store) == ["0000000002"] and waits == [60.0, 120.0, 240.0] and len(broken.reads) == 4
    assert any(r.levelno == logging.ERROR and broken.accession_number in r.getMessage() for r in caplog.records)
    print("\n=== SANITY: 13F unparseable filing ===")
    print("  a parse failure gets the configured retries and is reported unread; the next filing saves normally. Validated.")


# --------------------------------------------------------------------------- #
# AC-002, M4 cases                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("years_history", "since"), [(5, "2021-09-30"), (31, None)])
def test_m4_1_absent_table_reads_from_the_history_floor(tmp_path, filing_store, monkeypatch, years_history, since):
    ctx = _ctx(tmp_path, filing_store)
    windows, _ = _patch(monkeypatch, [_filing(1, "2026-09-29")])

    _run(ctx, years_history=years_history)

    source_start = cast(Resume, Tables.sec13f_hr.resume).source_start
    since = source_start if since is None else since
    assert windows == [(pd.Timestamp(since), _AS_OF)]
    print(f"\n=== SANITY: M4-1 absent table (years_history={years_history}) ===")
    print(f"  window {since}:{_AS_OF:%Y-%m-%d}: as_of - years_history, never before the table's {source_start} source start. Validated.")


def test_m4_3_forward_window_is_the_frontier_minus_the_overlap(tmp_path, filing_store, monkeypatch):
    ctx = _ctx(tmp_path, filing_store)
    _seed_hr(filing_store, "0000000009", "2026-09-20")
    windows, _ = _patch(monkeypatch, [_filing(1, "2026-09-12"), _filing(2, "2026-09-25")])

    _run(ctx)

    assert windows == [(pd.Timestamp("2026-09-13"), _AS_OF)]
    assert _saved_ciks(filing_store) == ["0000000002", "0000000009"]
    print("\n=== SANITY: M4-3 forward window ===")
    print("  stored max filing_date 2026-09-20 -> window 2026-09-13:2026-09-30 (7-day overlap); the 09-12 filing is outside. Validated.")


def test_m4_4_a_hole_older_than_the_overlap_is_reached_only_by_a_filing_window(tmp_path, filing_store, monkeypatch):
    ctx = _ctx(tmp_path, filing_store)
    _seed_hr(filing_store, "0000000009", "2026-09-20")
    lost = _filing(1, "2026-08-01")
    _patch(monkeypatch, [lost])

    _run(ctx)
    nightly = _saved_ciks(filing_store)
    _run(ctx, filing_window=("2026-07-30", "2026-08-02"))

    assert nightly == ["0000000009"] and lost.reads == [lost.accession_number]
    assert _saved_ciks(filing_store) == ["0000000001", "0000000009"]
    assert _stored_watermark(filing_store) == pd.Timestamp("2026-09-20")  # the backfill left the window alone
    print("\n=== SANITY: M4-4 hole behind the overlap ===")
    print("  the nightly window never lists the 2026-08-01 filing; --filing-window reads it and the frontier stays 2026-09-20. Validated.")


def test_m4_5_an_empty_info_table_is_no_failure_and_holds_nothing(tmp_path, filing_store, monkeypatch):
    """Zero-row unit: 13F keeps no marker (the walk is market-wide by filing date); an empty info table
    simply saves nothing and is not a failure."""
    empty = _FakeFiling("0000000001", "2026-09-28", "2026-06-30", [], accession="0000000001-26-000001")
    ctx = _ctx(tmp_path, filing_store)
    _, waits = _patch(monkeypatch, [empty, _filing(2, "2026-09-29")])

    _run(ctx)

    assert _saved_ciks(filing_store) == ["0000000002"] and waits == []
    print("\n=== SANITY: M4-5 zero-row filing ===")
    print("  an empty info table saves nothing, is not retried and holds nothing back (no marker for 13F). Validated.")


def test_m4_7_deleted_rows_inside_the_window_are_read_again(tmp_path, filing_store, monkeypatch):
    ctx = _ctx(tmp_path, filing_store)
    filings = [_filing(1, "2026-09-20"), _filing(2, "2026-09-29")]
    _patch(monkeypatch, filings)
    _run(ctx)
    filing_store.delete(Tables.sec13f_hr, where={"cik": "0000000002"})
    for filing in filings:
        filing.reads.clear()

    _run(ctx, as_of=_AS_OF + pd.Timedelta(days=1))

    assert _saved_ciks(filing_store) == ["0000000001", "0000000002"]
    assert filings[1].reads == [filings[1].accession_number]  # deleting the existing packet requires a source reread
    print("\n=== SANITY: M4-7 deleted rows ===")
    print("  deleting the newest filing's rows moved the frontier back to 2026-09-20; the next run re-read and re-saved it. Validated.")


def test_m4_8_a_scoped_run_leaves_the_next_window_unchanged(tmp_path, filing_store, monkeypatch):
    ctx = _ctx(tmp_path, filing_store)
    _seed_hr(filing_store, "0000000009", "2026-09-20")
    inside, beyond = _filing(1, "2026-09-18"), _filing(2, "2026-09-25")
    windows, _ = _patch(monkeypatch, [inside, beyond])

    _run(ctx, tickers=["AAPL"])
    scoped_watermark = _stored_watermark(filing_store)
    _run(ctx)

    assert windows[0] == (pd.Timestamp("2026-09-13"), pd.Timestamp("2026-09-20"))  # -t never reads past the frontier
    assert scoped_watermark == pd.Timestamp("2026-09-20") and beyond.reads == [beyond.accession_number]
    assert windows[1] == (pd.Timestamp("2026-09-13"), _AS_OF)  # the full run's window is unchanged
    assert _saved_ciks(filing_store) == ["0000000001", "0000000002", "0000000009"]
    print("\n=== SANITY: M4-8 -t then full run ===")
    print("  -t AAPL read 2026-09-13:2026-09-20 only and left the frontier at 2026-09-20; the full run then read 2026-09-13:2026-09-30. Validated.")


def test_listing_reads_one_quarterly_index_at_a_time(monkeypatch):
    """A multi-year window is listed quarter by quarter, each request clipped to the window, in order and
    on a fresh HTTP client (edgartools leaks a pooled connection per 304); empty quarters are skipped."""
    calls: list[str] = []
    resets: list[int] = []

    def _get_filings(form: Any, filing_date: str) -> Any:
        calls.append(filing_date)
        if filing_date.startswith("2000-04-01"):
            return None
        return SimpleNamespace(data=pa.table({"accession_number": [filing_date]}))

    monkeypatch.setattr(f13, "get_filings", _get_filings)
    monkeypatch.setattr(f13, "close_clients", lambda: resets.append(len(calls)))
    monkeypatch.setattr(f13, "Filings", lambda data: SimpleNamespace(data=data))
    listing = f13._listing(["13F-HR"], pd.Timestamp("1999-07-01"), pd.Timestamp("2000-08-15"))

    expected = ["1999-07-01:1999-09-30", "1999-10-01:1999-12-31", "2000-01-01:2000-03-31", "2000-04-01:2000-06-30", "2000-07-01:2000-08-15"]
    assert calls == expected
    assert resets == list(range(len(expected)))  # a fresh client before every quarterly request
    assert listing.data.column("accession_number").to_pylist() == [w for w in expected if not w.startswith("2000-04-01")]
    assert f13._listing(["13F-HR"], pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-05")) is not None
    print("\n=== SANITY: quarterly listing ===")
    print(
        f"  1999-07-01:2000-08-15 -> {len(calls)} sequential single-quarter requests, clipped at both ends; an empty quarter is skipped. Validated."
    )
