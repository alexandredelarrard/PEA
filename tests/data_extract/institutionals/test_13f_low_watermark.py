"""
test_13f_low_watermark.py (tests/data_extract/institutionals/test_13f_low_watermark.py)
---------------------------------------------------------------------------------------
The nightly 13F walk (M4) on a real SQLite `DataStore` with fake filings (no network): the resume
work list is missing/failed accessions inside fixed configured filing dates. Successful books save
in batches while failed reads alone get retry rounds after the first pass. The 13F data-set ingest is in `test_13f_zips.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.institutionals import fetch_13f as f13
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.window import Window, WindowRun
from src.validate.checks.institutionals import check_institutionals
from tests.data_extract.edgar_fixtures import fake_context
from tests.data_extract.institutionals.test_13f_one_walk import ROSTER, _FakeFiling, _line, _patch_walk, _seed_roster
from tests.support.store_identity import store_identity

_AS_OF = pd.Timestamp("2026-09-30")
_UNIVERSE = ["AAPL", "MSFT"]
_AAPL = [_line("037833100", "APPLE INC", 1_000.0, 10)]


def _patch(monkeypatch: pytest.MonkeyPatch, filings: list[_FakeFiling]) -> tuple[list[Window], list[float]]:
    windows = _patch_walk(monkeypatch, filings)
    waits: list[float] = []
    monkeypatch.setattr(f13, "_sleep", waits.append)
    return windows, waits


def _ctx(tmp_path: Any, store: Any) -> Any:
    ctx = fake_context(tmp_path, store, _UNIVERSE)
    ctx._identity_resolver = store_identity(store)
    _seed_roster(store, [ROSTER])
    return ctx


def _filing(n: int, filed: str, **kwargs: Any) -> _FakeFiling:
    return _FakeFiling(f"{n:010d}", filed, "2026-06-30", list(_AAPL), accession=f"{n:010d}-26-{n:06d}", **kwargs)


def _saved_ciks(store: Any) -> list[str]:
    df = store.load(Tables.sec13f_hr, columns=["cik"], optional=True)
    return [] if df is None else sorted(df["cik"].unique())


def _run(ctx: Any, as_of: pd.Timestamp = _AS_OF, *, filing_window: Window | None = None, full: bool = False) -> int:
    ctx.run = WindowRun("full" if full else "incremental", "13F source run", as_of)
    ctx.config.data_extract.sec13f.batch_size = 1
    return f13.fetch_13f(ctx, filing_window=filing_window)


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
    saver = f13._Saver(ctx, save_every=100)
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
    assert windows[1].start <= pd.Timestamp(failing.filing_date)  # the next window includes the failed filing
    assert _saved_ciks(filing_store) == ["0000000001", "0000000002", "0000000003", "0000000009"]
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and failing.accession_number in r.getMessage()]
    assert len(errors) == 1 and "skipped" in errors[0], errors
    assert not any("held back" in r.getMessage() for r in caplog.records)
    print("\n=== SANITY: 13F recent failure ===")
    print(f"  night 1: the 2026-09-28 filing failed after 3 rounds {waits}; saved {night1}, watermark {watermark1:%Y-%m-%d};")
    print(f"  night 2 window starts {windows[1].start:%Y-%m-%d} <= 2026-09-28 and retries the failed accession only. Validated.")


def test_a_failure_recovered_in_a_round_saves_every_filing(tmp_path, filing_store, monkeypatch):
    filings = [_filing(1, "2026-09-27"), _filing(2, "2026-09-28", fail_reads=1), _filing(3, "2026-09-29")]
    ctx = _ctx(tmp_path, filing_store)
    _, waits = _patch(monkeypatch, filings)

    _run(ctx)

    assert _saved_ciks(filing_store) == ["0000000001", "0000000002", "0000000003"]
    assert waits == [60.0]
    print("\n=== SANITY: 13F retry round ===")
    print(f"  one transient failure recovered in round 1 (waits {waits}); all 3 filings saved in order. Validated.")


def test_an_older_failed_accession_is_logged_and_remains_retryable(
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
@pytest.mark.parametrize("start", ["2021-09-30", "1999-07-01"])
def test_absent_table_uses_the_fixed_configured_start(tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch, start: str) -> None:
    ctx = _ctx(tmp_path, filing_store)
    ctx.config.data.tables.sec13f_hr.start_date = start
    windows, _ = _patch(monkeypatch, [_filing(1, "2026-09-29")])
    _run(ctx)
    assert windows == [Window(pd.Timestamp(start), _AS_OF)]
    print("SANITY: a cold 13F table uses configured fixed dates, without moving history years.")


def test_missing_older_filing_is_discovered_behind_the_stored_frontier(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(tmp_path, filing_store)
    _seed_hr(filing_store, "0000000009", "2026-09-20")
    old, recent = _filing(1, "2026-09-12"), _filing(2, "2026-09-25")
    windows, _ = _patch(monkeypatch, [old, recent])
    _run(ctx)
    assert windows == [Window(pd.Timestamp("1999-07-01"), _AS_OF)]
    assert _saved_ciks(filing_store) == ["0000000001", "0000000002", "0000000009"]
    print("SANITY: an older missing accession is discovered in the cached filing index despite a later stored filing.")


def test_explicit_recovery_narrows_dates_without_replaying_completed_packets(
    tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(tmp_path, filing_store)
    _seed_hr(filing_store, "0000000009", "2026-09-20")
    lost = _filing(1, "2026-08-01")
    windows, _ = _patch(monkeypatch, [lost])
    _run(ctx)
    recovery = Window(pd.Timestamp("2026-07-30"), pd.Timestamp("2026-08-02"))
    _run(ctx, filing_window=recovery)
    assert windows[-1] == recovery and lost.reads == [lost.accession_number]
    assert _saved_ciks(filing_store) == ["0000000001", "0000000009"]
    assert _stored_watermark(filing_store) == pd.Timestamp("2026-09-20")
    print("SANITY: named recovery stays within config bounds and valid completed packets remain reused.")


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


def test_explicit_full_replays_only_the_requested_window(tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    before, inside, after = (_filing(n, day) for n, day in enumerate(["2026-08-01", "2026-09-18", "2026-09-25"], 1))
    windows, _ = _patch(monkeypatch, [before, inside, after])
    _run(ctx)
    for filing in (before, inside, after):
        filing.reads.clear()
    recovery = Window(pd.Timestamp("2026-09-15"), pd.Timestamp("2026-09-20"))
    _run(ctx, filing_window=recovery, full=True)
    assert windows[-1] == recovery
    assert not before.reads and inside.reads == [inside.accession_number] and not after.reads
    assert _saved_ciks(filing_store) == ["0000000001", "0000000002", "0000000003"]
    print("SANITY: Context full mode replays only complete source packets inside the explicit finite Window; outside packets survive.")


def test_listing_uses_the_shared_index_with_actual_sorting(tmp_path: Path, filing_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path, filing_store)
    original = _filing(2, "2026-09-18")
    amended = _FakeFiling("0000000002", "2026-09-18", "2026-06-30", list(_AAPL), form="13F-HR/A", accession="0000000001-26-000001")
    outside = _filing(3, "2026-08-18")
    windows, _ = _patch(monkeypatch, [amended, outside, original])
    window = Window(pd.Timestamp("2026-09-01"), _AS_OF)
    listed = f13._listing(ctx, window)
    assert windows == [window] and listed == [original, amended]
    print("SANITY: shared index bounds discard older metadata; same-day originals precede amendments regardless of accession order.")
