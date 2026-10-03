"""Tests for the extraction run-manifest (`src/data_extract/utils/common/run_manifest.py`):
the single JSON checkpoint (`data/extraction_manifest.json`) that records, per DB table, the
last run's date / ticker count / rows added, and drives the `since` window for the EDGAR
filing-listing fetchers. All offline (tmp_path-based fake context), no network/DB needed.
"""

from __future__ import annotations

import json
import os
import time
import types
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.run_manifest import changed_scope_tickers, manifest_window, record_run
from src.data_extract.utils.common.run_manifest import get_entry as _get_entry
from src.data_store.schema import Tables
from tests.data_extract.fake_context import extract_config

get_entry: Any = _get_entry
FIVE = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _ctx(tmp_path) -> Any:
    return types.SimpleNamespace(paths={"DATA_STORE": tmp_path}, config=extract_config())


def test_record_run_roundtrip(tmp_path):
    ctx = _ctx(tmp_path)
    assert get_entry(ctx, "sec_8k") is None  # nothing recorded yet

    record_run(ctx, "sec_8k", ticker_count=5, rows_added=12, is_full_rescan=True)
    entry = get_entry(ctx, "sec_8k")

    assert (tmp_path / "extraction_manifest.json").exists()
    assert entry["ticker_count"] == 5
    assert entry["rows_added"] == 12
    assert entry["last_run_date"] == pd.Timestamp.today().strftime("%Y-%m-%d")
    assert entry["last_full_rescan_date"] == entry["last_run_date"]

    print("\n=== SANITY CHECK: run_manifest round-trip ===")
    print(f"  wrote/read extraction_manifest.json: {entry}. Validated.")


def test_identity_scope_change_is_detected_per_ticker_with_unchanged_universe(tmp_path):
    ctx = _ctx(tmp_path)
    prior = {"AAA": "same", "BBB": "old"}
    current = {"AAA": "same", "BBB": "new"}
    assert changed_scope_tickers({"identity_scope_fingerprints": prior}, current) == frozenset({"BBB"})

    record_run(
        ctx,
        "sec_8k",
        ticker_count=2,
        rows_added=0,
        identity_scope_fingerprints=current,
    )
    assert get_entry(ctx, "sec_8k")["identity_scope_fingerprints"] == current
    print("\nSANITY: unchanged ticker count still invalidates only BBB after its identity scope changes.")


def test_record_run_does_not_clobber_sibling_tables(tmp_path):
    ctx = _ctx(tmp_path)
    record_run(ctx, "sec_8k", ticker_count=5, rows_added=12, is_full_rescan=True)
    record_run(ctx, "sec_13d", ticker_count=3, rows_added=1, is_full_rescan=True)

    assert get_entry(ctx, "sec_8k")["ticker_count"] == 5  # untouched by the sec_13d write
    assert get_entry(ctx, "sec_13d")["ticker_count"] == 3

    print("\n=== SANITY CHECK: run_manifest multi-table merge ===")
    print("  sec_8k and sec_13d entries coexist in one file without clobbering. Validated.")


def test_record_run_preserves_last_full_rescan_date_on_routine_run(tmp_path):
    ctx = _ctx(tmp_path)
    first = pd.Timestamp.today().normalize() - pd.Timedelta(days=10)
    record_run(ctx, Tables.def14a_edgar, ticker_count=5, rows_added=5, is_full_rescan=True, run_date=first)

    # a routine (non-rescan) run a few days later: last_run_date advances, but
    # last_full_rescan_date must stay pinned to the last TRUE full rescan
    second = first + pd.Timedelta(days=3)
    record_run(ctx, Tables.def14a_edgar, ticker_count=5, rows_added=2, is_full_rescan=False, run_date=second)

    entry = get_entry(ctx, Tables.def14a_edgar)
    assert entry["last_run_date"] == second.strftime("%Y-%m-%d")
    assert entry["last_full_rescan_date"] == first.strftime("%Y-%m-%d")

    print("\n=== SANITY CHECK: last_full_rescan_date pinned across routine runs ===")
    print(f"  last_run_date advanced to {entry['last_run_date']} while last_full_rescan_date stayed at {entry['last_full_rescan_date']}. Validated.")


def test_manifest_window_full_rescan_on_first_run(tmp_path):
    ctx = _ctx(tmp_path)
    fallback = pd.Timestamp("2011-01-01")
    since, is_full_rescan = manifest_window(ctx, "sec_13d", FIVE, fallback_since=fallback, full_rescan_days=30)
    assert is_full_rescan is True
    assert since == fallback

    print("\n=== SANITY CHECK: manifest_window on a first run ===")
    print(f"  no prior entry -> full rescan, since={since}. Validated.")


def test_manifest_window_full_rescan_when_ticker_count_changes(tmp_path):
    ctx = _ctx(tmp_path)
    record_run(ctx, "sec_13d", ticker_count=5, rows_added=1, is_full_rescan=True, tickers=FIVE)

    fallback = pd.Timestamp("2011-01-01")
    since, is_full_rescan = manifest_window(ctx, "sec_13d", [*FIVE, "FFF"], fallback_since=fallback, full_rescan_days=30)
    assert is_full_rescan is True
    assert since == fallback

    print("\n=== SANITY CHECK: manifest_window on a universe change ===")
    print("  5 -> 6 tickers -> full rescan (new ticker needs full history). Validated.")


def test_manifest_window_full_rescan_when_same_size_membership_changes(tmp_path):
    ctx = _ctx(tmp_path)
    record_run(ctx, "sec_13d", ticker_count=2, rows_added=1, is_full_rescan=True, tickers=["AAA", "BBB"])

    fallback = pd.Timestamp("2011-01-01")
    since, is_full_rescan = manifest_window(ctx, "sec_13d", ["AAA", "CCC"], fallback_since=fallback, full_rescan_days=30)
    assert is_full_rescan is True and since == fallback
    print("\n=== SANITY CHECK: manifest membership identity ===")
    print("  AAA/BBB -> AAA/CCC keeps count=2 but forces a full rescan. Validated.")


def test_manifest_window_full_rescan_after_self_heal_window_elapses(tmp_path):
    ctx = _ctx(tmp_path)
    stale = pd.Timestamp.today().normalize() - pd.Timedelta(days=31)
    record_run(ctx, "sec_13d", ticker_count=5, rows_added=1, is_full_rescan=True, run_date=stale, tickers=FIVE)

    fallback = pd.Timestamp("2011-01-01")
    since, is_full_rescan = manifest_window(ctx, "sec_13d", FIVE, fallback_since=fallback, full_rescan_days=30)
    assert is_full_rescan is True
    assert since == fallback

    print("\n=== SANITY CHECK: manifest_window self-heal after 30 days ===")
    print(f"  last full rescan {stale.date()} is 31 days old (>= 30) -> forces a fresh full rescan, since={since}. Validated.")


def test_manifest_window_narrows_on_routine_rerun(tmp_path):
    ctx = _ctx(tmp_path)
    recent = pd.Timestamp.today().normalize() - pd.Timedelta(days=5)
    record_run(ctx, "sec_13d", ticker_count=5, rows_added=1, is_full_rescan=True, run_date=recent, tickers=FIVE)

    fallback = pd.Timestamp("2011-01-01")
    since, is_full_rescan = manifest_window(ctx, "sec_13d", FIVE, fallback_since=fallback, full_rescan_days=30)
    assert is_full_rescan is False
    assert since == recent

    print("\n=== SANITY CHECK: manifest_window narrows on a routine rerun ===")
    print(f"  same ticker set, rescan not due -> since={since} (the manifest's last run date, inclusive), not the full-history fallback. Validated.")


def test_manifest_window_same_size_swap_gives_the_new_ticker_the_full_window(tmp_path):
    """The count is derived from the ticker list, so a caller cannot pass a count without the list."""
    ctx = _ctx(tmp_path)
    recent = pd.Timestamp.today().normalize() - pd.Timedelta(days=5)
    record_run(ctx, "sec_13d", ticker_count=2, rows_added=1, is_full_rescan=True, run_date=recent, tickers=["AAA", "BBB"])
    fallback = pd.Timestamp("2011-01-01")

    swap_since, swap_full = manifest_window(ctx, "sec_13d", ["AAA", "CCC"], fallback_since=fallback, full_rescan_days=30)
    same_since, same_full = manifest_window(ctx, "sec_13d", ["BBB", "AAA"], fallback_since=fallback, full_rescan_days=30)

    assert (swap_since, swap_full) == (fallback, True)
    assert (same_since, same_full) == (recent, False)
    print("\n=== SANITY CHECK: manifest_window keyed on the ticker list ===")
    print(f"  AAA/BBB -> AAA/CCC (same size) -> since={swap_since.date()} full; same set reordered -> since={same_since.date()} narrow. Validated.")


def test_a_crash_between_temp_write_and_replace_keeps_the_previous_manifest(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    record_run(ctx, "sec_8k", ticker_count=5, rows_added=12, is_full_rescan=True)
    before = get_entry(ctx, "sec_8k")

    def _crash(*_args, **_kwargs):
        raise OSError("simulated crash before the rename")

    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(OSError, match="simulated crash"):
        record_run(ctx, "sec_8k", ticker_count=6, rows_added=99, is_full_rescan=False)
    monkeypatch.undo()

    after = get_entry(ctx, "sec_8k")
    assert after == before
    assert json.loads((tmp_path / "extraction_manifest.json").read_text(encoding="utf-8"))["sec_8k"]["rows_added"] == 12
    print("\n=== SANITY CHECK: atomic manifest write ===")
    print(
        f"  os.replace raised after the temp write -> the manifest still loads with rows_added={after['rows_added']} (the previous run). Validated."
    )


def _replace_failing(times: int) -> Any:
    """An `os.replace` that raises `PermissionError` (a Windows reader holds the target) `times` times, then renames."""
    real_replace, calls = os.replace, []

    def _replace(src: Any, dst: Any) -> None:
        calls.append(1)
        if len(calls) <= times:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        real_replace(src, dst)

    _replace.calls = calls  # type: ignore[attr-defined]
    return _replace


def test_a_held_manifest_is_retried_and_leaves_no_temp_file(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    record_run(ctx, "sec_8k", ticker_count=5, rows_added=12, is_full_rescan=True)
    replace = _replace_failing(times=1)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    record_run(ctx, "sec_8k", ticker_count=6, rows_added=99, is_full_rescan=False)
    monkeypatch.undo()

    assert len(replace.calls) == 2
    assert get_entry(ctx, "sec_8k")["rows_added"] == 99
    assert list(tmp_path.glob("*.tmp")) == []
    print("\n=== SANITY CHECK: manifest held by a reader ===")
    print(f"  1st os.replace raised PermissionError, the 2nd renamed: rows_added={get_entry(ctx, 'sec_8k')['rows_added']}, no .tmp left. Validated.")


def test_a_manifest_held_for_good_raises_keeps_the_previous_file_and_no_temp(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    record_run(ctx, "sec_8k", ticker_count=5, rows_added=12, is_full_rescan=True)
    monkeypatch.setattr(os, "replace", _replace_failing(times=10_000))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        record_run(ctx, "sec_8k", ticker_count=6, rows_added=99, is_full_rescan=False)
    monkeypatch.undo()

    assert get_entry(ctx, "sec_8k")["rows_added"] == 12
    assert list(tmp_path.glob("*.tmp")) == []
    print("\n=== SANITY CHECK: manifest held for good ===")
    print("  every os.replace raised PermissionError -> record_run raised, previous manifest (rows_added=12) intact, no .tmp left. Validated.")
