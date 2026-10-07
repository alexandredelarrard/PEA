"""The resume planners on the real `DataStore` (SQLite): `document_worklist` (M2, EDGAR documents, on a
seeded local index), `series_windows` (M1) and `archive_worklist` (M3, bulk archives), each with the
AC-002 cases. Each case plans from the table's own rows, its `schema.Resume` contract and the run date only."""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.common.edgar_driver import FilingStamp, marker_row
from src.data_extract.utils.common.resume import (
    DONE_PER_KEY,
    KEY_ESTABLISHED,
    KEY_NEW,
    KEY_ROWLESS,
    archive_worklist,
    document_worklist,
)
from src.data_extract.utils.common.sec_utils import CIK_MAPPING_COLS
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_context, fake_filing, seed_index

_AS_OF = pd.Timestamp("2026-09-30")
_FORMS = ("8-K", "8-K/A")


def _rows(cik: int, dates: list[str], prefix: str) -> list[tuple]:
    return [(cik, f"Co {cik}", "8-K", d, f"{prefix}-{i:03d}") for i, d in enumerate(dates)]


def _universe(ctx, tickers: dict[str, str], added_on: dict[str, str] | None = None) -> pd.DataFrame:
    cik_map = pd.DataFrame({"ticker": list(tickers), "cik": [c.zfill(10) for c in tickers.values()]})
    if added_on is not None:
        ctx.store.drop(Tables.sp500_tickers)  # recreated with the column
        ctx.store.save(
            Tables.sp500_tickers,
            pd.DataFrame(
                {col: ["x"] * len(tickers) for col in CIK_MAPPING_COLS}
                | {"ticker": list(tickers), "cik": list(tickers.values()), "added_on": [pd.Timestamp(added_on.get(t, "2000-01-01")) for t in tickers]}
            ),
        )
    return cik_map


def _plan(ctx, cik_map: pd.DataFrame, as_of: pd.Timestamp = _AS_OF, table=Tables.sec_8k, **kwargs):
    defaults = {"forms": _FORMS, "identity": None, "years_history": 5}
    return document_worklist(ctx, table, cik_map, as_of, **(defaults | kwargs))


def _store(ctx, ticker: str, accessions: list[str], filed: str = "2026-01-05", item: str = "8.01") -> None:
    ctx.store.save(
        Tables.sec_8k,
        pd.DataFrame({"ticker": ticker, "accession_number": accessions, "item": item, "filing_date": pd.Timestamp(filed), "form": "8-K"}),
    )


def _units(work, ticker: str) -> list[str]:
    return work.units[ticker]["accession"].tolist() if ticker in work.units else []


def test_absent_table_lists_every_index_entry_from_the_floor(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    seed_index(ctx, _rows(1, ["2019-06-01", "2022-03-01", "2026-09-01"], "a"))

    work = _plan(ctx, _universe(ctx, {"AAA": "1"}))

    assert _units(work, "AAA") == ["a-001", "a-002"]  # 2019 is before as_of - 5 years
    assert work.key_class == {"AAA": KEY_ROWLESS} and work.counts == {KEY_ROWLESS: 2}
    print("\n=== SANITY CHECK: M2 absent table ===")
    print(f"  no table -> {_units(work, 'AAA')} (2019 filing below the 5-year floor is not listed).")


def test_a_new_key_gets_its_full_history_and_other_keys_are_unchanged(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "NEW"])
    seed_index(ctx, _rows(1, ["2024-01-02", "2026-09-20"], "a") + _rows(2, ["2022-05-02", "2023-05-02", "2026-09-21"], "n"))
    cik_map = _universe(ctx, {"AAA": "1", "NEW": "2"}, added_on={"NEW": "2026-09-29"})
    _store(ctx, "AAA", ["a-000"])

    work = _plan(ctx, cik_map)

    assert _units(work, "NEW") == ["n-000", "n-001", "n-002"]
    assert _units(work, "AAA") == ["a-001"]
    assert work.key_class == {"AAA": KEY_ESTABLISHED, "NEW": KEY_NEW}
    assert work.counts == {KEY_NEW: 3, "forward": 1, "gap": 0}
    print("\n=== SANITY CHECK: M2 new key ===")
    print(f"  NEW (added 2026-09-29) lists its whole history {_units(work, 'NEW')}; AAA only its missing 8-K; counts {work.counts}.")


def test_forward_entries_and_an_old_gap_are_both_listed(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    seed_index(ctx, _rows(1, ["2023-03-01", "2025-06-02", "2026-09-10", "2026-09-28"], "a"))
    _store(ctx, "AAA", ["a-001", "a-002"])

    work = _plan(ctx, _universe(ctx, {"AAA": "1"}))

    assert _units(work, "AAA") == ["a-000", "a-003"]  # a 3-year-old gap and a forward filing
    assert work.counts == {"forward": 1, "gap": 1}
    print("\n=== SANITY CHECK: M2 forward + gap ===")
    print(f"  stored the middle two -> lists the 2023 gap and the 2026-09-28 forward filing; counts {work.counts}.")


def test_a_zero_row_filing_marker_is_never_listed_again(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    seed_index(ctx, _rows(1, ["2026-09-10", "2026-09-11"], "a"))
    marker = marker_row(Tables.sec_8k, "AAA", FilingStamp.of(fake_filing("a-000", 1, "2026-09-10"), "0000000001"))
    sqlite_store.save(Tables.sec_8k, marker)

    work = _plan(ctx, _universe(ctx, {"AAA": "1"}))

    assert _units(work, "AAA") == ["a-001"]
    assert sqlite_store.load(Tables.sec_8k, optional=True) is None  # the marker is not data
    print("\n=== SANITY CHECK: M2 zero-row unit ===")
    print("  a-000 stored only as a marker -> not listed; consumers still read no row.")


def test_a_failed_unit_is_listed_next_night_and_a_deleted_one_again(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    seed_index(ctx, _rows(1, ["2026-09-10", "2026-09-11", "2026-09-12"], "a"))
    cik_map = _universe(ctx, {"AAA": "1"})
    _store(ctx, "AAA", ["a-000", "a-002"])  # night 1: a-001 failed

    night2 = _plan(ctx, cik_map, as_of=_AS_OF + pd.Timedelta(days=1))
    sqlite_store.delete(Tables.sec_8k, {"accession_number": "a-002"})
    night3 = _plan(ctx, cik_map, as_of=_AS_OF + pd.Timedelta(days=2))

    assert _units(night2, "AAA") == ["a-001"]
    assert _units(night3, "AAA") == ["a-001", "a-002"]
    print("\n=== SANITY CHECK: M2 failed + deleted ===")
    print("  the failed a-001 is listed the next night; deleting a-002's rows lists it again (repair by delete).")


def test_a_ticker_run_then_a_full_run_lists_only_what_is_missing(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"])
    seed_index(ctx, _rows(1, ["2026-09-10", "2026-09-11"], "a") + _rows(2, ["2026-09-12"], "b"))
    cik_map = _universe(ctx, {"AAA": "1", "BBB": "2"})

    scoped = _plan(ctx, cik_map[cik_map["ticker"] == "AAA"])
    _store(ctx, "AAA", _units(scoped, "AAA"))
    everyone = _plan(ctx, cik_map)

    assert list(scoped.units) == ["AAA"] and _units(everyone, "AAA") == [] and _units(everyone, "BBB") == ["b-000"]
    full = _plan(ctx, cik_map, full=True)
    assert full.size == 3
    print("\n=== SANITY CHECK: M2 -t then full ===")
    print("  -t AAA lists only AAA; the following full run lists only BBB's filing; -F lists all 3 again.")


def test_role_markers_are_done_per_key(tmp_path, sqlite_store):
    """BLK files a 13G about AAA: the index lists it under both. BLK's filer-role marker must not hide it from AAA."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BLK"])
    seed_index(ctx, [(1, "AAA Inc", "SC 13G", "2026-09-10", "blk-13g"), (2, "BlackRock", "SC 13G", "2026-09-10", "blk-13g")])
    cik_map = _universe(ctx, {"AAA": "1", "BLK": "2"})
    stamp = FilingStamp.of(fake_filing("blk-13g", 2, "2026-09-10", form="SC 13G"), "0000000002")
    sqlite_store.save(Tables.sec_13g, marker_row(Tables.sec_13g, "BLK", stamp))

    forms = ("SC 13G", "SC 13G/A")
    per_key = _plan(ctx, cik_map, table=Tables.sec_13g, forms=forms, done_scope=DONE_PER_KEY)
    table_wide = _plan(ctx, cik_map, table=Tables.sec_13g, forms=forms)

    assert _units(per_key, "AAA") == ["blk-13g"] and _units(per_key, "BLK") == []
    assert _units(table_wide, "AAA") == []  # what a table-wide done set would lose
    print("\n=== SANITY CHECK: role markers per key ===")
    print("  BLK's filer-role marker leaves the 13G listed for its subject AAA; a table-wide done set would have dropped it.")


def test_the_cap_keeps_the_newest_documents(tmp_path, sqlite_store, caplog):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"])
    seed_index(ctx, _rows(1, ["2026-09-01", "2026-09-03", "2026-09-05"], "a") + _rows(2, ["2026-09-02", "2026-09-04"], "b"))

    with caplog.at_level("ERROR"):
        work = _plan(ctx, _universe(ctx, {"AAA": "1", "BBB": "2"}), cap=3)

    assert work.uncapped == 5 and work.size == 3
    assert _units(work, "AAA") == ["a-001", "a-002"] and _units(work, "BBB") == ["b-001"]
    assert any("exceeds the per-run cap of 3" in r.getMessage() for r in caplog.records)
    print("\n=== SANITY CHECK: cap ===")
    print("  5 listed, cap 3 -> the 3 newest across keys (each key oldest first), ERROR names the uncapped 5.")


def test_done_set_narrowing_keeps_units(tmp_path, sqlite_store):
    """A table-wide done set holding accessions the index never lists (another ticker's, an old one)
    plans exactly as the hand-computed difference."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"])
    seed_index(ctx, _rows(1, ["2025-06-02", "2026-09-10", "2026-09-28"], "a") + _rows(2, ["2026-09-12", "2026-09-20"], "b"))
    cik_map = _universe(ctx, {"AAA": "1", "BBB": "2"})
    _store(ctx, "AAA", ["a-001"])
    _store(ctx, "BBB", ["b-000"])
    _store(ctx, "ZZZ", ["z-000", "z-001"])  # stored, never listed for the planned keys
    _store(ctx, "AAA", ["a-old"], filed="2015-01-05")  # below the floor, so not in the index read

    work = _plan(ctx, cik_map)

    assert _units(work, "AAA") == ["a-000", "a-002"] and _units(work, "BBB") == ["b-001"]
    assert work.key_class == {"AAA": KEY_ESTABLISHED, "BBB": KEY_ESTABLISHED}
    assert work.counts == {"forward": 2, "gap": 1}
    print("\n=== SANITY CHECK: done-set narrowing ===")
    print(f"  5 stored, 3 never listed -> AAA {_units(work, 'AAA')}, BBB {_units(work, 'BBB')}, counts {work.counts}.")


# --------------------------------------------------------------------------- #
# M1: `series_windows` (dated per-key series)                                 #
# --------------------------------------------------------------------------- #
_UNTIL = pd.Timestamp("2026-09-30")
_SESSIONS = pd.bdate_range("2026-08-03", _UNTIL)


def _bars(ticker: str, dates) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ticker, "date": pd.DatetimeIndex(dates), "close_split": 1.0, "close_total": 1.0})


def _series(ctx, keys: list[str], *, as_of: pd.Timestamp = _AS_OF, full: bool = False, table=Tables.prices):
    from src.data_extract.utils.common.resume import series_windows, trading_calendar

    return series_windows(ctx, table, keys, as_of, until=_UNTIL, years_history=5, full=full, calendar=trading_calendar(ctx))


def _spans(work, key: str) -> list[tuple[str, str]]:
    return [(str(a.date()), str(b.date())) for a, b in work.windows[key]]


def test_m1_absent_table_gives_every_key_its_full_window(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"])

    work = _series(ctx, ["AAA", "BBB"])

    assert _spans(work, "AAA") == _spans(work, "BBB") == [("2021-09-30", "2026-09-30")]
    assert work.key_class == {"AAA": KEY_NEW, "BBB": KEY_NEW} and len(work.groups()) == 1
    print("\n=== SANITY CHECK: M1 absent table ===")
    print("  no prices table -> both keys from as_of - 5y to the last session, in ONE download group.")


def test_m1_a_new_key_alone_gets_its_full_window(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "NEW"])
    _universe(ctx, {"AAA": "1", "NEW": "2"}, added_on={"NEW": "2026-09-28"})
    sqlite_store.save(Tables.prices, _bars("AAA", _SESSIONS))
    sqlite_store.save(Tables.prices, _bars("NEW", _SESSIONS[-3:]))

    work = _series(ctx, ["AAA", "NEW"])

    assert _spans(work, "NEW") == [("2021-09-30", "2026-09-30")]
    assert _spans(work, "AAA") == [("2026-09-23", "2026-09-30")]
    assert work.key_class == {"AAA": KEY_ESTABLISHED, "NEW": KEY_NEW} and len(work.groups()) == 2
    print("\n=== SANITY CHECK: M1 new key ===")
    print("  NEW (added 2026-09-28, inside the 7-day overlap) -> full window; AAA stays forward from 2026-09-23.")


def test_m1_a_new_key_is_pulled_in_full_once_then_resumes_forward(tmp_path, sqlite_store):
    """REQ-010: the second run of the night (and every later night inside the new-key window) is forward-only
    once the key's stored history reaches the floor; a key whose history starts past the tolerance stays full."""
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "NEW", "LATE"])
    _universe(ctx, {"AAA": "1", "NEW": "2", "LATE": "3"}, added_on={"NEW": "2026-09-28", "LATE": "2026-09-28"})
    sqlite_store.save(Tables.prices, _bars("AAA", _SESSIONS))

    first = _series(ctx, ["NEW"])
    ((since, until),) = first.windows["NEW"]
    sqlite_store.save(Tables.prices, _bars("NEW", pd.bdate_range(since, until)[1:]))  # floor 09-30 (Thu): first bar 10-01
    sqlite_store.save(Tables.prices, _bars("LATE", pd.bdate_range("2021-10-15", until)))  # 15 days past the floor
    again = _series(ctx, ["AAA", "NEW", "LATE"])
    next_night = _series(ctx, ["AAA", "NEW", "LATE"], as_of=_AS_OF + pd.Timedelta(days=1))

    assert _spans(first, "NEW") == [("2021-09-30", "2026-09-30")] and first.key_class["NEW"] == KEY_NEW
    assert _spans(again, "NEW") == _spans(again, "AAA") == [("2026-09-23", "2026-09-30")]
    assert again.key_class["NEW"] == KEY_ESTABLISHED and next_night.key_class["NEW"] == KEY_ESTABLISHED
    assert _spans(again, "LATE") == [("2021-09-30", "2026-09-30")] and again.key_class["LATE"] == KEY_NEW
    print("\n=== SANITY CHECK: M1 new key, same-night rerun (REQ-010) ===")
    print(f"  NEW: full {_spans(first, 'NEW')} once; stored from 2021-10-01 (1 day past the floor) -> rerun {_spans(again, 'NEW')};")
    print("  LATE, stored only from 2021-10-15 (past the 7-day tolerance), still gets the full window.")


def test_m1_forward_window_is_the_key_s_own_last_date_minus_the_overlap(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "LAG"])
    sqlite_store.save(Tables.prices, _bars("AAA", _SESSIONS))
    sqlite_store.save(Tables.prices, _bars("LAG", _SESSIONS[_SESSIONS <= pd.Timestamp("2026-09-10")]))

    work = _series(ctx, ["AAA", "LAG"])

    assert _spans(work, "AAA") == [("2026-09-23", "2026-09-30")]
    assert _spans(work, "LAG") == [("2026-09-03", "2026-09-30")]
    print("\n=== SANITY CHECK: M1 forward window ===")
    print("  each key resumes from its OWN last date - 7 days: AAA from 09-23, the lagging LAG from 09-03.")


def test_m1_a_rowless_established_key_resumes_from_the_table_frontier(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "OLD"])
    _universe(ctx, {"AAA": "1", "OLD": "2"}, added_on={})
    sqlite_store.save(Tables.prices, _bars("AAA", _SESSIONS))

    work = _series(ctx, ["AAA", "OLD"])

    assert _spans(work, "OLD") == [("2026-09-23", "2026-09-30")] and work.key_class["OLD"] == KEY_ROWLESS
    print("\n=== SANITY CHECK: M1 rowless key (D1a) ===")
    print("  OLD has no row and is not new -> table-wide last date - overlap, not the full history.")


def test_m1_a_three_session_hole_becomes_a_three_session_window(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"])
    hole = _SESSIONS[(_SESSIONS >= pd.Timestamp("2026-08-17")) & (_SESSIONS <= pd.Timestamp("2026-08-19"))]
    sqlite_store.save(Tables.prices, _bars("AAA", _SESSIONS))
    sqlite_store.save(Tables.prices, _bars("BBB", _SESSIONS.difference(hole)))

    work = _series(ctx, ["AAA", "BBB"])

    assert _spans(work, "BBB") == [("2026-08-17", "2026-08-19"), ("2026-09-23", "2026-09-30")]
    assert _spans(work, "AAA") == [("2026-09-23", "2026-09-30")]
    groups = [(str(a.date()), str(b.date()), keys) for a, b, keys in work.groups()]
    assert groups == [("2026-08-17", "2026-08-19", ["BBB"]), ("2026-09-23", "2026-09-30", ["AAA", "BBB"])]
    print("\n=== SANITY CHECK: M1 hole ===")
    print(f"  BBB misses 3 sessions {list(hole.strftime('%m-%d'))} -> one extra 3-session window; groups {groups}.")


def test_m1_zero_row_units_do_not_apply():
    """A daily series has no 'read but empty' unit: a session with no bar is a hole and is re-fetched, never marked."""
    assert all(t.empty_marker is None for t in (Tables.prices, Tables.dividends, Tables.prices_splits, Tables.short_interest))
    print("\n=== SANITY CHECK: M1 zero-row unit ===")
    print("  not applicable: no M1 table declares an empty-filing marker.")


def test_m1_a_failed_night_is_re_listed_and_deleted_rows_are_re_fetched(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"])
    sqlite_store.save(Tables.prices, _bars("AAA", _SESSIONS[_SESSIONS <= pd.Timestamp("2026-09-28")]))
    sqlite_store.save(Tables.prices, _bars("BBB", _SESSIONS))  # the calendar keeps every session

    night1 = _series(ctx, ["AAA"])
    night2 = _series(ctx, ["AAA"], as_of=_AS_OF + pd.Timedelta(days=1))  # night 1 saved nothing
    sqlite_store.delete(Tables.prices, {"ticker": "AAA", "date": [pd.Timestamp("2026-08-20"), pd.Timestamp("2026-09-28")]})
    night3 = _series(ctx, ["AAA"], as_of=_AS_OF + pd.Timedelta(days=2))

    assert _spans(night1, "AAA") == _spans(night2, "AAA") == [("2026-09-21", "2026-09-30")]
    assert _spans(night3, "AAA") == [("2026-08-20", "2026-08-20"), ("2026-09-18", "2026-09-30")]
    print("\n=== SANITY CHECK: M1 failed night + deleted rows ===")
    print("  a failed night leaves the frontier, so night 2 lists the same window; deleting 08-20 (interior) and")
    print("  09-28 (the tail) re-lists exactly those: a 1-session hole and a forward window from the new last date.")


def test_m1_a_ticker_run_then_a_full_run_re_pulls_nothing(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA", "BBB"])
    sqlite_store.save(Tables.prices, _bars("BBB", _SESSIONS))

    scoped = _series(ctx, ["AAA"])
    ((since, until),) = scoped.windows["AAA"]
    sqlite_store.save(Tables.prices, _bars("AAA", _SESSIONS[(_SESSIONS >= since) & (_SESSIONS <= until)]))
    everyone = _series(ctx, ["AAA", "BBB"])

    assert list(scoped.windows) == ["AAA"]
    assert _spans(everyone, "AAA") == _spans(everyone, "BBB") == [("2026-09-23", "2026-09-30")]
    assert _spans(_series(ctx, ["AAA", "BBB"], full=True), "BBB") == [("2021-09-30", "2026-09-30")]
    print("\n=== SANITY CHECK: M1 -t then full ===")
    print("  -t AAA plans only AAA; the following full run is forward-only for both; -F re-lists the whole window.")


def test_m1_merge_unions_each_key_s_windows(tmp_path, sqlite_store):
    ctx = fake_context(tmp_path, sqlite_store, ["AAA"])
    sqlite_store.save(Tables.prices, _bars("AAA", _SESSIONS))
    sqlite_store.save(Tables.dividends, pd.DataFrame({"ticker": "AAA", "date": _SESSIONS[_SESSIONS <= pd.Timestamp("2026-09-15")], "dividends": 0.0}))

    merged = _series(ctx, ["AAA"]).merge(_series(ctx, ["AAA"], table=Tables.dividends))

    assert _spans(merged, "AAA") == [("2026-09-08", "2026-09-30")]
    print("\n=== SANITY CHECK: prices + dividends windows ===")
    print("  dividends lag to 09-15 -> the one download for AAA starts at 09-08, covering both tables.")


# --------------------------------------------------------------------------- #
# M3: bulk archives (`archive_worklist`)                                        #
# --------------------------------------------------------------------------- #
_NOTES = (Tables.notes_num, Tables.notes_text)


def _archive_ctx(tmp_path, store, tickers: list[str], added_on: dict[str, str] | None = None):
    ctx = fake_context(tmp_path, store, tickers, redundant_ticks=[])
    if added_on is not None:
        _universe(ctx, {t: str(i + 1) for i, t in enumerate(tickers)}, added_on=added_on)
    return ctx


def _save_notes(store, table, ticker: str, periods: list[str]) -> None:
    rows = [{"adsh": f"{ticker}-{p}", "tag": "t", "ddate": pd.Timestamp("2026-06-30"), "qtrs": 0, "ticker": ticker, "period": p} for p in periods]
    store.save(table, pd.DataFrame(rows))


def _save_pension(store, ticker: str, quarters: list[str]) -> None:
    rows = [{"cik": ticker, "tag": "t", "ddate": pd.Timestamp("2025-12-31"), "qtrs": 0, "ticker": ticker, "quarter": q} for q in quarters]
    store.save(Tables.pension_facts, pd.DataFrame(rows))


def _archive(ctx, tables, published: list[str], keys: list[str], cached: set[str] | None = None, **kwargs):
    return archive_worklist(ctx, tables, published, set(published) if cached is None else cached, keys, _AS_OF, **kwargs)


def test_notes_period_counts_only_when_in_both_tables(tmp_path, sqlite_store):
    ctx = _archive_ctx(tmp_path, sqlite_store, ["AAPL"])
    _save_notes(sqlite_store, Tables.notes_num, "AAPL", ["2026_07"])
    _save_notes(sqlite_store, Tables.notes_text, "AAPL", ["2026_07", "2026_08"])

    work = _archive(ctx, _NOTES, ["2026_07", "2026_08"], ["AAPL"])

    assert work.pending == ["2026_08"] and work.rescan == []
    print("\n=== SANITY CHECK: M3 notes period stored in one table only ===")
    print("  2026_08 holds notes_text rows but no notes_num row -> parsed again (the old union skipped it).")


def test_m3_absent_tables_list_every_published_period_from_the_source_start(tmp_path, sqlite_store):
    ctx = _archive_ctx(tmp_path, sqlite_store, ["AAPL"])

    ftd = _archive(ctx, (Tables.sec_fails_to_deliver,), ["200906b", "200907a", "202609a"], ["AAPL"])
    notes = _archive(ctx, _NOTES, ["2026_07", "2026_08"], ["AAPL"])

    assert ftd.pending == ["200907a", "202609a"]  # 200906b ends before the 2009-07-01 source start
    assert notes.pending == ["2026_07", "2026_08"] and not ftd.scoped
    print("\n=== SANITY CHECK: M3 absent tables ===")
    print(f"  FTD -> {ftd.pending} (200906b clamped by source_start); notes -> {notes.pending}.")


def test_m3_new_key_re_parses_every_cached_period_for_itself_only(tmp_path, sqlite_store, caplog):
    ctx = _archive_ctx(tmp_path, sqlite_store, ["AAA", "NEW", "OLD"], added_on={"NEW": "2026-09-27", "OLD": "2026-09-28"})
    _save_pension(sqlite_store, "AAA", ["2025q4", "2026q1", "2026q2"])
    _save_pension(sqlite_store, "OLD", ["2026q2"])  # new, but already holds a row: not re-parsed

    with caplog.at_level("WARNING"):
        work = _archive(ctx, (Tables.pension_facts,), ["2025q4", "2026q1", "2026q2", "2026q3"], ["AAA", "NEW", "OLD"], cached={"2026q1", "2026q2"})

    assert work.pending == ["2026q3"]  # the newest published period, parsed for every key
    assert work.rescan_keys == ["NEW"] and work.rescan == ["2026q1", "2026q2"]  # 2025q4 is stored but not on disk
    assert dict(work.units()) == {"2026q1": ["NEW"], "2026q2": ["NEW"], "2026q3": ["AAA", "NEW", "OLD"]}
    assert "2025q4" in caplog.text
    print("\n=== SANITY CHECK: M3 new key ===")
    print(f"  units {dict(work.units())}: NEW alone re-reads the cached stored quarters; OLD (new, with a row) does not;")
    print("  the stored but uncached 2025q4 is named in a WARNING and never downloaded for a new key.")


def test_m3_new_key_window_is_seven_days_for_every_archive(tmp_path, sqlite_store):
    added = {"D6": "2026-09-24", "D8": "2026-09-22"}  # 6 and 8 days before as_of 2026-09-30
    ctx = _archive_ctx(tmp_path, sqlite_store, ["AAA", "D6", "D8"], added_on=added)
    _save_pension(sqlite_store, "AAA", ["2026q1"])
    _save_notes(sqlite_store, Tables.notes_num, "AAA", ["2026_07"])
    _save_notes(sqlite_store, Tables.notes_text, "AAA", ["2026_07"])

    pension = _archive(ctx, (Tables.pension_facts,), ["2026q1"], ["AAA", "D6", "D8"])  # 95-day table overlap
    notes = _archive(ctx, _NOTES, ["2026_07"], ["AAA", "D6", "D8"])  # 62-day table overlap

    assert pension.rescan_keys == notes.rescan_keys == ["D6"]
    print("\n=== SANITY CHECK: M3 new-key window ===")
    print("  D6 (added 6 days ago, no row) is re-parsed; D8 (8 days ago) is not, although the pension overlap is 95 days")
    print("  and the notes overlap 62: the new-key window is 7 days for every archive; -t D8 -F is the way back.")


def test_m3_zero_row_failed_and_deleted_periods_are_parsed_again(tmp_path, sqlite_store):
    ctx = _archive_ctx(tmp_path, sqlite_store, ["AAA"])
    _save_pension(sqlite_store, "AAA", ["2026q1", "2026q2"])
    published = ["2026q1", "2026q2", "2026q3"]

    night1 = _archive(ctx, (Tables.pension_facts,), published, ["AAA"])  # 2026q3: no universe row, or its ZIP failed
    night2 = _archive(ctx, (Tables.pension_facts,), published, ["AAA"])
    sqlite_store.delete(Tables.pension_facts, {"quarter": "2026q1"})
    night3 = _archive(ctx, (Tables.pension_facts,), published, ["AAA"])

    assert night1.pending == night2.pending == ["2026q3"]
    assert night3.pending == ["2026q1", "2026q3"]
    print("\n=== SANITY CHECK: M3 zero-row / failed / deleted period ===")
    print("  a period with no stored row (zero universe rows or a failed download) is listed every night (accepted);")
    print("  deleting 2026q1 lists it again.")


def test_m3_a_ticker_run_then_a_full_run_re_parses_nothing(tmp_path, sqlite_store):
    ctx = _archive_ctx(tmp_path, sqlite_store, ["AAA", "BBB"])
    _save_pension(sqlite_store, "AAA", ["2026q1"])
    _save_pension(sqlite_store, "BBB", ["2026q1"])
    published = ["2026q1", "2026q2"]

    scoped = _archive(ctx, (Tables.pension_facts,), published, ["AAA"])
    everyone = _archive(ctx, (Tables.pension_facts,), published, ["AAA", "BBB"])
    scoped_full = _archive(ctx, (Tables.pension_facts,), published, ["AAA"], full=True)
    full = _archive(ctx, (Tables.pension_facts,), published, ["AAA", "BBB"], full=True, cached=set())

    assert scoped.scoped and list(scoped.units()) == [] and scoped.held == ["2026q2"]
    assert not everyone.scoped and dict(everyone.units()) == {"2026q2": ["AAA", "BBB"]}
    assert dict(scoped_full.units()) == {"2026q1": ["AAA"]}
    assert dict(full.units()) == {"2026q1": ["AAA", "BBB"], "2026q2": ["AAA", "BBB"]}  # -F reads stored periods even when not cached
    print("\n=== SANITY CHECK: M3 -t then full ===")
    print("  -t AAA parses nothing (2026q2 parsed for AAA alone would hide it from BBB) and leaves 2026q2 to the")
    print("  full run, which parses only 2026q2; -t AAA -F re-reads the stored quarters for AAA; -F re-reads everything.")
