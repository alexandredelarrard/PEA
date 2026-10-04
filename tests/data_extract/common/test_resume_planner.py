"""`resume.document_worklist` (M2, EDGAR documents) on the real `DataStore` (SQLite) and a seeded local
index: the eight AC-002 cases, the per-key done set of role-marker tables and the cap. Each case
plans from the table's own rows, its `schema.Resume` contract and the run date only."""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.common.edgar_driver import FilingStamp, marker_row
from src.data_extract.utils.common.resume import DONE_PER_KEY, KEY_ESTABLISHED, KEY_NEW, KEY_ROWLESS, document_worklist
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
    defaults = {"forms": _FORMS, "registrants": {}, "identity": None, "years_history": 5}
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
