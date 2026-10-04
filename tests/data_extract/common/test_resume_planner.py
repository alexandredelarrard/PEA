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
