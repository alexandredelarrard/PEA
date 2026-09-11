"""
Point-in-time roster accessor (src/utils/superinvestor_roster.py).

The whole reason `superinvestor_roster` is a table and not a `{cik: name}` JSON is that a
selector must be able to ask "who was a superinvestor AS OF date D". These tests pin that
contract on the real store: the most recent snapshot at or before the date and never a
later one, the two-codes-one-manager collapse, and the union used as the 13F walk scope.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from src.data_store.schema import Tables
from src.utils.superinvestor_roster import (
    roster_as_of, roster_cik_union, roster_map_as_of)


def _rows(snapshot, pairs):
    """(snapshot_date, dataroma_code, manager_name, cik) rows for one snapshot."""
    return [{"snapshot_date": snapshot, "dataroma_code": code, "manager_name": name,
             "cik": cik, "resolution": "edgar" if cik else "unresolved",
             "source_url": "https://example.invalid/"} for code, name, cik in pairs]


def _ctx(store, rows=None):
    if rows:
        store.replace(Tables.superinvestor_roster.name, pd.DataFrame(rows))
    return SimpleNamespace(store=store)


# Three snapshots. 2016 carries two managers that 2026 has dropped (the survivorship case)
# and the `brk`/`BRK` duplicate-code pair that resolves to ONE manager.
_SEED = (
    _rows(date(2013, 1, 1), [("brk", "Warren Buffett - Berkshire", "0001067983"),
                             ("ARLINGTON", "Allan Mecham - Arlington Value", "0001568820")])
    + _rows(date(2016, 1, 1), [("brk", "Warren Buffett - Berkshire", "0001067983"),
                               ("BRK", "Berkshire Hathaway", "0001067983"),
                               ("ARLINGTON", "Allan Mecham - Arlington Value", "0001568820"),
                               ("WGRNX", "David Winters - Wintergreen", "0001360079")])
    + _rows(date(2026, 1, 1), [("BRK", "Berkshire Hathaway", "0001067983"),
                               ("GLRE", "David Einhorn - Greenlight", "0001079114"),
                               ("CMAFX", "Century Management", None)])
)


def test_roster_as_of_reads_the_snapshot_at_or_before_the_date(sqlite_store):
    ctx = _ctx(sqlite_store, _SEED)
    mid = roster_as_of(ctx, "2016-06-30")
    assert mid == {"0001067983", "0001568820", "0001360079"}
    # the two managers 2026 dropped ARE in the 2016 pool -- the survivorship fix
    assert {"0001568820", "0001360079"} <= mid
    assert "0001079114" not in mid                 # Greenlight joined later: no look-ahead
    assert roster_as_of(ctx, "2013-06-30") == {"0001067983", "0001568820"}   # 2013 snapshot
    assert roster_as_of(ctx, "2012-12-31") == set()                         # before the first
    print("\n=== SANITY: point-in-time roster ===")
    print(f"  as_of 2016-06-30 -> {len(mid)} CIKs including Arlington (0001568820) and "
          f"Wintergreen (0001360079), both absent from today's roster; as_of 2013-06-30 -> "
          f"{len(roster_as_of(ctx, '2013-06-30'))}; before the first snapshot -> 0. "
          "Never reads a later snapshot. Validated on the real store.")


def test_duplicate_codes_collapse_to_one_manager(sqlite_store):
    ctx = _ctx(sqlite_store, _SEED)
    snap = [r for r in _SEED if r["snapshot_date"] == date(2016, 1, 1)]
    assert sum(r["cik"] == "0001067983" for r in snap) == 2      # two rows, two codes
    assert sorted(roster_as_of(ctx, "2016-06-30")).count("0001067983") == 1
    names = roster_map_as_of(ctx, "2016-06-30")
    assert names["0001067983"] == "Berkshire Hathaway"            # first in dataroma_code order
    assert len(names) == len(roster_as_of(ctx, "2016-06-30"))
    print("\n=== SANITY: two codes, one manager ===")
    print("  'brk' and 'BRK' are two rows in the 2016 snapshot and one member "
          "(0001067983) in the roster set; the name map is 1:1 with it. Validated.")


def test_unresolved_manager_never_becomes_a_cik(sqlite_store):
    ctx = _ctx(sqlite_store, _SEED)
    latest = roster_as_of(ctx)                                    # no date -> latest snapshot
    assert latest == {"0001067983", "0001079114"}                 # CMAFX has a NULL cik
    assert None not in latest and "" not in latest
    print("\n=== SANITY: unresolved manager ===")
    print(f"  the 2026 snapshot has 3 rows, one with a NULL cik -> {len(latest)} CIKs; the "
          "NULL is dropped here and made loud by the WRITER, not silently turned into a key. "
          "Validated.")


def test_union_is_the_walk_scope_not_a_point_in_time_set(sqlite_store):
    ctx = _ctx(sqlite_store, _SEED)
    union = roster_cik_union(ctx)
    assert union == {"0001067983", "0001568820", "0001360079", "0001079114"}
    assert union > roster_as_of(ctx)                              # strictly bigger than today's
    print("\n=== SANITY: 13F walk scope ===")
    print(f"  union over all snapshots = {len(union)} CIKs vs {len(roster_as_of(ctx))} on "
          "today's roster. Fetching only today's is what makes a dropped manager "
          "unrecoverable later. Validated.")


def test_missing_table_returns_empty_not_a_raise(sqlite_store):
    ctx = _ctx(sqlite_store)                                      # table never written
    assert roster_as_of(ctx) == set() and roster_map_as_of(ctx) == {}
    assert roster_cik_union(ctx) == set()
    print("\n=== SANITY: cold start ===")
    print("  an unwritten superinvestor_roster -> empty set/map/union (the caller warns and "
          "skips the elite features), never a crash mid-cube. Validated.")
