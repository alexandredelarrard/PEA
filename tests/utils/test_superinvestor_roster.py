"""
Point-in-time roster accessor (src/utils/superinvestor_roster.py).

The whole reason `superinvestor_roster` is a table and not a `{cik: name}` JSON is that a
selector must be able to ask "who was a superinvestor AS OF date D". These tests pin that
contract on the real store: the most recent snapshot at or before the date and never a
later one, the two-codes-one-manager collapse, the union used as the 13F walk scope, and the
manager identity across a filer-CIK succession (a fixture `config_dir` holds the chains).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import src.utils.superinvestor_roster as sr
from src.data_store.schema import Tables
from src.utils.superinvestor_roster import roster_as_of, roster_cik_union, roster_map_as_of

#: Appaloosa's filer succession: Appaloosa Management LP files to 2015-12-31, Appaloosa LP from 2016-03-31.
_AM_OLD, _AM_NEW = "0001006438", "0001656456"
_APPALOOSA = {_AM_OLD: [{"cik": _AM_OLD, "from": None, "to": "2015-12-31"}, {"cik": _AM_NEW, "from": "2016-03-31", "to": None}]}


def _rows(snapshot, pairs):
    """(snapshot_date, dataroma_code, manager_name, cik) rows for one snapshot."""
    return [
        {
            "snapshot_date": snapshot,
            "dataroma_code": code,
            "manager_name": name,
            "cik": cik,
            "resolution": "edgar" if cik else "unresolved",
            "source_url": "https://example.invalid/",
        }
        for code, name, cik in pairs
    ]


def _config_dir(tmp_path: Path, manager_ciks: dict | None = None) -> str:
    """A fixture `<config_dir>/superinvestors/overrides.json` holding only `manager_ciks`."""
    folder = tmp_path / "configs" / "superinvestors"
    folder.mkdir(parents=True, exist_ok=True)
    blob = {"cik_overrides": {}, "unresolvable": {}, "manager_ciks": manager_ciks or {}}
    (folder / "overrides.json").write_text(json.dumps(blob), encoding="utf-8")
    return str(tmp_path / "configs")


def _ctx(store: Any, rows: Any = None, config_dir: str | None = None) -> Any:
    if rows:
        store.replace(Tables.superinvestor_roster.name, pd.DataFrame(rows))
    return SimpleNamespace(store=store, config_dir=config_dir)


# Three snapshots. 2016 carries two managers that 2026 has dropped (the survivorship case)
# and the `brk`/`BRK` duplicate-code pair that resolves to ONE manager.
_SEED = (
    _rows(date(2013, 1, 1), [("brk", "Warren Buffett - Berkshire", "0001067983"), ("ARLINGTON", "Allan Mecham - Arlington Value", "0001568820")])
    + _rows(
        date(2016, 1, 1),
        [
            ("brk", "Warren Buffett - Berkshire", "0001067983"),
            ("BRK", "Berkshire Hathaway", "0001067983"),
            ("ARLINGTON", "Allan Mecham - Arlington Value", "0001568820"),
            ("WGRNX", "David Winters - Wintergreen", "0001360079"),
        ],
    )
    + _rows(
        date(2026, 1, 1),
        [("BRK", "Berkshire Hathaway", "0001067983"), ("GLRE", "David Einhorn - Greenlight", "0001079114"), ("CMAFX", "Century Management", None)],
    )
)


def test_roster_as_of_reads_the_snapshot_at_or_before_the_date(sqlite_store, tmp_path):
    ctx = _ctx(sqlite_store, _SEED, _config_dir(tmp_path))
    mid = roster_as_of(ctx, "2016-06-30")
    assert mid == {"0001067983", "0001568820", "0001360079"}
    # the two managers 2026 dropped ARE in the 2016 pool -- the survivorship fix
    assert {"0001568820", "0001360079"} <= mid
    assert "0001079114" not in mid  # Greenlight joined later: no look-ahead
    assert roster_as_of(ctx, "2013-06-30") == {"0001067983", "0001568820"}  # 2013 snapshot
    assert roster_as_of(ctx, "2012-12-31") == set()  # before the first
    print("\n=== SANITY: point-in-time roster ===")
    print(
        f"  as_of 2016-06-30 -> {len(mid)} CIKs including Arlington (0001568820) and "
        f"Wintergreen (0001360079), both absent from today's roster; as_of 2013-06-30 -> "
        f"{len(roster_as_of(ctx, '2013-06-30'))}; before the first snapshot -> 0. "
        "Never reads a later snapshot. Validated on the real store."
    )


def test_duplicate_codes_collapse_to_one_manager(sqlite_store, tmp_path):
    ctx = _ctx(sqlite_store, _SEED, _config_dir(tmp_path))
    snap = [r for r in _SEED if r["snapshot_date"] == date(2016, 1, 1)]
    assert sum(r["cik"] == "0001067983" for r in snap) == 2  # two rows, two codes
    assert sorted(roster_as_of(ctx, "2016-06-30")).count("0001067983") == 1
    names = roster_map_as_of(ctx, "2016-06-30")
    assert names["0001067983"] == "Berkshire Hathaway"  # first in dataroma_code order
    assert len(names) == len(roster_as_of(ctx, "2016-06-30"))
    print("\n=== SANITY: two codes, one manager ===")
    print(
        "  'brk' and 'BRK' are two rows in the 2016 snapshot and one member (0001067983) in the roster set; the name map is 1:1 with it. Validated."
    )


def test_unresolved_manager_never_becomes_a_cik(sqlite_store, tmp_path):
    ctx = _ctx(sqlite_store, _SEED, _config_dir(tmp_path))
    latest = roster_as_of(ctx)  # no date -> latest snapshot
    assert latest == {"0001067983", "0001079114"}  # CMAFX has a NULL cik
    assert None not in latest and "" not in latest
    print("\n=== SANITY: unresolved manager ===")
    print(
        f"  the 2026 snapshot has 3 rows, one with a NULL cik -> {len(latest)} CIKs; the "
        "NULL is dropped here and made loud by the WRITER, not silently turned into a key. "
        "Validated."
    )


def test_union_is_the_walk_scope_not_a_point_in_time_set(sqlite_store, tmp_path):
    greenlight = {"0001079114": [{"cik": "0001079114", "to": "2023-12-31"}, {"cik": "0001489933", "from": "2024-03-31"}]}
    ctx = _ctx(sqlite_store, _SEED, _config_dir(tmp_path, greenlight))
    union = roster_cik_union(ctx)
    # Greenlight's successor filer (DME Capital) is walked although no snapshot names it
    assert union == {"0001067983", "0001568820", "0001360079", "0001079114", "0001489933"}
    assert union > roster_as_of(ctx)  # strictly bigger than today's
    print("\n=== SANITY: 13F walk scope ===")
    print(
        f"  union over all snapshots = {len(union)} CIKs vs {len(roster_as_of(ctx))} on "
        "today's roster. Fetching only today's is what makes a dropped manager "
        "unrecoverable later. Validated."
    )


def test_missing_table_returns_empty_not_a_raise(sqlite_store, tmp_path):
    ctx = _ctx(sqlite_store, config_dir=_config_dir(tmp_path))  # table never written
    assert roster_as_of(ctx) == set() and roster_map_as_of(ctx) == {}
    assert roster_cik_union(ctx) == set()
    print("\n=== SANITY: cold start ===")
    print(
        "  an unwritten superinvestor_roster -> empty set/map/union (the caller warns and "
        "skips the elite features), never a crash mid-cube. Validated."
    )


def test_manager_identity_across_succession(sqlite_store, tmp_path):
    """One manager across a filer-CIK succession: the roster reads the manager ID (oldest CIK) on
    both sides, and the books keep each filer's rows only inside its window, under that ID."""
    rows = _rows(date(2015, 12, 1), [("AM", "David Tepper - Appaloosa", _AM_OLD)]) + _rows(
        date(2016, 3, 31), [("AM", "David Tepper - Appaloosa", _AM_NEW), ("BRK", "Berkshire Hathaway", "0001067983")]
    )
    config_dir = _config_dir(tmp_path, _APPALOOSA)
    ctx = _ctx(sqlite_store, rows, config_dir)
    assert roster_as_of(ctx, "2015-12-31") == {_AM_OLD}
    assert roster_as_of(ctx, "2016-03-31") == {_AM_OLD, "0001067983"}
    assert roster_map_as_of(ctx, "2016-03-31")[_AM_OLD] == "David Tepper - Appaloosa"
    assert sr.manager_id(_AM_NEW, config_dir) == _AM_OLD and sr.manager_id("1067983", config_dir) == "0001067983"

    books = pd.DataFrame(
        {
            "cik": [_AM_OLD, _AM_OLD, _AM_OLD, _AM_NEW, "1656456", _AM_NEW, "0001067983"],
            # Postgres DATE columns come back as datetime.date
            "period": [
                date(2015, 9, 30),
                date(2015, 12, 31),
                date(2016, 3, 31),
                date(2015, 12, 31),
                date(2016, 3, 31),
                date(2016, 6, 30),
                date(2016, 6, 30),
            ],
            "value_usd": [1, 2, 99, 98, 3, 4, 5],
        }
    )
    out = sr.to_manager_books(books, config_dir=config_dir)
    chain = out[out["cik"] == _AM_OLD].sort_values("period")
    assert list(chain["value_usd"]) == [1, 2, 3, 4]  # 99 / 98: a filer outside its own window, dropped
    assert not chain["period"].duplicated().any()  # one book per quarter, never double-counted
    assert set(out["cik"]) == {_AM_OLD, "0001067983"} and out.loc[out["cik"] == "0001067983", "value_usd"].tolist() == [5]
    print("\n=== SANITY: manager identity across a succession ===")
    print(
        f"  roster_as_of 2015-12-31 and 2016-03-31 both -> {_AM_OLD} (Appaloosa LP stored in the 2016 snapshot); books: "
        f"{len(books)} rows -> {len(out)}, the predecessor's 2016-03-31 and the successor's 2015-12-31 rows dropped, "
        "the unpadded successor CIK relabelled, Berkshire untouched. Validated on the real store."
    )


def test_union_contains_whole_chain(sqlite_store, tmp_path):
    """The walk scope holds every filer of every ever-listed manager, whichever member a snapshot stored."""
    config_dir = _config_dir(tmp_path, _APPALOOSA)
    ctx = _ctx(sqlite_store, _rows(date(2015, 12, 1), [("AM", "David Tepper - Appaloosa", _AM_OLD)]), config_dir)
    assert roster_cik_union(ctx) == {_AM_OLD, _AM_NEW}
    assert sr.filer_ciks({_AM_OLD, "1067983"}, config_dir) == {_AM_OLD, _AM_NEW, "0001067983"}
    assert sr.filer_ciks({_AM_NEW}, config_dir) == {_AM_OLD, _AM_NEW}  # a member expands to its whole chain
    print("\n=== SANITY: walk scope expands chains ===")
    print(f"  a roster naming only {_AM_OLD} walks {sorted(roster_cik_union(ctx))}; a singleton stays itself. Validated.")


@pytest.mark.parametrize(
    ("manager_ciks", "match"),
    [
        ({_AM_OLD: [{"cik": _AM_OLD, "to": "2016-03-31"}, {"cik": _AM_NEW, "from": "2016-03-31"}]}, "overlap"),
        ({_AM_OLD: [{"cik": _AM_OLD}, {"cik": _AM_NEW, "from": "2016-03-31"}]}, "overlap"),
        ({_AM_NEW: [{"cik": _AM_OLD, "to": "2015-12-31"}, {"cik": _AM_NEW, "from": "2016-03-31"}]}, "oldest"),
        ({_AM_OLD: [{"cik": _AM_NEW, "from": "2016-03-31"}, {"cik": _AM_OLD, "to": "2015-12-31"}]}, "oldest"),
        (
            {
                _AM_OLD: [{"cik": _AM_OLD, "to": "2015-12-31"}, {"cik": _AM_NEW, "from": "2016-03-31"}],
                "0000000001": [{"cik": "0000000001", "to": "2010-12-31"}, {"cik": _AM_NEW, "from": "2011-03-31"}],
            },
            "more than one chain",
        ),
    ],
    ids=["shared-quarter", "open-predecessor", "id-not-oldest", "not-chronological", "cik-in-two-chains"],
)
def test_loader_rejects_overlapping_windows(tmp_path, manager_ciks, match):
    """A malformed chain must fail at load, never silently double-count or drop a quarter."""
    with pytest.raises(ValueError, match=match):
        sr.load_superinvestor_overrides(_config_dir(tmp_path, manager_ciks))
    print(f"\n=== SANITY: chain validation ({match}) ===\n  the loader raised on a malformed chain. Validated.")
