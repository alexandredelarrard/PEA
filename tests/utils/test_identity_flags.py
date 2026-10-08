"""Known-truth cases for the tape-mix rule of `identity_flags` read against `security_master` lines."""

from __future__ import annotations

import pandas as pd

from src.utils.identity_flags import identity_flags

_LINEAGE = ("entity_id", "canonical_ticker", "cik", "role", "symbol", "valid_from", "valid_to", "status", "sources", "oracle", "evidence")
_MASTER = ("canonical_company", "source", "source_symbol", "cusip", "lineage_role", "lineage_reason", "valid_from", "valid_to")
_ACTIVITY = pd.DataFrame(columns=["cik", "source", "first", "last"])


def _window(ticker: str, cik: str, start: str, end: str | None, role: str = "cik_window", sources: str = "roster") -> tuple:
    oracle = "roster" if sources == "roster" else "manual"
    return (f"E-{ticker}", ticker, cik, role, None, start, end, "ok", sources, oracle, "")


def _symbol(ticker: str, cik: str, symbol: str, start: str, end: str | None, sources: str = "form345") -> tuple:
    return (f"E-{ticker}", ticker, cik, "symbol", symbol, start, end, "ok", sources, "", "")


def _lineage() -> pd.DataFrame:
    rows = [
        # acquired constituent: COH's IIV line, concurrent on a windowed CIK, covered by master lines
        _window("COH", "0000000100", "1900-01-01", None),
        _window("COH", "0000000101", "1900-01-01", "2022-09-06", sources="manual"),
        _symbol("COH", "0000000100", "COH", "2010-01-01", None),
        _symbol("COH", "0000000101", "IIV", "2008-01-01", "2022-09-06"),
        # declared secondary class on the home CIK, master spells it without the separator
        _window("BRK", "0000000200", "1900-01-01", None),
        _symbol("BRK", "0000000200", "BRK", "2005-01-01", None),
        _symbol("BRK", "0000000200", "BRK-A", "2005-01-01", None),
        # overlap entirely before the tape era (master starts 2009-06-26)
        _window("OLD", "0000000300", "1900-01-01", None),
        _window("OLD", "0000000301", "1900-01-01", "2009-06-01", sources="manual"),
        _symbol("OLD", "0000000300", "OLD", "2000-01-01", None),
        _symbol("OLD", "0000000301", "OLX", "2000-01-01", "2009-05-01"),
        # uncovered overlap on a windowed CIK: still an action
        _window("UNC", "0000000400", "1900-01-01", None),
        _window("UNC", "0000000401", "1900-01-01", "2016-01-01", sources="manual"),
        _symbol("UNC", "0000000400", "UNC", "2010-01-01", None),
        _symbol("UNC", "0000000401", "UNX", "2012-01-01", "2015-01-01"),
        # partly covered: master covers UNY only until 2013, the rest is concurrent on a windowed CIK
        _symbol("UNC", "0000000401", "UNY", "2012-01-01", "2015-01-01"),
        # overlap on an event-only CIK (no window at those dates): not used for tape resolution
        _window("NOW", "0000000500", "1900-01-01", None),
        _window("NOW", "0000000501", "1900-01-01", None, role="cik_event", sources="manual"),
        _symbol("NOW", "0000000500", "NOW", "2010-01-01", None),
        _symbol("NOW", "0000000501", "NWX", "2012-01-01", "2015-01-01"),
        # a master line excluded as `unclassified` is a default, not a decision: still an action
        _symbol("UNC", "0000000401", "UNZ", "2012-01-01", "2015-01-01"),
        # redundant class while the ticker trades on cover pages only, covered by a secondary-class line
        _window("RED", "0000000600", "1900-01-01", None),
        _symbol("RED", "0000000600", "RED", "2012-01-01", "2016-01-01", sources="dei"),
        _symbol("RED", "0000000600", "REX", "2012-01-01", None),
    ]
    return pd.DataFrame(rows, columns=list(_LINEAGE))


def _master() -> pd.DataFrame:
    rows = [
        ("COH", "ftd", "IIV", "C1", "canonical_current", "tape_symbol", "2009-06-26", "2022-09-06"),
        ("COH", "ftd", "COH", "C2", "acquired_constituent", "event_only_cik", "2010-01-01", "2022-09-06"),
        ("BRK", "ftd", "BRKA", "B1", "secondary_class", "class_description", "2009-06-26", None),
        ("UNC", "ftd", "UNY", "U1", "excluded", "manual_boundary", "2012-01-01", "2013-01-01"),
        ("UNC", "ftd", "UNZ", "U2", "excluded", "unclassified", "2009-06-26", None),
        ("RED", "ftd", "REX", "R1", "secondary_class", "class_description", "2009-06-26", None),
    ]
    frame = pd.DataFrame(rows, columns=list(_MASTER))
    frame["valid_from"] = [pd.Timestamp(value).date() for value in frame["valid_from"]]  # Postgres DATE round trip
    frame["valid_to"] = [None if value is None else pd.Timestamp(value).date() for value in frame["valid_to"]]
    return frame


def _tape_mix(flags: pd.DataFrame) -> dict[str, pd.Series]:
    part = flags[flags["kind"].eq("manual_review") & flags["evidence"].str.contains(" resolves to ")]
    return {str(row["evidence"]).split(" ", 1)[0]: row for _, row in part.iterrows()}


def test_master_decides_tape_mix_items_outside_the_tape_era_covered_or_unwindowed() -> None:
    lineage = _lineage()
    redundant = frozenset({"REX"})
    without = _tape_mix(identity_flags(lineage, _ACTIVITY, redundant_symbols=redundant))
    items = _tape_mix(identity_flags(lineage, _ACTIVITY, redundant_symbols=redundant, master=_master()))

    assert set(without) == {"IIV", "BRK-A", "OLX", "UNX", "UNY", "NWX", "UNZ", "REX"}, sorted(without)
    assert all(bool(row["action"]) for row in without.values()), "without a master every tape-mix pair is an action"
    assert set(items) == set(without)
    assert not items["IIV"]["action"] and "decided by security_master (canonical_current)" in items["IIV"]["evidence"]
    assert not items["BRK-A"]["action"] and "decided by security_master (secondary_class)" in items["BRK-A"]["evidence"]
    assert not items["OLX"]["action"] and "before the tape era" in items["OLX"]["evidence"]
    assert not items["NWX"]["action"] and "no CIK window" in items["NWX"]["evidence"]
    assert items["UNX"]["action"] and items["UNX"]["config_file"] == "configs/sec/symbol_tenure_manual.json"
    assert items["UNY"]["action"] and "2013-01-01..2015-01-01" in items["UNY"]["evidence"], items["UNY"]["evidence"]
    assert items["UNZ"]["action"] and "not covered by security_master: 2012-01-01..2015-01-01" in items["UNZ"]["evidence"]
    assert not items["REX"]["action"] and "decided by security_master (secondary_class)" in items["REX"]["evidence"]

    print("\n=== SANITY CHECK: tape-mix items read against security_master ===")
    for symbol, row in sorted(items.items()):
        print(f"  {symbol:6} {'ACTION' if row['action'] else 'info  '} {row['evidence']}")
    print("  OK: acquired constituent, declared and redundant classes, pre-tape-era and event-only-CIK overlaps are info;")
    print("      uncovered and merely-unclassified overlaps stay actions")


def test_without_a_master_the_flags_are_unchanged() -> None:
    lineage = _lineage()
    before = identity_flags(lineage, _ACTIVITY)
    explicit = identity_flags(lineage, _ACTIVITY, master=None)
    empty = identity_flags(lineage, _ACTIVITY, master=_master().iloc[0:0])

    pd.testing.assert_frame_equal(before, explicit)
    pd.testing.assert_frame_equal(before, empty)

    print("\n=== SANITY CHECK: no master ===")
    print(f"  {len(before)} flag(s), {int(before['action'].sum())} action(s), identical with master=None and with an empty master")
    print("  OK: the rule without security_master is byte-identical")
