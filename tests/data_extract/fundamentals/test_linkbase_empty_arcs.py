"""
The four empty-frame early returns of `calculation_arcs` / `statement_arcs`.

`statement_arcs`' "no arc survived the filter" branch returned `pd.DataFrame(columns=cols)`
against an undefined `cols`, so a filing whose whole calculation linkbase sits in the notes
raised `NameError` instead of returning the empty frame that routes a field to
`tag_fallback`. `filing_rows`' bare `except Exception` then reported it as "unreadable XBRL",
so **NEM, MO and AIZ each produced zero facts while the run reported success**.

The shape of the returned frame is the contract that matters: every caller indexes
`ARC_COLUMNS` on the result, so an empty frame of the wrong shape is as fatal as an
exception, only later and quieter.

Synthetic: whether a branch returns the right SHAPE is a known-truth question about the
code, not an economic one (docs/testing.md's parsing exception).
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    ARC_COLUMNS, calculation_arcs, statement_arcs,
)

#: A role the filer declares in the NOTES, so `NON_STATEMENT_ROLE` ("detail") rejects it,
#: paired with a `menucat` that is not `"Statements"`. Both tests of the union say no ->
#: `keep.any()` is False, which is the branch that carried the `NameError`.
_NOTE_ROLE = "http://x/role/DisclosureIncomeTaxesDetails"


class _FakeXbrl:
    """`calculation_arcs` only ever calls `calculation_linkbase()`."""

    def __init__(self, arcs) -> None:
        self._arcs = arcs

    def calculation_linkbase(self):
        if isinstance(self._arcs, Exception):
            raise self._arcs
        return self._arcs


def _arcs(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """(role_uri, menucat) -> the RAW calculation-linkbase frame."""
    return pd.DataFrame(
        [{"concept": "IncomeTaxExpenseBenefit", "concept_taxonomy": "us-gaap",
          "parent_concept": "NetIncomeLoss", "parent_taxonomy": "us-gaap",
          "weight": 1.0, "role_uri": role, "menucat": menucat,
          "is_abstract": False, "arc_filter": "both"}
         for role, menucat in rows],
        columns=ARC_COLUMNS)


def _assert_empty_arc_frame(frame: pd.DataFrame, label: str) -> None:
    assert isinstance(frame, pd.DataFrame), f"{label}: not a frame"
    assert frame.empty, f"{label}: expected no rows, got {len(frame)}"
    assert list(frame.columns) == ARC_COLUMNS, (
        f"{label}: shape contract broken: {list(frame.columns)}")


def test_statement_arcs_returns_empty_frame_when_no_arc_is_a_statement_arc():
    """The `NameError` branch: arcs exist, and every one of them is note-only."""
    filing = _FakeXbrl(_arcs([(_NOTE_ROLE, "Notes"), (_NOTE_ROLE, None)]))

    raw = calculation_arcs(filing)
    out = statement_arcs(filing)

    print("\n=== SANITY CHECK: statement_arcs, every arc note-only (the `cols` NameError) ===")
    print(f"  raw linkbase: {len(raw)} arc(s) on role .../DisclosureIncomeTaxesDetails")
    print(f"  statement_arcs: {len(out)} row(s), columns={list(out.columns)}")
    _assert_empty_arc_frame(out, "no arc survived the filter")
    assert len(raw) == 2, "calculation_arcs must keep the UNFILTERED arcs"
    print("  -> Returned the empty ARC_COLUMNS frame instead of raising; the filing routes "
          "to tag_fallback.")


def test_the_other_three_empty_returns_share_the_same_shape():
    """`calculation_arcs`' absent-linkbase and empty-frame returns, and `statement_arcs`'
    empty-input return. All three were already correct -- pinned so the shape stays one
    contract across the four sites."""
    cases = {
        "linkbase raises": _FakeXbrl(ValueError("no calculation linkbase in this filing")),
        "linkbase is None": _FakeXbrl(None),
        "linkbase is empty": _FakeXbrl(pd.DataFrame(columns=ARC_COLUMNS)),
    }

    print("\n=== SANITY CHECK: the other three empty-frame returns ===")
    for label, filing in cases.items():
        raw, out = calculation_arcs(filing), statement_arcs(filing)
        _assert_empty_arc_frame(raw, f"calculation_arcs, {label}")
        _assert_empty_arc_frame(out, f"statement_arcs, {label}")
        print(f"  {label}: calculation_arcs {len(raw)} row(s), "
              f"statement_arcs {len(out)} row(s), both ARC_COLUMNS")
    print("  -> All 4 empty returns now agree on one shape.")
