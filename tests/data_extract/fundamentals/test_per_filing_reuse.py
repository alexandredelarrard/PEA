"""Work that depends only on the filing, or only on the field, happens ONCE.

Three call-count pins, each guarding a specific reuse the resolver now depends on. They are
call-count tests, not wall-clock ones -- deliberately, because a wall clock on a synthetic
linkbase measures nothing, while a second `calculation_linkbase()` read is a fact about the
code that a later edit can silently reintroduce.

`edgar.xbrl.XBRL.calculation_linkbase` carries no cache of its own, which is what makes the
first of these worth pinning rather than trusting.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals import xbrl_linkbase as xl
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue

CATALOGUE = load_catalogue()

#: The three fields whose catalogue entry declares a `roll_up.any_of` -- the only ones route
#: 3b can ever apply to. Named here so gating the route on that declaration is a decision
#: this test re-checks rather than an assumption.
ROUTE_3B_FIELDS = {"capex", "costOfRevenue", "depAmort"}


class _CountingXbrl:
    """Just enough of an `XBRL` to serve a linkbase, and to say how often it was asked."""

    def __init__(self, arcs: pd.DataFrame):
        self._arcs = arcs
        self.reads = 0

    def calculation_linkbase(self) -> pd.DataFrame:
        self.reads += 1
        return self._arcs


def _linkbase() -> pd.DataFrame:
    return pd.DataFrame([
        {"concept": "Revenues", "concept_taxonomy": "us-gaap",
         "parent_concept": "Parent", "parent_taxonomy": "us-gaap", "weight": 1.0,
         "role_uri": "http://x/role/StatementOfIncome", "menucat": "Statements",
         "is_abstract": False}])


def test_one_calculation_linkbase_read_per_filing():
    """The fetch path needs BOTH views of the linkbase -- unfiltered for
    `segment_only_concepts`, face-statement-only for the graph -- and used to parse it twice
    to get them. `statement_arcs` now accepts the frame the caller already holds."""
    doubled = _CountingXbrl(_linkbase())
    xl.calculation_arcs(doubled)
    xl.statement_arcs(doubled)                       # the old shape: reads it again

    once = _CountingXbrl(_linkbase())
    arcs = xl.calculation_arcs(once)
    statements = xl.statement_arcs(once, arcs)       # the shape `rows_from_xbrl` uses

    assert doubled.reads == 2, "the two-view fixture no longer reproduces the old shape"
    assert once.reads == 1, f"the linkbase was parsed {once.reads} times for one filing"
    assert not statements.empty, "the handed-in frame was not filtered"

    print("\n=== SANITY CHECK: calculation_linkbase reads per filing ===")
    print(f"  both views, arcs re-read:  {doubled.reads} parse(s)")
    print(f"  both views, arcs passed:   {once.reads} parse(s)")


def test_the_candidate_list_is_built_once_per_field(monkeypatch):
    """`_candidates(spec, regime)` is a property of the field and the regime alone, and
    `resolve_field` asked for it again in every pass and inside route 3b."""
    calls: list[str] = []
    original = xl._candidates
    monkeypatch.setattr(xl, "_candidates",
                        lambda spec, regime: (calls.append(spec.name),
                                              original(spec, regime))[1])

    graph = xl.ArcGraph(xl.statement_arcs(_CountingXbrl(_linkbase())))
    xl.resolve_field(CATALOGUE.field("totalRevenue"), graph, frozenset({"Revenues"}),
                     CATALOGUE, regime=None)

    assert len(calls) == 1, f"the candidate list was rebuilt {len(calls)} times"

    print("\n=== SANITY CHECK: _candidates calls per (filing, field) ===")
    print(f"  one resolve_field(totalRevenue) -> {len(calls)} build(s)")


def test_route_3b_is_entered_only_by_the_fields_that_declare_it():
    """`_leaf_sum`'s prologue used to run for all 48 extracted fields to discover that 45 of
    them declare no `roll_up.any_of`. Gating it is only safe if the three that DO declare one
    still enter -- so assert the membership, not just the count."""
    declared = {name for name in CATALOGUE.extracted_fields
                if xl._roll_up(CATALOGUE.field(name), None).get("any_of")}
    assert declared == ROUTE_3B_FIELDS, (
        f"route 3b's population moved: {sorted(declared)}")

    by_regime = {regime: sorted(
        name for name in CATALOGUE.extracted_fields
        if xl._roll_up(CATALOGUE.field(name), regime).get("any_of"))
        for regime in CATALOGUE.regime_names}
    for regime, names in by_regime.items():
        assert ROUTE_3B_FIELDS <= set(names), (
            f"regime {regime!r} would skip {sorted(ROUTE_3B_FIELDS - set(names))}")

    print("\n=== SANITY CHECK: fields route 3b applies to ===")
    print(f"  no regime: {sorted(declared)} of {len(CATALOGUE.extracted_fields)} extracted")
    for regime, names in sorted(by_regime.items()):
        print(f"  {regime:12} {names}")
    print("  The other fields return on `_leaf_sum`'s first statement. Validated.")
