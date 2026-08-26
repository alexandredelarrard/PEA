"""
Cluster `876ab8a57bd8` -- ORCL `grossProfit`, 2 agreeing checks, TWO stacked defects.

`series_shape` (tier 2, high) called it `late_start`: 9 of 48 grid periods. `coverage_field`
(tier 1, medium) called it MIXED: 8 of 60 periods. Both were right, and both UNDER-reported,
because the periods that DID resolve were wrong too.

## Defect 1 -- a segment-note margin stored as consolidated gross profit

Oracle presents **no gross-profit and no total-cost-of-revenue subtotal** on the face of its
income statement; its three cost lines sit inside "Operating expenses" as company extensions
(`orcl:CloudAndSoftwareExpenses`, `orcl:HardwareExpenses`, `orcl:ServicesExpense`). The only
`us-gaap:GrossProfit` it tags lives in the SEGMENT INFORMATION note, labelled "Margin" -- the
total margin for its three reportable segments -- and Oracle publishes that total
UNDIMENSIONED, so it survives `entity_scope.consolidated_facts`.

Ground truth read off the filed statements, not `companyfacts`:
  * 0001193125-18-201034 (FY2018 10-K), `DisclosureSEGMENTINFORMATIONDetails` --
    21,825 + 1,807 + 655 = **24,287**, which is exactly what this pipeline stored.
    FY2017 20,618 + 1,709 + 690 = 23,017 and FY2016 19,882 + 1,771 + 757 = 22,410 match too.
  * 0001193125-26-277521 (FY2026 10-K) -- no `GrossProfit` fact and no `GrossProfit` arc
    anywhere, which is why the field correctly reads `not_disclosed` there.

`is_note_only` was meant to catch exactly this and could not: `ArcGraph` is built from
`statement_arcs`, which has ALREADY dropped every non-statement arc, so `roles_of` returns
an empty set and the guard's "silence is not evidence" rule answers False. The guard could
only ever fire on a note arc admitted by the `menucat == "Statements"` mis-categorisation.
ORCL's sole `GrossProfit` arc sits on `DisclosureSEGMENTINFORMATIONRECONCILIATIONDetails`
with `menucat == "Details"`, so it was dropped and the concept walked straight into
`tag_primary`.

The fix is `segment_only_concepts`, read off the UNFILTERED linkbase, and it is deliberately
NARROWER than "note-only". Measured on a 25-filing sample of the 54-ticker roster: of 192
resolved rows whose concept is note-only in the full linkbase, **3** sit on a segment role.
The other 189 are `WeightedAverageNumberOfSharesOutstandingBasic`, `ShareBasedCompensation`
and `CashCashEquivalentsRestrictedCash...` -- right values on arcs that genuinely live in
notes. Banning note-only concepts outright would have nulled all 192, which is this repo's
745-correct-rows-nulled precedent repeating itself. Measured before/after on 5 ORCL filings
plus CAT / JPM / CVS / COST: **16 cells changed, all ORCL grossProfit; 874 unchanged.**

## Defect 2 -- an eight-year forward-fill

`build_history._latest` returned the newest TTM row that had EVER been computed, with no
freshness bound. Once ORCL stopped tagging the segment margin the cell simply froze:
**24,238,000,000 from 2018-11-30 to 2026-05-31, 32 consecutive rows**, while `grossMargins`
divided it by a growing revenue and manufactured a collapse from 0.609 to 0.360. Measured
across the roster before the cap: **27 (ticker, field) pairs frozen 5+ years**, 49 for 2+ --
BRK-B `operatingIncome` 54 of 57 rows, XOM `dilutedShares` 51 of 51.

Both halves are synthetic known-truth here (docs/testing.md: parsing math gets fixtures);
the real-filing evidence is the measurement recorded above and in the two docstrings.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals import reason_codes as rc
from src.data_extract.utils.fundamentals.build_history import (
    TTM_STALENESS_DAYS, _is_stale)
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    SEGMENT_ONLY_CONCEPT, TAG_PRIMARY, ArcGraph, is_note_only, resolve_field,
    segment_only_concepts, statement_arcs)

CATALOGUE = load_catalogue("./configs")

#: ORCL's own role, verbatim from 0001193125-18-201034. Verbatim because the guard is a
#: naming heuristic over filer-authored strings: a placeholder would test the regex against
#: a string we invented rather than against the one that actually shipped.
_SEGMENT_ROLE = ("http://www.oracle.com/20180531/taxonomy/role/"
                 "DisclosureSEGMENTINFORMATIONRECONCILIATIONDetails")
_INCOME_ROLE = "http://x/role/ConsolidatedStatementsOfIncome"

_ARC_COLS = ["concept", "concept_taxonomy", "parent_concept", "parent_taxonomy",
             "weight", "role_uri", "menucat", "is_abstract", "arc_filter"]


class _FakeXbrl:
    """`statement_arcs` and `calculation_arcs` only ever call `calculation_linkbase()`."""

    def __init__(self, arcs: pd.DataFrame) -> None:
        self._arcs = arcs

    def calculation_linkbase(self) -> pd.DataFrame:
        return self._arcs


def _arcs(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """(concept, parent, role, menucat) -> the RAW calculation-linkbase frame."""
    return pd.DataFrame(
        [{"concept": c, "concept_taxonomy": "us-gaap", "parent_concept": p,
          "parent_taxonomy": "us-gaap", "weight": 1.0, "role_uri": role,
          "menucat": menucat, "is_abstract": False, "arc_filter": "both"}
         for c, p, role, menucat in rows],
        columns=_ARC_COLS)


#: ORCL's shape: `GrossProfit` declared ONLY under the segment-note reconciliation, with a
#: perfectly ordinary income statement beside it that never mentions it.
_ORCL_ARCS = _arcs([
    ("GrossProfit", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
                    "ExtraordinaryItemsNoncontrollingInterest", _SEGMENT_ROLE, "Details"),
    ("CostsAndExpenses", "OperatingIncomeLoss", _INCOME_ROLE, "Statements"),
])


def _resolve(arcs: pd.DataFrame, available: set[str], segment_only: frozenset[str]):
    """`grossProfit` resolved against a synthetic filing reporting `available`."""
    return resolve_field(
        CATALOGUE.field("grossProfit"), ArcGraph(statement_arcs(_FakeXbrl(arcs))),
        frozenset(available), CATALOGUE, regime="industrial", ticker="ORCL",
        segment_only=segment_only)


# --------------------------------------------------------------------------- #
# Defect 1: the segment-note margin                                            #
# --------------------------------------------------------------------------- #
def test_the_note_guard_is_blind_to_an_arc_statement_arcs_already_dropped():
    """The DEFECT, pinned. `is_note_only` cannot see what never reached the graph."""
    graph = ArcGraph(statement_arcs(_FakeXbrl(_ORCL_ARCS)))
    assert not graph.knows("GrossProfit")
    assert graph.roles_of("GrossProfit") == frozenset()
    assert is_note_only(graph, "GrossProfit") is False
    print("\n=== SANITY CHECK: why the existing guard missed ORCL ===")
    print("  GrossProfit's only arc is on a *Details role -> statement_arcs drops it ->")
    print("  roles_of() is empty -> is_note_only() answers False ('silence is not")
    print("  evidence') -> the concept walks into tag_primary. Confirmed.")


def test_a_segment_only_concept_is_found_on_the_unfiltered_linkbase():
    """The fix reads the arcs `statement_arcs` threw away, and only those roles decide."""
    found = segment_only_concepts(_ORCL_ARCS)
    assert "GrossProfit" in found
    assert "CostsAndExpenses" not in found
    print("\n=== SANITY CHECK: segment_only_concepts ===")
    print(f"  GrossProfit withheld: {'GrossProfit' in found}; the income-statement node "
          f"CostsAndExpenses withheld: {'CostsAndExpenses' in found}")
    print("  Read off the UNFILTERED frame, so the dropped arc is still evidence.")
    print("  Validated.")


def test_the_parent_a_segment_note_reconciles_to_is_never_withheld():
    """The CHILD of a reconciliation arc is the segment aggregate; the PARENT is the
    consolidated line it reconciles TO.

    Reading both ends of the arc -- which is what `ArcGraph._all_roles` does, correctly,
    for a different question -- condemns the wrong one. Measured before this was narrowed:
    ORCL lost `pretaxIncome` for 2017-08-31 ($2,500M) and 2018-08-31 ($2,540M), because in
    10-Q 0001564590-18-023315 its real consolidated pretax element carries no arc except as
    the parent of the segment note's `GrossProfit`.
    """
    parent = ("IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
              "ExtraordinaryItemsNoncontrollingInterest")
    found = segment_only_concepts(_ORCL_ARCS)
    assert parent not in found
    assert "GrossProfit" in found
    print("\n=== SANITY CHECK: only the child end of a reconciliation arc is withheld ===")
    print(f"  child  GrossProfit withheld: {'GrossProfit' in found}")
    print(f"  parent {parent[:40]}... withheld: {parent in found}")
    print("  ORCL's pretaxIncome survives. Validated.")


def test_orcl_gross_profit_resolves_to_a_reason_coded_null_not_a_segment_margin():
    """The whole cluster in one assertion: 24,287,000,000 must NOT become gross profit."""
    before = _resolve(_ORCL_ARCS, {"GrossProfit", "CostsAndExpenses"}, frozenset())
    assert before.method == TAG_PRIMARY and before.concept == "us-gaap:GrossProfit"

    after = _resolve(_ORCL_ARCS, {"GrossProfit", "CostsAndExpenses"},
                     segment_only_concepts(_ORCL_ARCS))
    assert not after.resolved
    assert after.dc_code == SEGMENT_ONLY_CONCEPT
    assert after.segment_rejected == ("GrossProfit",)
    print("\n=== SANITY CHECK: cluster 876ab8a57bd8 ===")
    print(f"  before: {before.method} on {before.concept} -> FY2018 stored")
    print("          24,287,000,000, the sum of three SEGMENT margins")
    print(f"  after:  unresolved, dc_code={after.dc_code}, "
          f"segment_rejected={list(after.segment_rejected)}")
    print("  Oracle presents no gross-profit subtotal, so a reason-coded NULL is the")
    print("  correct answer. Validated.")


def test_the_refusal_is_not_relaxable_unlike_the_note_guard():
    """A segment aggregate is a DIFFERENT MEASURE, not a narrower basis.

    `resolve_field` relaxes the note-role guard when it is the only thing standing between
    the field and a null -- a narrow real number beats nothing. That reasoning does not
    transfer: relaxing to a segment total silently swaps the measure. Verified live before
    the fix went in, where the guard alone left the value in place and merely stamped
    `role_only_retained: true` on it.
    """
    after = _resolve(_ORCL_ARCS, {"GrossProfit"}, frozenset({"GrossProfit"}))
    assert not after.resolved and not after.role_only_retained
    print("\n=== SANITY CHECK: no relaxation pass restores it ===")
    print(f"  resolved={after.resolved}, role_only_retained={after.role_only_retained}, "
          f"dc_code={after.dc_code}. Validated.")


def test_a_concept_on_a_face_statement_is_never_segment_only():
    """The guard must not touch a filer that reports a real gross profit.

    The measured cost of getting this wrong is 189 correct rows: share counts,
    `ShareBasedCompensation` and restricted-cash totals all carry note-only arcs.
    """
    arcs = _arcs([
        ("GrossProfit", "OperatingIncomeLoss", _INCOME_ROLE, "Statements"),
        ("GrossProfit", "SegmentReportingInformationLineItems", _SEGMENT_ROLE, "Details"),
    ])
    found = segment_only_concepts(arcs)
    assert "GrossProfit" not in found
    resolution = _resolve(arcs, {"GrossProfit"}, found)
    assert resolution.resolved and resolution.concept == "us-gaap:GrossProfit"
    print("\n=== SANITY CHECK: one face-statement arc is enough to keep a concept ===")
    print(f"  declared on BOTH an income-statement and a segment role -> "
          f"GrossProfit withheld: {'GrossProfit' in found}, "
          f"resolved={resolution.resolved}. Validated.")


def test_silence_is_still_not_evidence():
    """A concept with NO arc at all is not condemned for having no roles.

    The same rule `is_note_only` documents, and load-bearing for the same reason: a `dei:`
    cover-page tag or a leaf like `goodwill` can never carry a calculation arc.
    """
    assert "Goodwill" not in segment_only_concepts(_ORCL_ARCS)
    assert segment_only_concepts(pd.DataFrame(columns=_ARC_COLS)) == frozenset()
    print("\n=== SANITY CHECK: an undeclared concept is untouched ===")
    print("  no arc -> absent from the segment-only set; empty linkbase -> empty set.")
    print("  Validated.")


# --------------------------------------------------------------------------- #
# Defect 2: the eight-year forward-fill                                        #
# --------------------------------------------------------------------------- #
def test_a_ttm_from_this_quarter_is_kept_even_when_the_dates_disagree_by_days():
    """`fiscal_end` and the TTM grid come from different columns and legitimately differ.

    ORCL files a quarter ending 2014-01-31 against a 2014-02-28 calendar -- 28 days apart,
    unambiguously the same quarter.
    """
    row = pd.Series({"period_end": pd.Timestamp("2014-01-31"), "value": 1.0})
    assert not _is_stale(row, pd.Timestamp("2014-02-28"))
    print("\n=== SANITY CHECK: the tolerance admits 52/53-week drift ===")
    print(f"  28 days apart, cap is {TTM_STALENESS_DAYS} -> kept. Validated.")


def test_the_ttm_that_froze_orcl_for_eight_years_is_refused():
    """The exact case: a 2018-08-31 window carried onto a 2026-05-31 row."""
    row = pd.Series({"period_end": pd.Timestamp("2018-08-31"), "value": 24_238_000_000.0})
    assert _is_stale(row, pd.Timestamp("2026-05-31"))
    print("\n=== SANITY CHECK: cluster 876ab8a57bd8, defect 2 ===")
    print("  ORCL grossProfit's last computable TTM ended 2018-08-31 and was carried to")
    print("  2026-05-31 -- 32 rows frozen at 24,238,000,000, with grossMargins falling")
    print("  0.609 -> 0.360 off a constant numerator. Now refused. Validated.")


def test_the_previous_quarter_is_already_too_stale():
    """One quarter of carry is already a silent basis error, so the cap sits below it."""
    row = pd.Series({"period_end": pd.Timestamp("2025-02-28"), "value": 1.0})
    assert _is_stale(row, pd.Timestamp("2025-05-31"))
    print("\n=== SANITY CHECK: the cap is half a quarter, not a whole one ===")
    print(f"  92 days apart vs a {TTM_STALENESS_DAYS}-day cap -> refused, so the tolerance")
    print("  can only ever admit the SAME fiscal quarter. Validated.")


def test_an_unknown_fiscal_end_refuses_nothing():
    """Absence of a bound is not a bound of zero: a row with no `fiscal_end` has nothing
    to be stale against, and refusing there would null every ticker's first event."""
    row = pd.Series({"period_end": pd.Timestamp("2020-01-31"), "value": 1.0})
    assert not _is_stale(row, pd.NaT)
    assert not _is_stale(pd.Series({"period_end": pd.NaT, "value": 1.0}),
                         pd.Timestamp("2020-01-31"))
    print("\n=== SANITY CHECK: NaT on either side refuses nothing ===")
    print("  no fiscal_end -> no refusal; no period_end -> no refusal. Validated.")


def test_the_stale_code_is_an_absence_not_a_qualifier():
    """A `stale_ttm` cell is NULL by construction, so it must not sit in `IS_QUALIFIER` --
    a qualifier means a value is present and a gate would stop looking for one."""
    assert rc.STALE_TTM in rc.ALL_CODES and rc.STALE_TTM not in rc.IS_QUALIFIER
    print("\n=== SANITY CHECK: stale_ttm is an absence code ===")
    print(f"  in ALL_CODES={rc.STALE_TTM in rc.ALL_CODES}, "
          f"in IS_QUALIFIER={rc.STALE_TTM in rc.IS_QUALIFIER}. Validated.")
