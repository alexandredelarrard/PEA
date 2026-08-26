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
    GROSS_PROFIT_IDENTITY_TOLERANCE, TTM_STALENESS_DAYS, _contradicts_gross_profit,
    _gross_profit_identity, _is_stale)
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


# --------------------------------------------------------------------------- #
# Defect 3: the derivation that should have filled the hole                    #
# --------------------------------------------------------------------------- #
def _facts(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(field, period_end, value) -> the `visible` frame the identity guard reads."""
    return pd.DataFrame(
        [{"field": f, "period_end": pd.Timestamp(pe), "duration_type": "annual",
          "value": v} for f, pe, v in rows])


def test_a_filer_that_never_tags_gross_profit_gets_the_derived_number():
    """ORCL's real end state. Deleting the wrong number is only half the job.

    Reg S-X Rule 5-03 never required a gross-profit line -- the phrase appears zero times in
    its text -- so Oracle presenting none is not a filing defect, and every data vendor
    publishes an Oracle gross margin by computing it. FY2026: 67,357 - (17,597 + 868 +
    4,556) = 44,336, a 65.8% margin.
    """
    row = {"totalRevenue": 67_357e6, "costOfRevenue": 23_021e6, "grossProfit": None}
    visible = _facts([("totalRevenue", "2026-05-31", 67_357e6),
                      ("costOfRevenue", "2026-05-31", 23_021e6)])
    got = _gross_profit_identity(row, visible)
    assert got == 44_336e6
    print("\n=== SANITY CHECK: the derivation, on ORCL's FY2026 numbers ===")
    print(f"  67,357 - 23,021 = {got / 1e6:,.0f}  ->  margin "
          f"{got / row['totalRevenue']:.1%}")
    print("  Oracle files no gross-profit line; this is how every vendor has the number.")
    print("  Validated.")


def test_the_identity_is_refused_where_the_filer_s_own_tags_break_it():
    """CAT and COST, measured: their tagged `grossProfit` is well below revenue minus our
    `costOfRevenue`, so our cost figure is short and a derived number would overstate.

    On the as-filed facts, 11 of 13 tickers that tag all three satisfy the identity in
    100% of rows; CAT breaks it on 24 rows by +22.5% and COST on 6 by +20.3%.
    """
    visible = _facts([("totalRevenue", "2024-12-31", 100.0),
                      ("costOfRevenue", "2024-12-31", 60.0),
                      ("grossProfit", "2024-12-31", 25.0)])       # 40 != 25
    assert _contradicts_gross_profit(visible)
    row = {"totalRevenue": 110.0, "costOfRevenue": 66.0, "grossProfit": None}
    assert _gross_profit_identity(row, visible) is None
    print("\n=== SANITY CHECK: a filer that contradicts the identity is refused ===")
    print("  filed gross profit 25 vs revenue-cost 40 -> contradiction -> no derivation.")
    print("  Validated.")


def test_a_filer_that_agrees_within_rounding_still_gets_the_derivation():
    """The guard must not fire on `decimals=-6` rounding, only on a real disagreement."""
    visible = _facts([("totalRevenue", "2024-12-31", 100.0),
                      ("costOfRevenue", "2024-12-31", 60.0),
                      ("grossProfit", "2024-12-31", 40.2)])       # 0.5% out
    assert not _contradicts_gross_profit(visible)
    row = {"totalRevenue": 110.0, "costOfRevenue": 66.0, "grossProfit": None}
    assert _gross_profit_identity(row, visible) == 44.0
    print("\n=== SANITY CHECK: the tolerance absorbs rounding, not disagreement ===")
    print(f"  0.5% apart vs a {GROSS_PROFIT_IDENTITY_TOLERANCE:.0%} band -> derived. "
          "Validated.")


def test_an_identity_short_one_term_is_not_an_approximation_of_itself():
    """No `costOfRevenue`, no derivation. This is the rule that keeps a leg-sum from
    silently standing in for a total -- the same one that forbids a `totalLiabilities`
    `roll_up.any_of`."""
    visible = _facts([("totalRevenue", "2026-05-31", 67_357e6)])
    assert _gross_profit_identity(
        {"totalRevenue": 67_357e6, "costOfRevenue": None, "grossProfit": None},
        visible) is None
    assert _gross_profit_identity(
        {"totalRevenue": None, "costOfRevenue": 23_021e6, "grossProfit": None},
        visible) is None
    print("\n=== SANITY CHECK: a missing term refuses rather than approximates ===")
    print("  no cost -> None; no revenue -> None. Validated.")


def test_silence_from_a_filer_is_not_a_contradiction():
    """A filer that simply never tags `grossProfit` has not disagreed with anything.

    The distinction the `minorityInterest` bridge had to learn: absence of a tag is not
    evidence against, and a lifetime rule asserted over a regime is what nearly claimed UNH
    earns no premiums.
    """
    visible = _facts([("totalRevenue", "2026-05-31", 67_357e6),
                      ("costOfRevenue", "2026-05-31", 23_021e6)])
    assert not _contradicts_gross_profit(visible)
    assert not _contradicts_gross_profit(pd.DataFrame(
        columns=["field", "period_end", "duration_type", "value"]))
    print("\n=== SANITY CHECK: never tagging it is not disagreeing with it ===")
    print("  two of three tags present -> no contradiction; empty frame -> none.")
    print("  Validated.")


def test_operating_income_is_deliberately_not_derived():
    """Its declared `derived_fallback` is NOT an identity and must stay unwired.

    Measured on the same substrate: `revenue - cost - SG&A - R&D - D&A` lands within 1% of
    the filed figure in **0.5% of 550 rows**, mean absolute error 29.3%, mean signed bias
    -18.1% -- it omits restructuring, impairment, acquisition-related and
    intangible-amortisation lines. Wiring it would inject exactly the plausible-but-wrong
    number this cluster exists to remove.
    """
    import inspect

    from src.data_extract.utils.fundamentals import build_history as bh

    source = inspect.getsource(bh._snapshot)
    assert "_gross_profit_identity" in source
    assert "_operating_income_identity" not in source
    catalogue_says = CATALOGUE.field("operatingIncome").raw.get("derived_fallback")
    assert catalogue_says, "the catalogue still declares it, and that is fine -- as prose"
    print("\n=== SANITY CHECK: operatingIncome stays unwired, on purpose ===")
    print(f"  catalogue declares: {catalogue_says}")
    print("  0.5% of 550 rows land within 1% of the filed value -> not an identity.")
    print("  Validated.")


# --------------------------------------------------------------------------- #
# The config half: teaching costOfRevenue to read ORCL's own cost captions      #
# --------------------------------------------------------------------------- #
def test_cost_of_revenue_can_reach_route_3b_at_all():
    """Route 3b returns immediately unless the field declares `any_of` + `anchor` +
    `anchor_role`, so without all three ORCL's cost captions are unreachable however well
    the by_ticker register describes them."""
    roll_up = CATALOGUE.field("costOfRevenue").raw["roll_up"]
    assert roll_up["anchor"] == ["CostsAndExpenses"]
    assert roll_up["anchor_role"] == "income_statement"
    assert roll_up["any_of"]
    # A cost ADDS to total operating expenses -- the opposite sign to capex under the
    # investing node. Verified on ORCL 0001193125-26-277521: every child of
    # `CostsAndExpenses` carries weight 1.0.
    assert roll_up["leaf_weight"] == 1.0
    print("\n=== SANITY CHECK: costOfRevenue is route-3b eligible ===")
    print(f"  anchor={roll_up['anchor']} role={roll_up['anchor_role']} "
          f"leaf_weight={roll_up['leaf_weight']} groups={len(roll_up['any_of'])}")
    print("  Validated.")


def test_the_total_concept_is_not_also_a_leaf_group():
    """`CostOfRevenue` is this field's `total_concept`. Listing it as a group too would let
    a filer that declares it as one LEG beside others have it counted twice."""
    groups = CATALOGUE.field("costOfRevenue").raw["roll_up"]["any_of"]
    assert "CostOfRevenue" not in {c for group in groups for c in group}
    print("\n=== SANITY CHECK: the total is not also a leg ===")
    print(f"  any_of members: {sorted({c for g in groups for c in g})}")
    print("  CostOfRevenue absent, so route 1 keeps it. Validated.")


def test_orcl_declares_its_cost_captions_with_disjoint_era_groups():
    """Within a group the members must never co-occur, or the sum double-counts.

    Measured on all 17 ORCL 10-K linkbases 2010-2026: each group holds era variants with
    DISJOINT year ranges -- CloudAndSoftware (2026) succeeds CloudServicesAndLicenseSupport,
    HardwareExpenses (2017-2026) succeeds HardwareSystemsProductsCost (2012-2016), and so on.
    """
    leaves, not_leaves = CATALOGUE.filer_leaves("ORCL", "costOfRevenue")
    assert leaves, "ORCL must declare its cost captions or the field is unreachable"
    seen: set[str] = set()
    for group in leaves:
        for concept in group:
            assert concept.startswith("orcl:"), "only EXTENSIONS belong in the register"
            assert concept not in seen, f"{concept} is in two groups -- it would double-count"
            seen.add(concept)
    # The one extension under the anchor that is NOT a cost of revenue. Naming it is what
    # lets guard 3 tell "classified as excluded" from "never looked at".
    assert "orcl:RestructuringAndOtherExpenses" in not_leaves
    assert not (seen & set(not_leaves))
    print("\n=== SANITY CHECK: ORCL's cost register ===")
    print(f"  {len(leaves)} era groups, {len(seen)} distinct extensions, no overlap")
    print(f"  not_leaves: {sorted(not_leaves)}")
    print("  Validated.")


def test_every_register_entry_carries_written_evidence():
    """A per-filer override with no evidence is the guess the register exists to replace --
    `load_catalogue` refuses one, and this pins the ORCL entry specifically."""
    entry = CATALOGUE.ticker_exceptions["ORCL"]["costOfRevenue"]
    assert entry.get("evidence") and entry.get("verified")
    for figure in ("23,021", "16,927", "15,143"):
        assert figure in entry["evidence"], f"{figure} must be in the evidence"
    print("\n=== SANITY CHECK: the register entry is evidenced ===")
    print(f"  verified {entry['verified']}, evidence reproduces the filed statements:")
    print("  FY2026 17,597+868+4,556 = 23,021; FY2025 = 16,927; FY2024 = 15,143.")
    print("  Validated.")
