"""
Phase 3 of the fundamentals rebuild: resolution driven by the filer's own XBRL calculation
linkbase instead of a priority-ordered candidate-tag list.

Split per docs/testing.md: the synthetic fixtures prove the resolution MATH (which route
wins, which weight is applied, where a climb must not go), and the real-filing tests prove
it fires on actual 10-Ks across all six accounting regimes. Neither half is sufficient
alone -- a synthetic linkbase cannot contain APA's company-extension revenue element, and a
real filing cannot be made to disagree with itself on demand.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals import entity_scope as scope
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    FIELD_SUM, LINKBASE_ROOT, LINKBASE_SUM, LINKBASE_TOTAL, TAG_FALLBACK, TAG_PRIMARY,
    UNRESOLVED, ArcGraph, discover_root, resolve_field, statement_arcs)

CATALOGUE = load_catalogue("./configs")

#: (concept, taxonomy, parent, weight) -> the arc frame `statement_arcs` returns.
_ARC_COLS = ["concept", "concept_taxonomy", "parent_concept", "parent_taxonomy",
             "weight", "role_uri", "menucat", "is_abstract", "arc_filter"]

#: The default role a synthetic arc sits on. It must READ like a real income statement:
#: since Phase 3c the role URI is load-bearing -- `discover_root` requires one, which is
#: what stops a cash-flow or balance-sheet root being returned as revenue -- so a
#: placeholder like `.../role/IS` would make the fixtures silently untestable.
_INCOME_ROLE = "http://x/role/ConsolidatedStatementsOfOperations"


def _arcs(rows: list[tuple[str, str, str, float]], role: str = _INCOME_ROLE):
    return pd.DataFrame(
        [{"concept": c, "concept_taxonomy": tax, "parent_concept": p,
          "parent_taxonomy": "us-gaap", "weight": w, "role_uri": role,
          "menucat": "Statements", "is_abstract": False, "arc_filter": "both"}
         for c, tax, p, w in rows],
        columns=_ARC_COLS)


# --------------------------------------------------------------------------- #
# Synthetic known-truth: which route wins, and why                            #
# --------------------------------------------------------------------------- #
def test_declared_total_beats_the_legs_that_disagree_with_it():
    """When the filer declares a total AND its legs, the total wins and says so.

    This is the core contract: the legs are what the total is CHECKED against, never a
    substitute for reading it. If a filer's own legs do not foot to its own total, that is
    a validator finding (Phase 7), not a licence for the extractor to prefer the sum.
    """
    graph = ArcGraph(_arcs([
        ("LongTermDebtCurrent", "us-gaap", "DebtCurrent", 1.0),
        ("ShortTermBorrowings", "us-gaap", "DebtCurrent", 1.0),
    ]))
    available = frozenset({"DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"})

    resolution = resolve_field(CATALOGUE.field("shortTermDebt"), graph, available,
                               CATALOGUE)

    assert resolution.method == LINKBASE_TOTAL
    assert resolution.concept == "us-gaap:DebtCurrent"
    print("\n=== SANITY CHECK: total-vs-legs precedence ===")
    print(f"  legs declared     : LongTermDebtCurrent + ShortTermBorrowings")
    print(f"  filer's own total : DebtCurrent")
    print(f"  resolved to       : {resolution.concept} via {resolution.method}")
    print("  OK: The declared total wins; the legs stay a cross-check.")


def test_both_legs_are_summed_when_no_total_is_reported():
    """The `shortTermDebt` defect, fixed.

    `LongTermDebtCurrent` and `ShortTermBorrowings` are DISJOINT legs whose sum is
    `DebtCurrent` (FASB: "Amount of debt and lease obligation, classified as current").
    The old priority list kept one and silently discarded the other -- measured across
    2,017 (ticker, period) cells on 111 tickers, the discarded leg was the LARGER one
    54.4% of the time.
    """
    graph = ArcGraph(_arcs([
        ("LongTermDebtCurrent", "us-gaap", "DebtCurrent", 1.0),
        ("ShortTermBorrowings", "us-gaap", "DebtCurrent", 1.0),
    ]))
    # The total is declared in the structure but NOT reported -- exactly the 2,017 cells.
    available = frozenset({"LongTermDebtCurrent", "ShortTermBorrowings"})

    resolution = resolve_field(CATALOGUE.field("shortTermDebt"), graph, available,
                               CATALOGUE)

    assert resolution.method == LINKBASE_SUM
    assert [c for c, _ in resolution.children] == ["LongTermDebtCurrent",
                                                   "ShortTermBorrowings"]
    assert all(w == 1.0 for _, w in resolution.children)
    print("\n=== SANITY CHECK: disjoint legs are summed, not chosen between ===")
    print(f"  route    : {resolution.method}")
    print(f"  children : {[(c, w) for c, w in resolution.children]}")
    print("  OK: Both legs kept. The old resolver discarded one (the larger, 54.4% of the time).")


def test_a_partial_leg_set_yields_no_value_rather_than_a_wrong_one():
    """A sum missing a leg is not the total. Better an explained NULL than a plausible
    wrong number -- that preference is the whole point of the reason-code layer."""
    graph = ArcGraph(_arcs([
        ("LongTermDebtCurrent", "us-gaap", "DebtCurrent", 1.0),
        ("ShortTermBorrowings", "us-gaap", "DebtCurrent", 1.0),
    ]))
    resolution = resolve_field(CATALOGUE.field("shortTermDebt"), graph,
                               frozenset({"LongTermDebtCurrent"}), CATALOGUE)

    assert resolution.method != LINKBASE_SUM
    print("\n=== SANITY CHECK: partial roll-up refused ===")
    print(f"  only one of two legs reported -> route {resolution.method}, "
          f"concept {resolution.concept}")
    print("  OK: No partial sum was emitted.")


def test_negative_weights_are_preserved():
    """22% of Statements arcs carry weight -1.0 -- a real contra-account, not noise.
    Dropping the sign silently doubles a subtraction into an addition."""
    graph = ArcGraph(_arcs([
        ("PropertyPlantAndEquipmentGross", "us-gaap", "PropertyPlantAndEquipmentNet", 1.0),
        ("AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
         "us-gaap", "PropertyPlantAndEquipmentNet", -1.0),
    ]))
    kids = dict(graph.children_of("PropertyPlantAndEquipmentNet"))

    assert kids["AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment"] == -1.0
    assert not graph.is_pure_aggregation("PropertyPlantAndEquipmentNet") if hasattr(
        graph, "is_pure_aggregation") else True
    print("\n=== SANITY CHECK: contra-account weight ===")
    print(f"  accumulated depreciation arc weight = "
          f"{kids['AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment']}")
    print("  OK: Sign preserved.")


def test_root_discovery_finds_an_extension_total_under_a_standard_subtotal():
    """The APA shape, synthetically: the filer's revenue total is a COMPANY EXTENSION, so
    no candidate list can name it, but the linkbase declares it under a standard pretax
    subtotal."""
    pretax = ("IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
              "ExtraordinaryItemsNoncontrollingInterest")
    graph = ArcGraph(_arcs([
        ("RevenuesAndOther", "apa", pretax, 1.0),
        ("CostsAndExpenses", "us-gaap", pretax, -1.0),
        ("Revenues", "us-gaap", "RevenuesAndOther", 1.0),
    ]))
    found = discover_root(graph, frozenset({"RevenuesAndOther", "CostsAndExpenses"}))

    assert found is not None
    concept, anchor = found
    assert concept == "RevenuesAndOther"
    assert anchor == pretax
    print("\n=== SANITY CHECK: structural root discovery ===")
    print(f"  discovered {graph.qualified(concept)} under anchor {anchor}")
    print("  OK: An extension total is reachable; a candidate list could never name it.")


def test_root_discovery_never_returns_a_margin_subtotal():
    """DTE declares `IncomeLoss...BeforeIncomeTaxes <- +1 OperatingIncomeLoss`. Without
    excluding the anchors themselves, the operating MARGIN would be stored as revenue."""
    pretax = ("IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
              "ExtraordinaryItemsNoncontrollingInterest")
    graph = ArcGraph(_arcs([
        ("OperatingIncomeLoss", "us-gaap", pretax, 1.0),
        ("IncomeTaxExpenseBenefit", "us-gaap", pretax, -1.0),
        ("CostsAndExpenses", "us-gaap", "OperatingIncomeLoss", -1.0),
        ("RegulatedOperatingRevenue", "us-gaap",
         "RegulatedAndUnregulatedOperatingRevenue", 1.0),
    ]))
    available = frozenset({"OperatingIncomeLoss", "RegulatedAndUnregulatedOperatingRevenue",
                           "RegulatedOperatingRevenue", "CostsAndExpenses"})
    found = discover_root(graph, available)

    assert found is not None
    concept, anchor = found
    assert concept == "RegulatedAndUnregulatedOperatingRevenue", (
        f"resolved to {concept}, which is a margin subtotal, not a top line")
    print("\n=== SANITY CHECK: margin subtotal rejected as revenue ===")
    print(f"  anchor's only positive child was OperatingIncomeLoss (a subtotal) -> skipped")
    print(f"  fell through to the parentless revenue root: {concept} (via {anchor})")
    print("  OK: Operating margin was NOT stored as revenue.")


def test_no_linkbase_still_takes_the_top_priority_concept():
    """Older filings ship no calculation linkbase at all. The catalogue's first choice is
    still taken -- that is `tag_primary`, not `tag_fallback`. The distinction is the whole
    point of the split: `tag_fallback` must mean "we could not use the filer's structure
    AND we did not get our first choice either", or its rate is not evidence about this
    design."""
    graph = ArcGraph(_arcs([]))
    resolution = resolve_field(CATALOGUE.field("totalAssets"), graph,
                               frozenset({"Assets"}), CATALOGUE)

    assert resolution.method == TAG_PRIMARY
    assert resolution.concept == "us-gaap:Assets"
    print("\n=== SANITY CHECK: empty linkbase ===")
    print(f"  route {resolution.method} -> {resolution.concept}")
    print("  OK: Degrades to the priority list instead of failing.")


def test_never_use_concepts_can_never_resolve_a_field():
    """`never_use` is part of the contract, not a log line. MAA tags $272M of multifamily
    development capex as `PaymentsToAcquireInProcessResearchAndDevelopment`; a name-keyed
    extractor books a REIT as an R&D spender."""
    spec = CATALOGUE.field("capex")
    banned = spec.never_use()
    assert "PaymentsToAcquireInProcessResearchAndDevelopment" in banned

    graph = ArcGraph(_arcs([
        ("PaymentsToAcquireInProcessResearchAndDevelopment", "us-gaap",
         "NetCashProvidedByUsedInInvestingActivities", -1.0),
    ]))
    resolution = resolve_field(
        spec, graph, frozenset({"PaymentsToAcquireInProcessResearchAndDevelopment"}),
        CATALOGUE)

    assert resolution.concept != "us-gaap:PaymentsToAcquireInProcessResearchAndDevelopment"
    assert resolution.dc_code is not None
    assert resolution.method == UNRESOLVED
    print("\n=== SANITY CHECK: never_use enforced ===")
    print(f"  MAA's IPR&D-tagged capex offered as the only candidate -> "
          f"dc_code={resolution.dc_code}")
    print("  OK: Refused. A REIT is not booked as an R&D spender.")


def test_bank_capex_is_declared_not_applicable_rather_than_guessed():
    """JPM/BAC/USB do not appear in the `PaymentsToAcquirePropertyPlantAndEquipment`
    CY2024 frame at all -- large banks bury premises purchases in "all other investing".
    Bank FCF is therefore null BY DESIGN, and the reason travels with it."""
    graph = ArcGraph(_arcs([]))
    resolution = resolve_field(CATALOGUE.field("capex"), graph,
                               frozenset({"PaymentsToAcquirePropertyPlantAndEquipment"}),
                               CATALOGUE, regime="bank")

    assert resolution.dc_code == "not_applicable"
    assert resolution.concept is None
    assert resolution.method == UNRESOLVED, (
        "a reason-coded absence is not a routing outcome and must not pollute a route rate")
    print("\n=== SANITY CHECK: regime-declared absence ===")
    print(f"  bank capex -> dc_code={resolution.dc_code} even though the tag is available")
    print("  OK: Absent by design, with the reason recorded.")


# --------------------------------------------------------------------------- #
# Phase 3c: the three defects the 26-ticker x 15-year sweep found             #
# --------------------------------------------------------------------------- #
class _FakeXbrl:
    """Just enough of an `XBRL` to exercise `statement_arcs`' filter."""

    def __init__(self, arcs: pd.DataFrame):
        self._arcs = arcs

    def calculation_linkbase(self) -> pd.DataFrame:
        return self._arcs


def _linkbase(rows: list[tuple[str, str, object]]) -> pd.DataFrame:
    """(concept, role_uri, menucat) -> a raw calculation-linkbase frame."""
    return pd.DataFrame(
        [{"concept": c, "concept_taxonomy": "us-gaap", "parent_concept": "Parent",
          "parent_taxonomy": "us-gaap", "weight": 1.0, "role_uri": role,
          "menucat": menucat, "is_abstract": False} for c, role, menucat in rows])


def test_statement_arcs_is_the_union_of_menucat_and_the_role_test():
    """3c.1. Each test is lossy where the other is not, so neither may be the sole filter.

    `menucat` is None for 100% of arcs on 418 of 1,544 filings (all 2011 to mid-2015) AND
    marks genuine face statements `Uncategorized` in the modern era. The role test recovers
    both -- but only if it excludes the SINGULAR `...Detail` footnote roles and, critically,
    Reg S-X **Schedule I/II parent-company-only** condensed statements, which look exactly
    like face statements and are not consolidated.
    """
    arcs = statement_arcs(_FakeXbrl(_linkbase([
        # kept by menucat alone: a Consolidated Schedule of Investments is a FACE
        # statement for an investment company, and FilingSummary says so -- but the word
        # `schedule` is what excludes the Reg S-X parent-only trap, so the role test drops
        # it. This is the direction of the union that the measured sample never exercised,
        # and it is why `menucat` is kept rather than replaced.
        ("KeptByMenucat", "http://x/role/ConsolidatedScheduleOfInvestments", "Statements"),
        # kept by the role alone: a face statement FilingSummary failed to categorise
        # (measured: APA 2022 STATEMENTOFCONSOLIDATEDOPERATIONS, menucat=Uncategorized)
        ("KeptByRole", "http://x/role/STATEMENTOFCONSOLIDATEDOPERATIONS", "Uncategorized"),
        ("KeptByRoleNullMenucat", "http://x/role/ConsolidatedBalanceSheets", None),
        # dropped: footnote detail, in the singular form the audit's first pattern missed
        ("Dropped_Detail", "http://x/role/DebtScheduleOfMaturitiesDetail", None),
        ("Dropped_Disclosure", "http://x/role/DisclosureIncomeTaxes", "Details"),
        # dropped: the parent-only trap. PGR and AFL both ship one of these.
        ("Dropped_ScheduleII",
         "http://x/role/ScheduleIiCondensedFinancialInformationOfRegistrantBalanceSheets",
         None),
    ])))

    kept = dict(zip(arcs["concept"], arcs["arc_filter"]))
    assert kept == {"KeptByMenucat": "menucat", "KeptByRole": "role_uri",
                    "KeptByRoleNullMenucat": "role_uri"}, kept
    print("\n=== SANITY CHECK: 3c.1 arc filter is a union ===")
    for concept, test in kept.items():
        print(f"  kept {concept:<24s} by {test}")
    print("  dropped: singular ...Detail, DisclosureIncomeTaxes, "
          "ScheduleII parent-only balance sheet")
    print("  OK: neither test alone keeps all three; the Schedule II role is excluded.")


def test_a_parentless_root_has_a_role_at_all():
    """3c.2, the blocker. `_role_of` indexed only the `concept` column, and a parentless
    root appears ONLY in `parent_concept` -- so `role_of` returned None for every root
    `discover_root` considers, and the role test approved for this phase would have
    rejected the correct answers along with the wrong ones."""
    graph = ArcGraph(_arcs([("RegulatedOperatingRevenue", "us-gaap",
                             "RegulatedAndUnregulatedOperatingRevenue", 1.0)]))

    assert graph.parent_of("RegulatedAndUnregulatedOperatingRevenue") is None
    assert graph.role_of("RegulatedAndUnregulatedOperatingRevenue") == _INCOME_ROLE
    print("\n=== SANITY CHECK: 3c.2 roots are role-indexed ===")
    print("  a parentless root now reports its role; before this fix it reported None")
    print("  OK: the role test in discover_root is actually evaluable.")


def test_root_discovery_rejects_the_balance_sheet_and_cash_flow_roots():
    """3c.2. The sweep stored `Assets` (18 rows), `LiabilitiesAndStockholdersEquity` (16),
    cash-flow period-increase totals (24) and `ComprehensiveIncomeNetOfTax` (14) as
    `totalRevenue`, APA reporting revenue of -$467M. The mechanism was ARC ORDER: the first
    qualifying root out of a dict won, and on DTE's 2020-04-28 10-Q the correct concept was
    present, reported, all-positive and simply came second."""
    cash_role = "http://x/role/ConsolidatedStatementsOfCashFlows"
    balance_role = "http://x/role/ConsolidatedBalanceSheets"
    arcs = pd.concat([
        # the WRONG roots, deliberately first in arc order
        _arcs([("CashAndCashEquivalentsPeriodIncreaseDecrease", "us-gaap",
                "CashCashEquivalentsRestrictedCashPeriodIncreaseDecrease", 1.0)],
              role=cash_role),
        _arcs([("AssetsCurrent", "us-gaap", "Assets", 1.0)], role=balance_role),
        # the RIGHT one, last
        _arcs([("RegulatedOperatingRevenue", "us-gaap",
                "RegulatedAndUnregulatedOperatingRevenue", 1.0),
               ("UnregulatedOperatingRevenue", "us-gaap",
                "RegulatedAndUnregulatedOperatingRevenue", 1.0)]),
    ], ignore_index=True)
    graph = ArcGraph(arcs)
    available = frozenset({"Assets", "AssetsCurrent",
                           "CashCashEquivalentsRestrictedCashPeriodIncreaseDecrease",
                           "RegulatedAndUnregulatedOperatingRevenue",
                           "RegulatedOperatingRevenue", "UnregulatedOperatingRevenue"})
    durations = available - {"Assets", "AssetsCurrent"}

    found = discover_root(graph, available, duration_concepts=durations)

    assert found == ("RegulatedAndUnregulatedOperatingRevenue", "linkbase_root_node"), found
    assert discover_root(
        graph, available, duration_concepts=durations,
        banned=frozenset({"RegulatedAndUnregulatedOperatingRevenue"})) is None
    print("\n=== SANITY CHECK: 3c.2 root discovery is constrained AND ranked ===")
    print("  arc order offers the cash-flow root first and the balance-sheet root second")
    print(f"  discover_root returns -> {found[0]}")
    print("  OK: role + period_type reject both; the correct root wins regardless of order.")


def test_a_zero_in_every_period_loses_to_a_real_number_but_survives_alone():
    """3c.3. ETN tags `Revenues = 0` while `SalesRevenueNet` ($20.9-22.6 bn) sits
    undimensioned in the same filing; VRT's 2018-2020 `Revenues = 0` is its WHOLE answer,
    because those filings are the GS Acquisition Holdings blank-cheque shell pre-merger.
    The same guard has to produce opposite outcomes, and it does -- pass 1 withholds the
    zero-only concepts, pass 2 restores them only when nothing else answers."""
    graph = ArcGraph(_arcs([]))
    spec = CATALOGUE.field("totalRevenue")

    etn = resolve_field(spec, graph, frozenset({"Revenues", "SalesRevenueNet"}), CATALOGUE,
                        zero_only=frozenset({"Revenues"}))
    vrt = resolve_field(spec, graph, frozenset({"Revenues"}), CATALOGUE,
                        zero_only=frozenset({"Revenues"}))

    assert etn.concept == "us-gaap:SalesRevenueNet" and not etn.zero_only_retained
    assert vrt.concept == "us-gaap:Revenues" and vrt.zero_only_retained
    print("\n=== SANITY CHECK: 3c.3 genuine zero vs tagging artefact ===")
    print(f"  ETN shape (a real top line exists)  -> {etn.concept}, "
          f"retained={etn.zero_only_retained}")
    print(f"  VRT shape (the zero is all there is) -> {vrt.concept}, "
          f"retained={vrt.zero_only_retained}")
    print("  OK: the artefact is skipped, the real zero is kept and flagged.")


def test_an_untestable_only_when_condition_subtracts_nothing():
    """3c.5. `_resolve_subtractions` fell back to subtracting UNCONDITIONALLY whenever there
    was no single resolved concept to test siblinghood against. A `linkbase_sum` of
    `LongTermDebtCurrent + ShortTermBorrowings` contains no lease leg by construction, so
    that removed an amount that was never there -- 158 negative `shortTermDebt` values
    across 10 tickers, worst -$893M. Not subtracting is the safe direction."""
    from src.data_extract.utils.fundamentals.xbrl_linkbase import _resolve_subtractions

    spec = CATALOGUE.field("ppeNet")                  # the field that declares _only_when
    graph = ArcGraph(_arcs([
        ("PropertyPlantAndEquipmentNet", "us-gaap", "AssetsNoncurrent", 1.0),
        ("FinanceLeaseRightOfUseAsset", "us-gaap", "AssetsNoncurrent", 1.0),
    ]))
    available = frozenset({"PropertyPlantAndEquipmentNet", "FinanceLeaseRightOfUseAsset"})

    sibling = _resolve_subtractions(spec, graph, available, "PropertyPlantAndEquipmentNet")
    untestable = _resolve_subtractions(spec, graph, available, None)

    off_linkbase = _resolve_subtractions(spec, graph, available, "SomeTagRouteConcept")

    assert sibling == (), "a separately-presented sibling was never inside the total"
    assert untestable == (), "with no concept to test, containment is unproven"
    assert off_linkbase == (), (
        "a concept the linkbase never mentions cannot satisfy a structural condition -- "
        "reading that silence as 'not a sibling, therefore subtract' left 75 of the 127 "
        "surviving negative shortTermDebt values in place")

    # `shortTermDebt` demands the STRONGER test: positive evidence of containment. Measured
    # on 31 filings spanning every route that still subtracted, NO filer on the roster
    # declares a lease leg beneath its debt total -- the sibling test had never actually
    # discriminated anything, it was only ever the leg's absence letting the subtraction
    # through. ASC 842-20-45-1 requires operating lease liabilities to be presented
    # separately, so silence means OUTSIDE here, unlike ppeNet's ASC 842-20-45-4 case.
    debt = CATALOGUE.field("shortTermDebt")
    folded = ArcGraph(_arcs([
        ("FinanceLeaseLiabilityCurrent", "us-gaap", "DebtCurrent", 1.0)]))
    separate = ArcGraph(_arcs([
        ("DebtCurrent", "us-gaap", "LiabilitiesCurrent", 1.0),
        ("FinanceLeaseLiabilityCurrent", "us-gaap", "LiabilitiesCurrent", 1.0)]))
    legs = frozenset({"DebtCurrent", "FinanceLeaseLiabilityCurrent"})
    inside = _resolve_subtractions(debt, folded, legs, "DebtCurrent")
    outside = _resolve_subtractions(debt, separate, legs, "DebtCurrent")

    assert inside == ("FinanceLeaseLiabilityCurrent",), inside
    assert outside == (), outside
    print("\n=== SANITY CHECK: 3c.5 conditional subtraction ===")
    print(f"  ppeNet, sibling presentation         -> subtract {sibling}")
    print(f"  ppeNet, no concept to test           -> subtract {untestable}")
    print(f"  ppeNet, concept absent from linkbase -> subtract {off_linkbase}")
    print(f"  shortTermDebt, lease DECLARED inside -> subtract {inside}")
    print(f"  shortTermDebt, lease presented apart -> subtract {outside}")
    print("  OK: an unprovable containment never removes an amount.")


def test_a_leg_weight_is_only_trusted_against_this_fields_own_total():
    """3c.9. A calculation weight says how a concept foots into ITS OWN parent; the
    catalogue's `roll_up.sum` says how legs foot into THIS FIELD. Those coincide only when
    the legs' shared parent IS the field's total.

    MSFT declares `SellingAndMarketingExpense` and `GeneralAndAdministrativeExpense` as
    -1.0 children of `OperatingIncomeLoss` -- correct, they reduce operating income -- while
    `sellingGeneralAdmin`'s total is `SellingGeneralAndAdministrativeExpense`. Applying the
    -1.0 made SG&A **-$34.7bn on 159 of 202 rows**, the largest wrong number in either
    26-ticker sweep and the only field of 1,770 `linkbase_sum` rows affected.
    """
    from src.data_extract.utils.fundamentals.xbrl_linkbase import _linkbase_weights

    legs = ["SellingAndMarketingExpense", "GeneralAndAdministrativeExpense"]
    msft = ArcGraph(_arcs([(leg, "us-gaap", "OperatingIncomeLoss", -1.0) for leg in legs]))
    available = frozenset(legs)

    # the legs' parent is a subtotal this field never claims -> the weight is not about us
    untrusted = _linkbase_weights(msft, legs, available,
                                  {"SellingGeneralAndAdministrativeExpense"})
    # the same graph, for a field that DOES claim that parent -> the sign is load-bearing
    trusted = _linkbase_weights(msft, legs, available, {"OperatingIncomeLoss"})

    assert [w for _, w in untrusted] == [1.0, 1.0], untrusted
    assert [w for _, w in trusted] == [-1.0, -1.0], trusted
    print("\n=== SANITY CHECK: 3c.9 weights belong to a parent ===")
    print(f"  parent is NOT the field's total -> weights {[w for _, w in untrusted]}")
    print(f"  parent IS the field's total     -> weights {[w for _, w in trusted]}")
    print("  OK: an expense aggregate is no longer negated by its parent's sign.")


def test_a_composed_field_refuses_to_stand_in_for_its_missing_legs():
    """3c.9. `_compose` zero-filled every missing component, so `totalDebt` reported a LEASE
    LIABILITY as total debt on 213 of 2,655 in-sample rows (BRK-B $4.9-6.3bn, GS $2.1-2.4bn,
    META $7.6-16.7bn) and `ppeNet` reported `accumulatedDepreciation` alone as net PP&E on
    86 GS rows. A lease leg is genuinely optional; a debt leg is not, and net PP&E is a
    difference that cannot be made from one side."""
    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import _compose

    key = (2024, "FY", "instant", None, "2024-12-31")
    cell = {"value": 0.0, "fiscal_year": 2024, "fiscal_period": "FY",
            "duration_type": "instant", "period_start": None, "period_end": "2024-12-31",
            "period_days": None, "unit": "USD", "decimals": "-6"}

    def at(value):
        return {key: {**cell, "value": value}}

    debt = CATALOGUE.field("totalDebt")
    lease_only, reason = _compose(debt, tuple(debt.roll_up()),
                                  {"operatingLeaseLiability": at(6.29e9)})
    with_debt, ok = _compose(debt, tuple(debt.roll_up()),
                             {"longTermDebt": at(1.2e11), "operatingLeaseLiability": at(6.29e9)})

    ppe = CATALOGUE.field("ppeNet")
    one_leg, ppe_reason = _compose(ppe, tuple(ppe.roll_up()),
                                   {"accumulatedDepreciation": at(4.084e10)})
    both, ppe_ok = _compose(ppe, tuple(ppe.roll_up()),
                            {"ppeGross": at(6.0e10), "accumulatedDepreciation": at(-4.0e10)})

    assert lease_only == {} and reason == "incomplete_roll_up", (lease_only, reason)
    assert with_debt[key]["value"] == 1.2e11 + 6.29e9 and ok is None
    assert one_leg == {} and ppe_reason == "incomplete_roll_up", (one_leg, ppe_reason)
    assert both[key]["value"] == 2.0e10 and ppe_ok is None
    print("\n=== SANITY CHECK: 3c.9 composed fields need their load-bearing legs ===")
    print(f"  totalDebt, lease leg only  -> {lease_only or 'NULL'}  dc_code={reason}")
    print(f"  totalDebt, debt + lease    -> {with_debt[key]['value']:,.0f}")
    print(f"  ppeNet, one leg            -> {one_leg or 'NULL'}  dc_code={ppe_reason}")
    print(f"  ppeNet, gross less acc.dep -> {both[key]['value']:,.0f}")
    print("  OK: a reason-coded NULL instead of a lease liability called total debt.")


def test_a_text_sourced_field_is_not_asked_of_the_xbrl_walk():
    """3c.9. `employees` declares `"source": "text:10-K"` and is parsed out of the narrative
    by `fundamentals_employees.py`. Left in `extracted_fields` it resolved 0 times on all 52
    swept tickers and emitted one reason-coded row per filing -- ~1,600 rows a sweep saying a
    field the XBRL walk could never find was not found."""
    assert CATALOGUE.field("employees").raw.get("source", "").startswith("text")
    assert not CATALOGUE.field("employees").is_extracted
    assert "employees" not in CATALOGUE.extracted_fields
    concept_backed = [n for n in CATALOGUE.extracted_fields
                      if not CATALOGUE.field(n).raw.get("fallback_concepts")
                      and not CATALOGUE.field(n).raw.get("roll_up")]
    assert not concept_backed, (
        f"these fields are in the XBRL walk with nothing to resolve against: {concept_backed}")
    print("\n=== SANITY CHECK: 3c.9 the XBRL walk only asks for XBRL fields ===")
    print(f"  extracted_fields = {len(CATALOGUE.extracted_fields)} (employees excluded)")
    print("  OK: every remaining field has concepts or a roll-up to resolve against.")


# --------------------------------------------------------------------------- #
# Real filings: the six accounting regimes                                    #
# --------------------------------------------------------------------------- #
#: (ticker, GICS triple, the concept the FILER declares as its top line).
#: Every expectation is a measured fact from the live 10-K, not a guess.
_REGIME_CASES = [
    ("XOM", ("Energy", "Energy", "Integrated Oil & Gas"), "us-gaap:Revenues"),
    ("APA", ("Energy", "Energy", "Oil & Gas Exploration & Production"),
     "apa:RevenuesAndOther"),
    ("JPM", ("Financials", "Banks", "Diversified Banks"),
     "us-gaap:RevenuesNetOfInterestExpense"),
    ("DTE", ("Utilities", "Utilities", "Multi-Utilities"),
     "us-gaap:RegulatedAndUnregulatedOperatingRevenue"),
    ("MAA", ("Real Estate", "Equity Real Estate Investment Trusts (REITs)",
             "Multi-Family Residential REITs"), "us-gaap:Revenues"),
    ("MET", ("Financials", "Insurance", "Life & Health Insurance"), "us-gaap:Revenues"),
]

_EXPECTED_REGIME = {"XOM": "energy", "APA": "energy", "JPM": "bank", "DTE": "utility",
                    "MAA": "real_estate", "MET": "insurer"}


@pytest.fixture(scope="module")
def edgar_ready() -> bool:
    """Real-filing tests need SEC credentials and network. Skip rather than fail, matching
    the repo's convention for integration tests (docs/testing.md)."""
    if not os.getenv("SEC_USER_AGENT", "").strip():
        pytest.skip("SEC_USER_AGENT unset -- real-filing resolution tests need EDGAR")
    return True


@pytest.fixture(scope="module")
def resolved_regimes(edgar_ready) -> dict:
    """Resolve `totalRevenue` on one real 10-K per accounting regime. Module-scoped: each
    `filing.xbrl()` costs 1.4-5.8 s, so this is paid once for the whole file."""
    from edgar import Company, set_identity
    set_identity(os.getenv("SEC_USER_AGENT"))
    from src.data_extract.utils.fundamentals.xbrl_linkbase import resolve_field as rf

    out = {}
    for ticker, (sector, group, sub), _ in _REGIME_CASES:
        try:
            filing = Company(ticker).latest("10-K")
            xbrl = filing.xbrl()
        except Exception as exc:                            # noqa: BLE001
            pytest.skip(f"EDGAR unreachable for {ticker}: {exc}")
        facts = scope.consolidated_facts(xbrl.facts.to_dataframe())
        graph = ArcGraph(statement_arcs(xbrl))
        regime = CATALOGUE.regime_for(
            {"sector": sector, "industry_group": group, "sub_industry": sub},
            [str(r) for r in graph.arcs.get("role_uri", pd.Series(dtype=str))])
        out[ticker] = {
            "regime": regime, "graph": graph, "facts": facts,
            "available": scope.reported_concepts(facts),
            "resolution": rf(CATALOGUE.field("totalRevenue"), graph,
                             scope.reported_concepts(facts), CATALOGUE, regime),
            "accession": filing.accession_number,
        }
    return out


def test_each_regime_resolves_to_the_concept_its_filer_declares(resolved_regimes):
    """The §3.2 table, asserted rather than asserted-by-eye.

    The three that matter most: JPM must NOT resolve via `Revenues` (for a bank that tag
    already means NET revenue, so the basis silently switches by regime); APA must reach
    its extension total instead of the ASC-606 element it tags as literally $0.00; DTE must
    reach its regulated-revenue root rather than its operating margin.
    """
    print("\n=== SANITY CHECK: linkbase resolution across six accounting regimes ===")
    print(f"  {'ticker':7s} {'regime':13s} {'route':16s} concept")
    failures = []
    for ticker, _, expected in _REGIME_CASES:
        got = resolved_regimes[ticker]
        resolution = got["resolution"]
        print(f"  {ticker:7s} {str(got['regime']):13s} {resolution.method:16s} "
              f"{resolution.concept}")
        if resolution.concept != expected:
            failures.append(f"{ticker}: expected {expected}, got {resolution.concept}")
        if got["regime"] != _EXPECTED_REGIME[ticker]:
            failures.append(f"{ticker}: regime {got['regime']} != {_EXPECTED_REGIME[ticker]}")
    assert not failures, "; ".join(failures)
    print("  OK: All six resolve to the concept the FILER declares, on the right template.")


def test_apa_revenue_is_a_real_number_and_comes_from_an_extension(resolved_regimes):
    """The plan's headline Phase 3 criterion. APA carried `totalRevenue = 0` for 19 rows.

    The root cause is sharper than the research recorded: `us-gaap:Revenues` exists in
    APA's filing with 84 facts, EVERY ONE dimensioned (segment detail), while the real top
    line is `apa:RevenuesAndOther` -- a company-private element. No tag list can ever
    contain it.
    """
    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import _materialise
    got = resolved_regimes["APA"]
    resolution = got["resolution"]
    assert resolution.is_extension, "APA's top line is a company extension element"
    assert resolution.method == LINKBASE_ROOT

    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import _period_frame
    # `_materialise` returns ({accepted}, {refused}); only the accepted periods are the
    # filer's real top line.
    periods, refused = _materialise(resolution, _period_frame(got["facts"]))
    values = [p["value"] for p in periods.values()]
    assert values, "APA resolved a concept but produced no periods"
    assert all(v > 1e9 for v in values), f"APA revenue implausible: {values}"

    print("\n=== SANITY CHECK: APA, the zero-revenue chain ===")
    print(f"  accession      : {got['accession']}")
    print(f"  resolved       : {resolution.concept} (extension={resolution.is_extension})")
    print(f"  anchor         : {resolution.anchor}")
    print(f"  values         : {[f'${v/1e9:.3f}B' for v in sorted(values, reverse=True)]}")
    print(f"  refused        : {len(refused)} period(s)")
    print(f"  us-gaap:Revenues undimensioned? "
          f"{'Revenues' in got['available']}  <- the old resolver's target")
    print("  OK: Non-zero, non-null, and sourced from the filer's own declared total.")


def test_shares_outstanding_uses_the_multi_class_safe_cover_page_tag(resolved_regimes):
    """`dei:EntityCommonStockSharesOutstanding` is the ONLY summable share tag for a
    multi-class issuer, and it is namespaced in the catalogue while the linkbase and the
    reported-concept set are keyed bare.

    Matching the two forms literally skipped it on 5 of 6 filings and fell through to
    `CommonStockSharesOutstanding` -- a single share class. That is how a multi-class
    issuer's share count silently becomes one class of several, so this asserts the
    normalisation rather than trusting it.
    """
    from src.data_extract.utils.fundamentals.xbrl_linkbase import resolve_field as rf
    print("\n=== SANITY CHECK: multi-class share tag ===")
    wrong = []
    for ticker, _, _ in _REGIME_CASES:
        got = resolved_regimes[ticker]
        resolution = rf(CATALOGUE.field("sharesOutstanding"), got["graph"],
                        got["available"], CATALOGUE, got["regime"])
        print(f"  {ticker:6s} {resolution.method:14s} {resolution.concept}")
        if resolution.concept == "us-gaap:CommonStockSharesOutstanding":
            wrong.append(ticker)
    assert not wrong, (
        f"{wrong} resolved to the single-class CommonStockSharesOutstanding instead of the "
        "cover-page dei tag -- the multi-class NULL defect has returned")
    print("  OK: the cover-page dei tag wins; no filing fell back to a single share class.")


def test_route_labels_separate_priority_from_genuine_fallthrough(resolved_regimes):
    """With `tag_primary` split out, the plan's ~20% gate applies LITERALLY to
    `tag_fallback` -- no exclusions to argue about.

    Pooled as one label the rate was 27.8% against that gate, and defending it required
    arguing that cover-page `dei:` tags and reason-coded absences "should not count". They
    genuinely should not, but a metric that needs that argument is the wrong metric. The
    breadth-and-history version of this measurement lives in `test_linkbase_history.py`;
    this one keeps the six-regime snapshot honest and fast.
    """
    from src.data_extract.utils.fundamentals.xbrl_linkbase import resolve_field as rf
    tier1 = [f for f in CATALOGUE.by_tier(1) if CATALOGUE.field(f).is_extracted]

    counts: dict[str, int] = {}
    for ticker, _, _ in _REGIME_CASES:
        got = resolved_regimes[ticker]
        for name in tier1:
            resolution = rf(CATALOGUE.field(name), got["graph"], got["available"],
                            CATALOGUE, got["regime"])
            counts[resolution.method] = counts.get(resolution.method, 0) + 1

    resolved = {m: n for m, n in counts.items() if m != UNRESOLVED}
    total = sum(resolved.values())
    fallback = resolved.get(TAG_FALLBACK, 0) / total

    print("\n=== SANITY CHECK: resolution routes, TIER-1 fields x 6 regimes ===")
    for method, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        note = "  (not a route -- dc_code says why)" if method == UNRESOLVED else ""
        print(f"  {method:16s} {n:3d}{note}")
    print(f"  genuine tag_fallback = {fallback:.1%} of {total} resolved "
          f"(plan gate: >20% means re-examine)")
    assert fallback <= 0.20, (
        f"tag_fallback {fallback:.1%} exceeds the plan's 20% gate -- the linkbase premise "
        "needs re-examining before the full rebuild")
    print("  OK: the filer's own structure carries the totals.")


# --------------------------------------------------------------------------- #
# 4c.7: AXP's revenue basis, across the ASC-606 break                          #
# --------------------------------------------------------------------------- #
#: AXP's GICS. `Consumer Finance` maps to the **bank** regime through
#: `fundamentals_regimes.json`, which matters: v1 recorded AXP as routing to `industrial`
#: and that was wrong -- "Transaction & Payment Processing Services" is V and MA. Asserted
#: below rather than assumed, because the regime selects the two-leg roll-up this fix needs.
_AXP_GICS = {"sector": "Financials", "industry_group": "Financial Services",
             "sub_industry": "Consumer Finance"}


@pytest.fixture(scope="module")
def axp_revenue(edgar_ready) -> dict:
    """AXP `totalRevenue` from one pre-ASC-606 10-K and one recent one.

    Two filings, because the whole point is that a single Rule 9-04 basis now spans a break
    that used to be a basis switch: the older filing is where the banned extension used to
    win, the newer one is where the filer's own `RevenuesNetOfInterestExpense` already did.
    """
    from edgar import Company, set_identity

    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import filing_rows
    set_identity(os.getenv("SEC_USER_AGENT"))

    company = Company("AXP")
    out: dict[int, pd.DataFrame] = {}
    for year in (2016, 2025):
        try:
            filing = next(f for f in company.get_filings(form="10-K")
                          if pd.Timestamp(f.filing_date).year == year
                          and not str(f.form).upper().endswith("/A"))
            rows = filing_rows("AXP", str(company.cik), filing, CATALOGUE, _AXP_GICS)
        except Exception as exc:                                    # noqa: BLE001
            pytest.skip(f"EDGAR unreachable for AXP {year}: {exc}")
        out[year] = pd.DataFrame(rows)
    return out


def test_axp_carries_one_rule_9_04_basis_across_the_asc_606_break(axp_revenue):
    """The 4c.7 acceptance: AXP's top line must be ONE Reg S-X Rule 9-04 basis for its whole
    history, not post-provision before ASC 606 and pre-provision after.

    What it used to resolve to is `axp:TotalRevenuesNetOfInterestExpenseAfterProvisions
    ForLosses` -- a **company extension**, which is why it appears in none of AXP's four
    companyfacts namespaces and why the plan's us-gaap-only search called it absent. It is
    revenue AFTER the credit provision, i.e. a different caption from caption 10, so it is
    not comparable with JPM's. Banning it is only safe because AXP tags **both** legs of the
    regime's roll-up for its entire history (`InterestIncomeExpenseNet` and
    `NoninterestIncome`, 225 facts each, fy2009-2026) -- otherwise the ban would turn 91 rows
    into reason-coded nulls instead of into a comparable basis. Measured, the older filing
    moves onto the two legs and revenue RISES by the provision: FY2012 $31,582M against the
    $29,592M the post-provision element carried.
    """
    banned = "TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses"
    print("\n=== SANITY CHECK: AXP totalRevenue, one Rule 9-04 basis ===")
    for year, rows in sorted(axp_revenue.items()):
        assert set(rows["regime"].dropna().unique()) == {"bank"}, (
            "AXP must route to the bank regime -- the two-leg roll-up depends on it")
        block = rows[(rows.field == "totalRevenue") & rows.value.notna()
                     & (rows.duration_type == "annual")].sort_values("period_end")
        assert not block.empty, f"AXP {year}: no annual revenue at all -- the ban nulled it"
        for row in block.tail(3).itertuples():
            print(f"  {year} 10-K  {str(row.period_end)[:10]}  "
                  f"{row.value / 1e6:>10,.0f}M  {row.resolution_method:14s} "
                  f"{row.source_concept}")
            assert banned not in str(row.source_concept), (
                f"AXP {str(row.period_end)[:10]} still resolves post-provision")
            # A $30-70bn issuer. The post-provision element runs ~$2bn lower, so this band
            # only catches a collapse to a leg or to nothing, which is what the ban risked.
            assert 20e9 < row.value < 100e9, f"{row.value:,.0f} is not AXP's top line"
    early = axp_revenue[2016]
    early_block = early[(early.field == "totalRevenue") & early.value.notna()
                        & (early.duration_type == "annual")]
    assert (early_block["resolution_method"] == "linkbase_sum").all(), (
        "the pre-ASC-606 filing must resolve on the two-leg Rule 9-04 roll-up")
    print("  OK: no post-provision row survives, and the early filing sums the two legs.")
