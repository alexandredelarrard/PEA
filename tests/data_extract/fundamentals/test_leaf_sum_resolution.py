"""
Phase 4b of the fundamentals rebuild: route 3b, `statement_leaf_sum`.

The hole it closes is a CONCEPT-COVERAGE one, not a period one. `capex` resolved for
nothing on 9 of 52 swept tickers and `depAmort` for nothing on 9, killing Tier-1
`freeCashflow` and `ebitda` for ~17% of the roster each -- concentrated in exactly the
regimes a reader wants them for (an E&P, two utilities, three REIT/insurers, and MSFT,
GOOGL, ORCL and UNP). In every case the filer REPORTS the number; the resolver could not
name the concept it sits under.

Split per docs/testing.md, and here the split is load-bearing rather than conventional:

  * The synthetic half proves the three GUARDS, and each guard exists because a specific
    plausible design was measured and refuted. A synthetic linkbase is the only way to make
    a filer disagree with itself on demand -- to put a note-level `Depreciation` and a
    cash-flow `Depreciation` in the same graph, or an unclassifiable extension beside a
    good leaf.
  * The real half proves the numbers. `DTE FY2025 capex = $4,429M` is checkable against the
    filer's OWN arithmetic and no synthetic fixture can establish that.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    LINKBASE_TOTAL, PARTIAL_LEAF_SUM, STATEMENT_LEAF_SUM, TAG_FALLBACK, UNRESOLVED,
    ArcGraph, resolve_field)

CATALOGUE = load_catalogue("./configs")

_ARC_COLS = ["concept", "concept_taxonomy", "parent_concept", "parent_taxonomy",
             "weight", "role_uri", "menucat", "is_abstract", "arc_filter"]

#: Route 3b keys entirely off the ROLE: a leaf counts only where the filer declares it
#: beneath the anchor node on a cash-flow role. These two strings ARE the test subject, so
#: they must read like the real thing -- FASB's own bank cash-flow role is
#: `StatementOfCashFlowsIndirectDepositBasedOperations`, which is why the pattern matches
#: `cash flow` rather than anything narrower.
_CF_ROLE = "http://x/role/ConsolidatedStatementsOfCashFlows"
_NOTE_ROLE = "http://x/role/PropertyPlantAndEquipmentDetail"

_INVEST = "NetCashProvidedByUsedInInvestingActivities"
_OPERATE = "NetCashProvidedByUsedInOperatingActivities"


def _arcs(rows: list[tuple[str, str, str, float, str]]) -> pd.DataFrame:
    """(concept, taxonomy, parent, weight, role) -> the frame `statement_arcs` returns."""
    return pd.DataFrame(
        [{"concept": c, "concept_taxonomy": tax, "parent_concept": p,
          "parent_taxonomy": "us-gaap", "weight": w, "role_uri": role,
          "menucat": "Statements", "is_abstract": False, "arc_filter": "both"}
         for c, tax, p, w, role in rows],
        columns=_ARC_COLS)


# --------------------------------------------------------------------------- #
# Synthetic known-truth: the three guards                                     #
# --------------------------------------------------------------------------- #
def test_disjoint_leaves_are_summed_when_no_total_is_reported():
    """The route's whole reason to exist: route 3's `sum` is all-or-nothing, so a filer
    reporting 2 of the 3 declared capex children got NOTHING. Measured, no filer on either
    26-ticker roster reports all three, so route 3 fired for capex **zero times in 3,163
    filings**."""
    graph = ArcGraph(_arcs([
        ("PaymentsToAcquireOilAndGasPropertyAndEquipment", "us-gaap", _INVEST, -1.0,
         _CF_ROLE),
        ("PaymentsToAcquireOtherPropertyPlantAndEquipment", "us-gaap", _INVEST, -1.0,
         _CF_ROLE),
        ("PaymentsToAcquireBusinessesNetOfCashAcquired", "us-gaap", _INVEST, -1.0,
         _CF_ROLE),
    ]))
    available = frozenset({"PaymentsToAcquireOilAndGasPropertyAndEquipment",
                           "PaymentsToAcquireOtherPropertyPlantAndEquipment",
                           "PaymentsToAcquireBusinessesNetOfCashAcquired"})

    resolution = resolve_field(CATALOGUE.field("capex"), graph, available, CATALOGUE,
                              "energy")

    assert resolution.method == STATEMENT_LEAF_SUM
    picked = [c for c, _ in resolution.children]
    assert picked == ["PaymentsToAcquireOilAndGasPropertyAndEquipment",
                      "PaymentsToAcquireOtherPropertyPlantAndEquipment"]
    assert "PaymentsToAcquireBusinessesNetOfCashAcquired" not in picked
    assert resolution.anchor == _INVEST
    print("\n=== SANITY CHECK: disjoint capex leaves summed, acquisitions excluded ===")
    print(f"  route  : {resolution.method}  anchor: {resolution.anchor}")
    print(f"  summed : {picked}")
    print("  OK: A business acquisition sits in the same node and is not capex.")


def test_only_the_first_reported_alternative_in_a_group_is_taken():
    """Within a group the alternatives are era or naming variants of ONE line, so taking
    two would double-count. Measured on SWKS: `AmortizationOfIntangibleAssets` (4 filings),
    `OtherDepreciationAndAmortization` (13) and its own extension (17) never co-occur, and
    the middle two carry the identical $296M."""
    graph = ArcGraph(_arcs([
        ("Depreciation", "us-gaap", _OPERATE, 1.0, _CF_ROLE),
        ("AmortizationOfIntangibleAssets", "us-gaap", _OPERATE, 1.0, _CF_ROLE),
        ("OtherDepreciationAndAmortization", "us-gaap", _OPERATE, 1.0, _CF_ROLE),
    ]))
    # A filer that (unusually) reports BOTH members of the amortisation group.
    available = frozenset({"Depreciation", "AmortizationOfIntangibleAssets",
                           "OtherDepreciationAndAmortization"})

    resolution = resolve_field(CATALOGUE.field("depAmort"), graph, available, CATALOGUE)

    picked = [c for c, _ in resolution.children]
    assert resolution.method == STATEMENT_LEAF_SUM
    assert picked == ["Depreciation", "AmortizationOfIntangibleAssets"]
    assert "OtherDepreciationAndAmortization" not in picked
    print("\n=== SANITY CHECK: one alternative per group ===")
    print(f"  both amortisation spellings reported -> summed {picked}")
    print("  OK: The second spelling of the same line was not added twice.")


def test_a_note_level_leaf_is_refused_because_of_its_ROLE():
    """GUARD 1, and the sharpest refuted design in the section.

    "Sum the name-matching D&A leaves the filer reports" was measured against ground truth
    -- every (ticker, year) publishing BOTH the aggregate and the legs -- and reproduced the
    aggregate in only **14 of 84** cases. The cause is that AAPL's `us-gaap:Depreciation`
    is a **PP&E-NOTE disclosure** ($8,000M FY2025) and not its cash-flow line
    ($11,698M): **-31.6%**. A concept name cannot tell those apart; the role can.
    """
    graph = ArcGraph(_arcs([
        # the same concept, in a note. This is AAPL's actual shape.
        ("Depreciation", "us-gaap", "PropertyPlantAndEquipmentNet", 1.0, _NOTE_ROLE),
    ]))
    resolution = resolve_field(CATALOGUE.field("depAmort"), graph,
                               frozenset({"Depreciation"}), CATALOGUE)

    assert resolution.method != STATEMENT_LEAF_SUM
    print("\n=== SANITY CHECK: role guard ===")
    print(f"  a note-role `Depreciation` -> route {resolution.method}, "
          f"concept {resolution.concept}")
    print("  OK: Route 3b declined. Admitting it is a -31.6% answer on AAPL.")


def test_the_declared_weight_sign_excludes_a_contra_in_the_same_node():
    """GUARD 2. PGR, MET and CB each park a **-1.0**
    `AccretionAmortizationOfDiscountsAndPremiumsInvestments` inside the operating-activities
    node -- $124M, $1,840M and $409M. Its NAME contains 'Amortization', so a name filter
    admits it and books an investment-discount contra as depreciation. The weight does not.
    """
    graph = ArcGraph(_arcs([
        ("Depreciation", "us-gaap", _OPERATE, 1.0, _CF_ROLE),
        ("AccretionAmortizationOfDiscountsAndPremiumsInvestments", "us-gaap", _OPERATE,
         -1.0, _CF_ROLE),
    ]))
    available = frozenset({"Depreciation",
                           "AccretionAmortizationOfDiscountsAndPremiumsInvestments"})

    resolution = resolve_field(CATALOGUE.field("depAmort"), graph, available, CATALOGUE)

    assert resolution.method == STATEMENT_LEAF_SUM
    assert [c for c, _ in resolution.children] == ["Depreciation"]
    print("\n=== SANITY CHECK: weight-sign guard ===")
    print(f"  a -1.0 accretion contra beside a +1.0 depreciation -> "
          f"summed {[c for c, _ in resolution.children]}")
    print("  OK: The contra was excluded by its SIGN, not by its name.")


def test_the_declared_weight_is_never_used_as_a_MULTIPLIER():
    """The corollary of guard 2, and the reason a sign error here would be invisible.

    A capex payment is TAGGED POSITIVE and carries -1.0 because it reduces net investing
    cash. `_materialise` multiplies whatever weights it is handed, so passing the declared
    -1.0 through would turn a `non_negative` field negative on every single row -- a defect
    that looks like a sign-convention bug in the filer rather than in us.
    """
    graph = ArcGraph(_arcs([
        ("PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap", _INVEST, -1.0, _CF_ROLE),
    ]))
    resolution = resolve_field(
        CATALOGUE.field("capex"), graph,
        frozenset({"PaymentsToAcquirePropertyPlantAndEquipment"}), CATALOGUE, "utility")

    assert resolution.method == STATEMENT_LEAF_SUM
    assert [w for _, w in resolution.children] == [1.0]
    print("\n=== SANITY CHECK: the weight is an admission test, not a coefficient ===")
    print(f"  filer declares -1.0 -> children carry {resolution.children}")
    print("  OK: +1.0 stored. A non-negative field stays non-negative.")


def test_an_unclassifiable_extension_sibling_refuses_the_sum():
    """GUARD 3, and the crux of the whole section.

    "A negative-weight extension child of the investing node is capex" was the only
    candidate rule that would have reached DTE and NEE structurally. Measured across 17
    tickers x 941 filings it admits `apa:EquityMethodInvestmentContribution` ($501M, an
    investment), `nee:PurchasesOfSecuritiesInSpecialUseFunds` ($1.4-2.6bn, securities),
    `dte:ConsolidationOfVIES` and `eog:ChangesInComponentsOfWorkingCapital...`. The inverse
    framing fails on the same rows.

    So an unregistered extension in the field's own direction means the sum WOULD BE SHORT
    by an unknown amount, and a short capex is worse than a missing one: SWKS `depAmort`
    reads $392M against a true $614M (-36%) on exactly this shape.
    """
    graph = ArcGraph(_arcs([
        ("PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap", _INVEST, -1.0, _CF_ROLE),
        ("SomethingCapexLike", "acme", _INVEST, -1.0, _CF_ROLE),
    ]))
    available = frozenset({"PaymentsToAcquirePropertyPlantAndEquipment",
                           "SomethingCapexLike"})

    resolution = resolve_field(CATALOGUE.field("capex"), graph, available, CATALOGUE,
                               "utility")

    assert resolution.method != STATEMENT_LEAF_SUM
    print("\n=== SANITY CHECK: partial-leaf guard ===")
    print(f"  an unregistered `acme:` sibling -> route {resolution.method}, "
          f"dc_code {resolution.dc_code}")
    print("  OK: No short sum emitted.")


def test_a_refusal_still_lets_the_TAG_ROUTES_answer():
    """A refusal is NOT a null, and this is what makes guard 3 safe to leave strict.

    If route 3b terminated on refusal, every filer parking an unrelated extension in the
    node would LOSE a value it already had -- XOM (`xom:AdditionalInvestmentsAndAdvances`,
    45 filings), DUK (`duk:PurchasesOfEmissionAllowances`), EOG. Instead the refusal is
    carried forward and only becomes a `dc_code` if nothing else answers.
    """
    graph = ArcGraph(_arcs([
        ("PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap", _INVEST, -1.0, _CF_ROLE),
        ("SomethingCapexLike", "acme", _INVEST, -1.0, _CF_ROLE),
    ]))
    available = frozenset({"PaymentsToAcquirePropertyPlantAndEquipment",
                           "SomethingCapexLike"})

    resolution = resolve_field(CATALOGUE.field("capex"), graph, available, CATALOGUE,
                               "utility")

    assert resolution.method == TAG_FALLBACK
    assert resolution.concept == "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
    assert resolution.dc_code is None
    print("\n=== SANITY CHECK: refusal falls through, it does not null ===")
    print(f"  route {resolution.method} kept {resolution.concept}")
    print("  OK: The value a filer already had is not lost to a guard.")


def test_a_refusal_with_nothing_else_to_fall_back_on_is_reason_coded():
    """The other half: when the tag routes CANNOT answer either, the null must say that
    route 3b saw the leaves and refused -- not the generic `not_disclosed`, which would
    read as "the filer never reported it". The amount IS reported; naming it needs a
    `by_ticker` register entry."""
    graph = ArcGraph(_arcs([
        ("PaymentsToDevelopRealEstateAssets", "us-gaap", _INVEST, -1.0, _CF_ROLE),
        ("SomeDevelopmentSpend", "acme", _INVEST, -1.0, _CF_ROLE),
    ]))
    # Neither `PaymentsToAcquireProductiveAssets` nor `...PropertyPlantAndEquipment` --
    # the two capex candidates -- is reported, so route 5 has nothing.
    available = frozenset({"PaymentsToDevelopRealEstateAssets", "SomeDevelopmentSpend"})

    resolution = resolve_field(CATALOGUE.field("capex"), graph, available, CATALOGUE,
                               "real_estate")

    assert resolution.method == UNRESOLVED
    assert resolution.dc_code == PARTIAL_LEAF_SUM
    print("\n=== SANITY CHECK: an unexplained null becomes an explained one ===")
    print(f"  dc_code {resolution.dc_code}")
    print("  OK: `partial_leaf_sum` distinguishes 'refused' from 'never reported'.")


def test_a_registered_extension_leaf_completes_the_sum():
    """Option A in one test. DTE's whole capital programme is
    `dte:PlantAndEquipmentExpenditures{Utility,NonUtility}` -- company-extension elements
    that NO candidate list can ever contain, and that `companyfacts` cannot even see
    (it publishes no extension taxonomy at all, so DTE reads capex-blind there while
    tagging it in every 10-K)."""
    graph = ArcGraph(_arcs([
        ("PlantAndEquipmentExpendituresUtility", "dte", _INVEST, -1.0, _CF_ROLE),
        ("PlantAndEquipmentExpendituresNonUtility", "dte", _INVEST, -1.0, _CF_ROLE),
        ("ConsolidationOfVIES", "dte", _INVEST, -1.0, _CF_ROLE),
    ]))
    available = frozenset({"PlantAndEquipmentExpendituresUtility",
                           "PlantAndEquipmentExpendituresNonUtility",
                           "ConsolidationOfVIES"})

    resolution = resolve_field(CATALOGUE.field("capex"), graph, available, CATALOGUE,
                               "utility", ticker="DTE")

    assert resolution.method == STATEMENT_LEAF_SUM
    assert [c for c, _ in resolution.children] == [
        "PlantAndEquipmentExpendituresUtility", "PlantAndEquipmentExpendituresNonUtility"]
    print("\n=== SANITY CHECK: the per-filer extension register ===")
    print(f"  summed {[c for c, _ in resolution.children]}")
    print("  OK: `dte:ConsolidationOfVIES` is registered NOT a leaf, so it neither joins "
          "the sum nor refuses it.")


def test_the_same_graph_WITHOUT_the_register_refuses():
    """The register is what turns option B into option A, and this pins the difference.

    Same filing, same graph, no ticker: the extension legs are unclassifiable and the sum
    is refused. That is the measured cost of option B -- APA, MAA, PLD, SWKS, DTE, NEE and
    MSFT all stay null under it, which is why the plan's own table shows B fixing `capex`
    for EOG alone.
    """
    graph = ArcGraph(_arcs([
        ("PlantAndEquipmentExpendituresUtility", "dte", _INVEST, -1.0, _CF_ROLE),
        ("PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap", _INVEST, -1.0, _CF_ROLE),
    ]))
    available = frozenset({"PlantAndEquipmentExpendituresUtility",
                           "PaymentsToAcquirePropertyPlantAndEquipment"})

    with_register = resolve_field(CATALOGUE.field("capex"), graph, available, CATALOGUE,
                                  "utility", ticker="DTE")
    without = resolve_field(CATALOGUE.field("capex"), graph, available, CATALOGUE,
                            "utility", ticker="NOT-IN-THE-REGISTER")

    assert with_register.method == STATEMENT_LEAF_SUM
    assert len(with_register.children) == 2
    assert without.method != STATEMENT_LEAF_SUM
    print("\n=== SANITY CHECK: option A vs option B on one graph ===")
    print(f"  registered   : {with_register.method}, "
          f"{[c for c, _ in with_register.children]}")
    print(f"  unregistered : {without.method}, concept {without.concept}")
    print("  OK: Without a per-filer declaration the standard half is refused, not halved.")


def test_a_declared_but_UNREPORTED_extension_does_not_refuse():
    """The DTE subsidiary trap, from the other side.

    DTE declares `us-gaap:PaymentsToAcquirePropertyPlantAndEquipment` in its node but tags
    it ONLY dimensioned to `dte:DTEElectricMember` -- the subsidiary registrant, $3,686M
    against a $4,429M group. `entity_scope.consolidated_facts` drops it, so it is absent
    from `available`. It must therefore neither be summed NOR trigger a refusal: an element
    nobody reported contributes nothing and hides nothing.
    """
    graph = ArcGraph(_arcs([
        ("PlantAndEquipmentExpendituresUtility", "dte", _INVEST, -1.0, _CF_ROLE),
        ("SomeUnreportedExtension", "dte", _INVEST, -1.0, _CF_ROLE),
    ]))
    resolution = resolve_field(
        CATALOGUE.field("capex"), graph,
        frozenset({"PlantAndEquipmentExpendituresUtility"}), CATALOGUE, "utility",
        ticker="DTE")

    assert resolution.method == STATEMENT_LEAF_SUM
    assert [c for c, _ in resolution.children] == ["PlantAndEquipmentExpendituresUtility"]
    print("\n=== SANITY CHECK: an unreported sibling is not a partial-leaf risk ===")
    print(f"  summed {[c for c, _ in resolution.children]}")
    print("  OK: Relaxing the dimensional filter to 'fix' capex would store the "
          "SUBSIDIARY's number, 17% low and entirely plausible.")


def test_a_declared_total_still_beats_the_leaf_sum():
    """Route 3b is a FALLBACK and must stay one, or it silently re-bases every filer that
    already resolves. AAPL and VLO both declare `DepreciationDepletionAndAmortization` /
    `DepreciationAmortizationAndAccretionNet` and must keep them."""
    graph = ArcGraph(_arcs([
        ("DepreciationDepletionAndAmortization", "us-gaap", _OPERATE, 1.0, _CF_ROLE),
        ("Depreciation", "us-gaap", _OPERATE, 1.0, _CF_ROLE),
    ]))
    available = frozenset({"DepreciationDepletionAndAmortization", "Depreciation"})

    resolution = resolve_field(CATALOGUE.field("depAmort"), graph, available, CATALOGUE)

    assert resolution.method == LINKBASE_TOTAL
    assert resolution.concept == "us-gaap:DepreciationDepletionAndAmortization"
    print("\n=== SANITY CHECK: route order ===")
    print(f"  aggregate present -> {resolution.method} on {resolution.concept}")
    print("  OK: The leaf route did not pre-empt the filer's own declared aggregate.")


def test_insurer_capex_absence_is_structural_but_capex_stays_RESOLVABLE():
    """The insurer cell is NOT the bank cell, and the difference is measured.

    `bank.dc_code` short-circuits capex before resolution because all 6 swept banks tag
    nothing. The insurer regime cannot do that: PGR tags
    `PaymentsToAcquirePropertyPlantAndEquipment` in **63 of 63** filings ($65-364M, on
    every duration shape), while AFL, MET and CB tag no capex line at all in 63 filings
    each. One regime, two correct answers -- so the code attaches to the ABSENCE
    (`dc_code_when_absent`) rather than to the field.
    """
    tags_it = ArcGraph(_arcs([
        ("PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap", _INVEST, -1.0, _CF_ROLE),
    ]))
    tags_nothing = ArcGraph(_arcs([
        ("PaymentsToAcquireAvailableForSaleSecuritiesDebt", "us-gaap", _INVEST, -1.0,
         _CF_ROLE),
    ]))

    pgr = resolve_field(CATALOGUE.field("capex"), tags_it,
                        frozenset({"PaymentsToAcquirePropertyPlantAndEquipment"}),
                        CATALOGUE, "insurer", ticker="PGR")
    met = resolve_field(CATALOGUE.field("capex"), tags_nothing,
                        frozenset({"PaymentsToAcquireAvailableForSaleSecuritiesDebt"}),
                        CATALOGUE, "insurer", ticker="MET")

    assert pgr.resolved and pgr.method == STATEMENT_LEAF_SUM
    assert met.method == UNRESOLVED and met.dc_code == "not_applicable"
    print("\n=== SANITY CHECK: insurer capex, both answers ===")
    print(f"  PGR-shaped : {pgr.method} on {[c for c, _ in pgr.children]}")
    print(f"  MET-shaped : {met.method} / {met.dc_code}")
    print("  OK: A regime-wide `dc_code` would have deleted PGR's 63-of-63 real capex.")


def test_bank_capex_is_still_short_circuited_not_merely_absence_coded():
    """The bank cell must NOT drift to the insurer treatment. All 6 swept banks tag no
    capex, `freeCashflow` is null for them BY DESIGN, and a bank that happened to tag a
    PP&E line must not start resolving one -- the register's own measurement shows 3 of 8
    tagging banks are sporadic, which is what makes a bank TTM silently mix tagged and
    untagged quarters."""
    graph = ArcGraph(_arcs([
        ("PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap", _INVEST, -1.0, _CF_ROLE),
    ]))
    resolution = resolve_field(
        CATALOGUE.field("capex"), graph,
        frozenset({"PaymentsToAcquirePropertyPlantAndEquipment"}), CATALOGUE, "bank")

    assert resolution.method == UNRESOLVED
    assert resolution.dc_code == "not_applicable"
    assert not resolution.children
    print("\n=== SANITY CHECK: bank capex short-circuit intact ===")
    print(f"  even with a reported PP&E line -> {resolution.method} / {resolution.dc_code}")
    print("  OK: Route 3b did not reopen a field a regime declares meaningless.")


def test_a_field_with_no_any_of_is_untouched_by_route_3b():
    """The blast-radius test. Only `capex` and `depAmort` declare `roll_up.any_of`, so
    `shortTermDebt`'s route-3 behaviour must be bit-identical -- and `totalRevenue`'s
    coverage must be too, which the acceptance sweep asserts on real filings."""
    graph = ArcGraph(_arcs([
        ("LongTermDebtCurrent", "us-gaap", "DebtCurrent", 1.0, _CF_ROLE),
        ("ShortTermBorrowings", "us-gaap", "DebtCurrent", 1.0, _CF_ROLE),
    ]))
    available = frozenset({"LongTermDebtCurrent", "ShortTermBorrowings"})

    resolution = resolve_field(CATALOGUE.field("shortTermDebt"), graph, available,
                               CATALOGUE)

    assert resolution.method == "linkbase_sum"
    print("\n=== SANITY CHECK: no `any_of`, no route 3b ===")
    print(f"  shortTermDebt -> {resolution.method}")
    print("  OK: Route 3b is opt-in per field.")


def test_the_register_never_contradicts_itself():
    """A concept declared both a leaf and not-a-leaf would be resolved silently in favour
    of the leaf. The loader raises instead; this asserts the register as shipped is clean,
    and that every extension declaration carries written evidence."""
    for ticker, block in CATALOGUE.ticker_exceptions.items():
        for field, entry in block.items():
            leaves = {c for g in entry.get("leaves", []) for c in g}
            assert not (leaves & set(entry.get("not_leaves", []))), f"{ticker}/{field}"
            if leaves:
                assert entry.get("evidence"), f"{ticker}/{field} has no evidence"
            assert entry.get("verified"), f"{ticker}/{field} has no verified date"
    n_leaf = sum(1 for b in CATALOGUE.ticker_exceptions.values() for e in b.values()
                 if e.get("leaves"))
    print("\n=== SANITY CHECK: the extension register ===")
    print(f"  {len(CATALOGUE.ticker_exceptions)} filers, {n_leaf} with declared leaves, "
          "all carrying evidence")
    print("  OK: No concept is both a leaf and not a leaf.")


# --------------------------------------------------------------------------- #
# Real filings: the numbers                                                   #
# --------------------------------------------------------------------------- #
#: (ticker, GICS triple, field, the FY figure the filer's own statement supports).
#: Every expectation is a measured fact from the live 10-K, verified 2026-08-23.
_GROUND_TRUTH = [
    # DTE is the strongest case available: its own arithmetic confirms the leaf sum.
    # `dte:PaymentsToAcquireProductiveAssetsIncludingPaymentsToAcquireBusinessesNetOf
    # CashAcquired` $4,639M less `PaymentsToAcquireBusinessesNetOfCashAcquired` $210M
    # = $4,429M, exactly the sum of the two extension legs.
    ("DTE", ("Utilities", "Utilities", "Multi-Utilities"), "capex", 4_429e6),
    ("EOG", ("Energy", "Energy", "Oil & Gas Exploration & Production"), "capex", 6_594e6),
    ("MAA", ("Real Estate", "Equity Real Estate Investment Trusts (REITs)",
             "Multi-Family Residential REITs"), "capex", 765.7e6),
    ("MSFT", ("Information Technology", "Software & Services", "Systems Software"),
     "depAmort", 38_534e6),
    ("GOOGL", ("Communication Services", "Media & Entertainment",
               "Interactive Media & Services"), "depAmort", 21_136e6),
    # The regression guard: AAPL must keep its declared AGGREGATE and must NOT fall to the
    # PP&E-note `Depreciation`, which is $8,000M -- 31.6% low.
    ("AAPL", ("Information Technology", "Technology Hardware & Equipment",
              "Technology Hardware, Storage & Peripherals"), "depAmort", 11_698e6),
    ("VLO", ("Energy", "Energy", "Oil & Gas Refining & Marketing"), "depAmort", 3_158e6),
    # 4c.1 + the CSCO register entry, together. Before them route 1 took
    # `us-gaap:DepreciationDepletionAndAmortization` at EXACTLY $700,000,000 with
    # `decimals=-8` in FY2023, FY2024 and FY2025 alike -- one rounded narrative figure
    # repeated three years. The cash-flow line is the extension
    # `csco:DepreciationAmortizationAndOther`, and this asserts the number it produces.
    ("CSCO", ("Information Technology", "Technology Hardware & Equipment",
              "Communications Equipment"), "depAmort", 2_811e6),
]


@pytest.fixture(scope="module")
def edgar_ready() -> bool:
    if not os.getenv("SEC_USER_AGENT", "").strip():
        pytest.skip("SEC_USER_AGENT unset -- real-filing leaf-sum tests need EDGAR")
    return True


@pytest.fixture(scope="module")
def latest_rows(edgar_ready) -> dict:
    """`filing_rows` on one real 10-K per ground-truth ticker. Module-scoped: each
    `filing.xbrl()` costs 1.4-5.8 s and this file needs seven of them."""
    from edgar import Company, set_identity

    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import filing_rows
    set_identity(os.getenv("SEC_USER_AGENT"))

    out: dict[str, pd.DataFrame] = {}
    for ticker, (sector, group, sub), _, _ in _GROUND_TRUTH:
        if ticker in out:
            continue
        try:
            company = Company(ticker)
            filing = next(f for f in company.get_filings(form="10-K")
                          if not str(f.form).upper().endswith("/A"))
            rows = filing_rows(ticker, str(company.cik), filing, CATALOGUE,
                               {"sector": sector, "industry_group": group,
                                "sub_industry": sub})
        except Exception as exc:                            # noqa: BLE001
            pytest.skip(f"EDGAR unreachable for {ticker}: {exc}")
        out[ticker] = pd.DataFrame(rows)
    return out


@pytest.mark.parametrize("ticker,gics,field,expected", _GROUND_TRUTH,
                         ids=[f"{t}-{f}" for t, _, f, _ in _GROUND_TRUTH])
def test_the_latest_10k_reproduces_the_filers_own_figure(latest_rows, ticker, gics, field,
                                                         expected):
    rows = latest_rows[ticker]
    blk = rows[(rows.field == field) & rows.value.notna()
               & (rows.duration_type == "annual")]
    assert not blk.empty, f"{ticker} {field}: no annual value at all"
    got = float(blk.sort_values("period_end").iloc[-1].value)
    method = blk.sort_values("period_end").iloc[-1].resolution_method
    assert abs(got - expected) / expected < 0.005, (
        f"{ticker} {field}: {got:,.0f} vs expected {expected:,.0f} (via {method})")
    print(f"\n  {ticker:6s} {field:9s} {got/1e6:>10,.1f}M  via {method}")


def test_aapl_and_vlo_keep_the_aggregate_and_never_the_note_leaf(latest_rows):
    """§4b.4's refuted design, asserted as a standing regression guard. The note-level
    `Depreciation` is $8,000M for AAPL and $2,300M for VLO against declared aggregates of
    $11,698M and $3,158M."""
    for ticker, concept in (("AAPL", "us-gaap:DepreciationDepletionAndAmortization"),
                            ("VLO", "us-gaap:DepreciationAmortizationAndAccretionNet")):
        rows = latest_rows[ticker]
        blk = rows[(rows.field == "depAmort") & rows.value.notna()]
        assert not blk.empty, ticker
        assert set(blk.resolution_method) == {LINKBASE_TOTAL}, ticker
        assert set(blk.source_concept) == {concept}, ticker
        print(f"\n  {ticker:6s} depAmort via {LINKBASE_TOTAL} on {concept}")
    print("  OK: Neither fell to the note-level leaf sum.")


def test_dte_does_not_store_its_SUBSIDIARYs_capex(latest_rows):
    """The trap named in §4b.2. DTE tags
    `us-gaap:PaymentsToAcquirePropertyPlantAndEquipment` at $3,686M, but only dimensioned
    to `dte:DTEElectricMember`. Storing it would be 17% low and entirely plausible."""
    rows = latest_rows["DTE"]
    blk = rows[(rows.field == "capex") & rows.value.notna()]
    children: set[str] = set()
    for blob in blk.roll_up_children.dropna():
        children.update(concept for concept, _weight in json.loads(blob))
    assert children == {"PlantAndEquipmentExpendituresUtility",
                        "PlantAndEquipmentExpendituresNonUtility"}
    assert "PaymentsToAcquirePropertyPlantAndEquipment" not in children
    print(f"\n  DTE capex children: {sorted(children)}")
    print("  OK: The subsidiary-dimensioned PP&E line was never admitted.")


def test_a_composed_field_with_NO_component_at_all_is_still_reason_coded():
    """Not route 3b, but the same criterion, and the acceptance sweep is how it was found.

    `_compose` returned `(values, None)` when NO component field resolved anywhere in the
    filing, which is an UNEXPLAINED null -- and "zero unexplained nulls" is only checkable
    if it is literally zero. Measured on the Phase-4 ledgers: **79 rows across 7 tickers**,
    all `ppeNet` and `totalDebt`.

    The two reasons are genuinely different and must stay separable: `incomplete_roll_up`
    means the filer reported something and a load-bearing leg is missing; `not_disclosed`
    means it reported none of the legs.
    """
    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import _compose

    spec = CATALOGUE.field("ppeNet")
    components = tuple(f for f in spec.roll_up() if f in CATALOGUE.fields)
    nothing_values, nothing_code = _compose(spec, components, {})
    one_leg = {components[0]: {("2025", "FY", "annual", None, None): {"value": 1.0}}}
    _, partial_code = _compose(spec, components, one_leg)

    assert nothing_values == {} and nothing_code == "not_disclosed"
    assert partial_code == "incomplete_roll_up"
    print("\n=== SANITY CHECK: a composed field's null always has a reason ===")
    print(f"  no component resolved -> {nothing_code}")
    print(f"  one of two resolved   -> {partial_code}")
    print("  OK: the two absences stay distinguishable.")


def test_the_recorded_anchor_and_role_come_from_the_WINNING_ARC():
    """Provenance, and it is not cosmetic: `role_uri` is documented as corroborating the
    regime, because FASB's role URIs name the statement template.

    `graph.role_of` and `graph.parent_of` both answer across ALL roles and take the first
    arc, so for a concept a filer presents twice they return the wrong one. UNP declares
    `us-gaap:Depreciation` under the income statement's `CostsAndExpenses` AND under the
    cash-flow operating node; reading `parent_of` gave `CostsAndExpenses` and an
    income-statement role for a value that came off the cash-flow statement -- provenance
    that contradicts the guard which admitted it.
    """
    graph = ArcGraph(_arcs([
        # income statement FIRST, so a first-arc-wins lookup gets it wrong.
        ("Depreciation", "us-gaap", "CostsAndExpenses", 1.0,
         "http://x/role/ConsolidatedStatementsOfOperations"),
        ("Depreciation", "us-gaap", _OPERATE, 1.0, _CF_ROLE),
    ]))
    resolution = resolve_field(CATALOGUE.field("depAmort"), graph,
                               frozenset({"Depreciation"}), CATALOGUE)

    assert resolution.method == STATEMENT_LEAF_SUM
    assert resolution.anchor == _OPERATE
    assert resolution.role_uri == _CF_ROLE
    print("\n=== SANITY CHECK: provenance follows the arc, not the concept ===")
    print(f"  anchor   {resolution.anchor}")
    print(f"  role_uri {resolution.role_uri}")
    print("  OK: the income-statement arc listed first did not win the provenance.")
