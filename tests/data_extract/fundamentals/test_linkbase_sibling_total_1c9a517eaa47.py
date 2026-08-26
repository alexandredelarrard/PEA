"""
Cluster `1c9a517eaa47` -- MCD `capex`, 55 findings across 10 agreeing checks.

MCD tags `us-gaap:PaymentsToAcquireProductiveAssets` for the line **"Purchases of
restaurant and other businesses"** ($540.9M FY2019) and tags its real capex,
**"Capital expenditures"** ($2,393.7M), as `us-gaap:PaymentsToAcquirePropertyPlantAndEquipment`
-- declaring BOTH as children of `NetCashProvidedByUsedInInvestingActivities` at weight
-1.0. Route 1 accepted the catalogue's `total_concept` on `graph.knows` alone ("is it in
the statement structure at all"), so 67 rows of MCD's franchisee-acquisition line were
stored as capex, understating FY2017 by 96% ($77.0M against $1,853.7M).

Ground truth read off the filed statements, not `companyfacts`:
  * 0000063908-18-000010 (FY2017 10-K) -- ProductiveAssets $77.0M "Purchases of restaurant
    businesses" beside PP&E $1,853.7M "Capital expenditures".
  * 0000063908-20-000022 (FY2019 10-K) -- $540.9M beside $2,393.7M.
  * 0000063908-17-000025 (Q1 2017 10-Q) -- $3.1M beside $427.7M.

The fix is `xbrl_linkbase.sibling_leg`, and it takes TWO conditions. The second one is the
reason this file exists: condition 1 alone ALSO fires on AAPL and SWKS, which declare
`PaymentsToAcquireIntangibleAssets` beside the total and are entirely correct, and would
have cut AAPL's FY2014 capex from $9,571M to $242M. Every test below is synthetic
known-truth (docs/testing.md: parsing math gets fixtures), because a real filing cannot be
made to disagree with itself on demand -- the real-filing evidence is the measurement
recorded in `sibling_leg`'s docstring.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals import entity_scope as scope
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    LINKBASE_TOTAL, STATEMENT_LEAF_SUM, ArcGraph, resolve_field)

CATALOGUE = load_catalogue("./configs")

TOTAL = "PaymentsToAcquireProductiveAssets"
PPE = "PaymentsToAcquirePropertyPlantAndEquipment"
INTANGIBLES = "PaymentsToAcquireIntangibleAssets"
INVESTING = "NetCashProvidedByUsedInInvestingActivities"

#: The role must READ as a cash-flow statement: route 3b admits a leaf only beneath
#: `capex.roll_up.anchor` on a role matching `anchor_role`, so a placeholder URI would make
#: every fixture here silently resolve by a different route than production does.
_CF_ROLE = "http://x/role/ConsolidatedStatementOfCashFlows"

_ARC_COLS = ["concept", "concept_taxonomy", "parent_concept", "parent_taxonomy",
             "weight", "role_uri", "menucat", "is_abstract", "arc_filter"]


def _arcs(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(concept, parent, weight) -> the arc frame `statement_arcs` returns."""
    return pd.DataFrame(
        [{"concept": c, "concept_taxonomy": "us-gaap", "parent_concept": p,
          "parent_taxonomy": "us-gaap", "weight": w, "role_uri": _CF_ROLE,
          "menucat": "Statements", "is_abstract": False, "arc_filter": "both"}
         for c, p, w in rows],
        columns=_ARC_COLS)


def _resolve(arcs: pd.DataFrame, magnitudes: dict[str, float]):
    """`capex` resolved against a synthetic filing reporting exactly `magnitudes`."""
    return resolve_field(
        CATALOGUE.field("capex"), ArcGraph(arcs), frozenset(magnitudes), CATALOGUE,
        regime=None, magnitudes=magnitudes, ticker="TEST")


# --------------------------------------------------------------------------- #
# The defect                                                                   #
# --------------------------------------------------------------------------- #
def test_mcd_shape_total_beside_a_larger_leg_is_refused_as_the_total():
    """MCD: the declared "total" is SMALLER than a component FASB puts inside it.

    A concept FASB defines as PP&E + software + intangibles cannot be smaller than the
    PP&E it contains, so the element is being used for another line -- and the filer has
    already said as much structurally by declaring the two as siblings.
    """
    resolution = _resolve(
        _arcs([(PPE, INVESTING, -1.0), (TOTAL, INVESTING, -1.0)]),
        {TOTAL: 540_900_000.0, PPE: 2_393_700_000.0})

    assert resolution.method == STATEMENT_LEAF_SUM
    assert resolution.children == ((PPE, 1.0),)
    assert resolution.sibling_rejected == ((f"us-gaap:{TOTAL}", f"us-gaap:{PPE}"),)
    assert resolution.dc_code is None, "a refusal must never manufacture a NULL"
    print(f"MCD shape: route={resolution.method}, capex now reads the $2,393.7M "
          f"'Capital expenditures' leg, not the $540.9M restaurant-acquisition line")


# --------------------------------------------------------------------------- #
# The regression the defect's obvious fix would have caused                     #
# --------------------------------------------------------------------------- #
def test_aapl_shape_total_beside_a_smaller_leg_stays_on_route_1():
    """AAPL / SWKS: a sibling leg SMALLER than the total is an ordinary extra line.

    This is the guard that makes the rule safe. AAPL's `PaymentsToAcquireProductiveAssets`
    IS its $9,571M "Payments for acquisition of property, plant and equipment"
    (0001193125-14-383437), with intangibles a separate, smaller line. A structure-only
    rule refuses it and hands route 3b the intangibles leg alone -- a 97.5% understatement,
    and the same class of self-inflicted damage as the 745 correct rows once nulled by
    over-strict Q4 guards.
    """
    resolution = _resolve(
        _arcs([(TOTAL, INVESTING, -1.0), (INTANGIBLES, INVESTING, -1.0)]),
        {TOTAL: 9_571_000_000.0, INTANGIBLES: 1_107_000_000.0})

    assert resolution.method == LINKBASE_TOTAL
    assert resolution.concept == f"us-gaap:{TOTAL}"
    assert resolution.sibling_rejected == ()
    print(f"AAPL shape: route={resolution.method}, capex stays ${9_571:,}M and is NOT "
          f"replaced by the ${1_107:,}M intangibles leg")


# --------------------------------------------------------------------------- #
# The three ways the rule must decline to fire                                 #
# --------------------------------------------------------------------------- #
def test_a_leg_declared_beneath_the_total_is_a_real_roll_up():
    """`has_descendant` wins over `has_sibling`: a filer that declares the leg BENEATH the
    total is rolling up exactly as FASB intends, whatever the magnitudes say."""
    resolution = _resolve(
        _arcs([(TOTAL, INVESTING, -1.0), (PPE, TOTAL, 1.0)]),
        {TOTAL: 540_900_000.0, PPE: 2_393_700_000.0})

    assert resolution.method == LINKBASE_TOTAL
    assert resolution.sibling_rejected == ()
    print(f"declared roll-up: route={resolution.method} -- containment beats magnitude")


def test_an_unreported_leg_is_not_evidence():
    """The leg must be REPORTED. A structural declaration with no fact behind it proves
    nothing, and 10 of the 12 measured filers are in exactly this state."""
    resolution = _resolve(
        _arcs([(TOTAL, INVESTING, -1.0), (PPE, INVESTING, -1.0)]),
        {TOTAL: 1_276_000_000.0})

    assert resolution.method == LINKBASE_TOTAL
    assert resolution.sibling_rejected == ()
    print(f"unreported leg: route={resolution.method} -- silence is not evidence")


def test_a_refusal_never_fires_without_an_answer_to_hand_off_to():
    """Route 3b must have leaves. With the anchor node absent there is no leaf sum, so the
    rule stands down rather than turn a wrong number into a NULL -- which is exactly what
    happens on EQIX, whose unclassified `eqix:` extension makes route 3b refuse with
    `partial_leaf_sum`.
    """
    resolution = _resolve(
        _arcs([(TOTAL, "SomeOtherParent", -1.0), (PPE, "SomeOtherParent", -1.0)]),
        {TOTAL: 23_993_000.0, PPE: 363_990_000.0})

    assert resolution.dc_code is None
    assert resolution.method == LINKBASE_TOTAL
    print(f"no leaf sum available: route={resolution.method}, dc_code={resolution.dc_code} "
          f"-- the value is kept rather than nulled")


# --------------------------------------------------------------------------- #
# The filing-level summary the rule reads                                      #
# --------------------------------------------------------------------------- #
def test_peak_magnitudes_is_absolute_and_period_agnostic():
    """`peak_magnitudes` collapses a filing's many periods to one number per concept, on
    ABSOLUTE value -- a 10-Q tags quarterly, year-to-date and prior-year comparatives for
    the same concept, so there is no single period to compare on."""
    facts = pd.DataFrame({
        "concept": [f"us-gaap:{PPE}", f"us-gaap:{PPE}", f"us-gaap:{TOTAL}"],
        "numeric_value": [427_700_000.0, -1_853_700_000.0, 3_100_000.0],
    })
    peaks = scope.peak_magnitudes(facts)

    assert peaks[PPE] == 1_853_700_000.0
    assert peaks[TOTAL] == 3_100_000.0
    assert scope.peak_magnitudes(pd.DataFrame()) == {}
    print(f"peak_magnitudes: {PPE}={peaks[PPE]:,.0f} (absolute, across 2 periods), "
          f"{TOTAL}={peaks[TOTAL]:,.0f}")
