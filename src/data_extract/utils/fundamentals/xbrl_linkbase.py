"""
xbrl_linkbase.py (src/data_extract/utils/fundamentals/xbrl_linkbase.py)
--------------------------------------------------------------------------
The resolution primitive of the fundamentals rebuild: **read the roll-up the filer itself
declared**, instead of guessing which of several candidate us-gaap concepts it meant.

Why this exists. The old stack resolved a KPI by taking the highest-priority hit from an
ordered candidate list, on the premise that the candidates are substitutes. Measured on
this repo's own 7.8M-fact dump they are substitutes only 30-56% of the time, because
FASB's element definitions make several of them a superset and its subset, or two disjoint
legs of one total. The calculation linkbase is not a guess -- it is the filer's own
statement of which concept is its total and which concepts foot into it, with signed
weights.

The routes a field can resolve by, in the order they are tried (`Resolution.method`):

  1. ``linkbase_total``  a catalogue candidate is BOTH declared in the filer's statement
                         structure and actually reported -- use it, UNLESS the filer
                         declares it as a SIBLING of one of this field's own roll-up legs,
                         which is the filer stating that its "total" is just another leaf
                         (MCD tags `PaymentsToAcquireProductiveAssets` as "Purchases of
                         restaurant and other businesses", beside its real "Capital
                         expenditures"). See `sibling_leg`.
  2. ``linkbase_root``   no candidate is reported, so discover the filer's own top node
                         STRUCTURALLY (see below). Opt-in per field via
                         ``linkbase_root_discovery``.
  3. ``linkbase_sum``    no total, but the declared legs are reported -- sum them WITH
                         THEIR WEIGHTS (22% of Statements arcs carry -1.0, so the sign is
                         load-bearing, not defensive).
  3b. ``statement_leaf_sum``
                         no total and no complete declared roll-up, but the filer DOES
                         declare the field's constituent LINES beneath an anchor node of a
                         named statement -- sum the ones it reports. This is the `capex` /
                         `depAmort` route; see `_leaf_sum` for why route 3 could not do it
                         and for the three guards that make it safe.
  4. ``field_sum``       composed of other CATALOGUE FIELDS rather than of concepts
                         (`totalDebt`, `ppeNet`). Completed by the caller once those
                         fields are resolved; this module only reports the route.
  5. ``tag_primary``     the catalogue's HIGHEST-PRIORITY reported concept was taken, but the
                         filer's calculation linkbase has no arc for it. Expected and benign
                         for anything that is not a roll-up: a calculation arc exists only
                         where a filer declares a total-and-components relationship, so a
                         leaf (`goodwill`, `inventory`) or a cover-page `dei:` tag
                         (`sharesOutstanding`) can never have one.
  6. ``tag_fallback``    every linkbase route failed AND the winner was not the top-priority
                         concept -- the old resolver's behaviour, and the only route whose
                         rate is evidence about this design.
  7. ``unresolved``      no value by any route; `dc_code` says why. Not a route at all, and
                         excluded from any routing rate.

`tag_primary` and `tag_fallback` were ONE label until it was measured: pooling them (and
pooling in the reason-coded non-resolutions) reported 27.8% against a 20% architecture gate
and forced a judgement call about which fields "should" count. Split, the gate applies
literally to `tag_fallback` alone and needs no exclusions argued for.

**Why route 2 exists, and why a tag list can never replace it.** Measured on live 10-Ks:

  * **APA** reports its revenue total as ``apa:RevenuesAndOther`` -- a **company extension
    element**, $9.220 B FY2025 -- declared in its linkbase as the +1.0 child of the pretax
    node. Meanwhile ``us-gaap:Revenues`` exists in the same filing with **84 facts, every
    one of them dimensioned** (segment detail). So the old resolver could only pick a
    segment row or fall through, which is how APA ended up on an element it tags as
    literally $0.00 for 19 consecutive rows. No candidate list can ever contain
    ``apa:RevenuesAndOther``: it is company-private. Only the structure finds it.
  * **DTE** reports ``RegulatedAndUnregulatedOperatingRevenue`` and declares it as a
    PARENTLESS root of its income-statement role (its `OperatingIncomeLoss` arc set
    contains only ``-1 CostsAndExpenses``, the revenue arc being omitted -- a common and
    permitted filer omission).

Both are the same failure of the same wrong primitive, and both are repaired by asking the
structure rather than a list.

**Why there is no generic "climb to the parent".** An earlier version of this module
climbed from a matched candidate to its parent while the parent was a pure aggregation and
was not another field's declared total. Measured, it over-climbed catastrophically on real
filings -- `cash` reached ``AssetsCurrent``, `shortTermDebt` reached
``LiabilitiesAndStockholdersEquity``, `ppeNet` reached ``AssetsNoncurrent``, `netIncome`
reached ``ComprehensiveIncomeNetOfTax`` -- because the catalogue names only a handful of
concepts and therefore cannot supply a dense enough boundary. Balance-sheet lines do not
need climbing (their concepts are stable); revenue does, and route 2 handles revenue
directly and only. Keeping the general climb would have traded one silent-wrong-number
mechanism for another.

**The filer's own statement beats a bare tag hit (4c.1), in two halves.** A concept that is
not the field's face-statement line is a silent basis error rather than a missing value, and
measurement shows the mechanism is TWO distinct things:

  * `is_note_only` -- the concept IS declared, and every role it is declared on is a
    non-statement role. Applied to routes 1 and 5, on POSITIVE evidence only, never on
    silence: a leaf (`goodwill`) or a `dei:` cover-page tag can never carry a calculation arc
    and `tag_primary` is its correct home. It never costs a value either -- if the note-level
    concept is the filer's whole answer, `resolve_field` puts it back and flags
    `role_only_retained`. Measured on 13 swept tickers it fires **0 times**, so it is a
    standing guard against a class rather than a repair of a live one.
  * **the declaredness test, which is the one that fires.** Where the candidate carries no
    calculation arc at all and the filer DOES declare this field's own statement lines under
    its anchor node, route 3b wins. CSCO tags
    `us-gaap:DepreciationDepletionAndAmortization` at exactly **$700,000,000, `decimals=-8`,
    in fiscal 2023, 2024 and 2025 alike** -- a rounded narrative figure, the same number
    three years running -- while its cash-flow line is `csco:DepreciationAmortizationAndOther`
    at **$2,811M**, `decimals=-6`. No role test can see that (there is no arc) and no tag list
    can name the truth (it is a company extension). Scope is deliberately tiny: only `capex`
    and `depAmort` declare `roll_up.any_of` at all, so `sharesOutstanding` -- the XOM case
    that set the "priority dominates linkbase presence" rule -- cannot be touched.

Cost: ``calculation_linkbase()`` measured at **0.003-0.006 s** against ``xbrl()``'s
1.4-5.8 s on the same filing -- ~0.1%. Effectively free on a substrate already paid for,
which is what makes reading it per filing affordable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import cached_property

import pandas as pd

from src.data_extract.utils.fundamentals.kpi_catalogue import Catalogue, FieldSpec

#: The `menucat` value selecting face-statement arcs. edgartools' own docstring says this
#: column is 'S'/'D'/'N'; that is wrong -- the values come from FilingSummary.xml and are
#: `Statements` / `Details` / `Uncategorized` / None. Filtering on the wrong one silently
#: yields no arcs.
STATEMENTS_MENUCAT = "Statements"

#: Role URIs that are NOT a face statement. An EXCLUSION test, deliberately: filers name
#: their statement roles freely (`STATEMENTOFCONSOLIDATEDOPERATIONS`,
#: `ConsolidatedStatementsofIncomeSouthernCalculation`), so any positive pattern is a
#: guess, while the things that are not statements are named from a short vocabulary.
#:
#: Calibrated against `menucat` as ground truth on 28 filings x 26 tickers: **0 arcs
#: missing, 99 extra, all 99 verified face statements**. Two terms are load-bearing and
#: must not be dropped:
#:
#:   * `detail` SINGULAR -- footnote roles end `...Detail`, not `...Details`. The audit's
#:     first pattern used `details` and admitted 136 junk arcs on AFL and 203 on MTB.
#:   * `schedule` -- Reg S-X **Schedule I / II parent-company-only condensed financials**
#:     (`ScheduleIiCondensedFinancialInformationOfRegistrant...BalanceSheets`) contain none
#:     of the other words and look exactly like face statements. They are PARENT-ONLY, so
#:     admitting them would silently mix unconsolidated numbers into consolidated fields --
#:     the one failure mode nothing downstream could catch. PGR and AFL both ship one.
NON_STATEMENT_ROLE = re.compile(
    r"detail|disclosure|polic|parenthetical|schedule|tables?$|uncategor|highlight",
    re.IGNORECASE)

#: A role URI that names the SEGMENT-INFORMATION note. A strict subset of
#: `NON_STATEMENT_ROLE`, and separated from it because the two carry different verdicts: an
#: ordinary note-level fact is a real amount on a NARROW basis, so `resolve_field` relaxes
#: the note guard rather than return a null. A segment-note total is a DIFFERENT MEASURE --
#: the filer's own management aggregate, reconciled to a statement line rather than equal to
#: one -- so relaxing to it silently swaps the measure and nothing downstream can tell.
#:
#: ORCL is the measured case. Its `us-gaap:GrossProfit` is declared on exactly one arc,
#: `DisclosureSEGMENTINFORMATIONRECONCILIATIONDetails`, labelled "Margin": the total margin
#: for its three reportable segments (FY2018 21,825 + 1,807 + 655 = **24,287**, which is
#: precisely what this pipeline stored as consolidated gross profit for 26 rows). Oracle
#: presents no gross-profit and no total-cost-of-revenue subtotal on the face of its income
#: statement at all, so the correct answer for the field is a reason-coded NULL.
#:
#: Measured cost of the narrowness, on a 25-filing sample of the 54-ticker roster: of 192
#: resolved rows whose concept is note-only in the FULL linkbase, **3** sit on a segment
#: role. The other 189 are `WeightedAverageNumberOfSharesOutstandingBasic`,
#: `ShareBasedCompensation` and `CashCashEquivalentsRestrictedCash...` -- EPS-note and
#: cash-flow-supplement elements whose arcs genuinely live in notes and whose values are
#: right. Banning note-only concepts outright would have nulled all 192, which is the
#: 745-correct-rows-nulled precedent repeating itself.
SEGMENT_ROLE = re.compile(r"segment", re.IGNORECASE)

#: Which of the two tests admitted an arc: `menucat`, `role_uri`, or `both`. Carried so the
#: two populations stay separable -- the role test is a naming heuristic over filer-authored
#: strings and its output must remain auditable, not merged invisibly into the SEC's own
#: categorisation.
ARC_FILTER = "arc_filter"

#: A statement role that carries a REVENUE top line. Used by `discover_root` to reject the
#: cash-flow, comprehensive-income and balance-sheet roots that a parentless-root search
#: otherwise picks at random.
#:
#: Both subtractions are load-bearing and both were found by MEASUREMENT, not by reading:
#:
#:   * `comprehensive` -- `ConsolidatedStatementsOfComprehensiveIncome` matches `income`.
#:   * `cash flow` -- the FASB standard role for a bank's cash-flow statement is
#:     `StatementOfCashFlowsIndirectDepositBasedOperations`, which matches **operations**.
#:     Without this, MTB stored 27 rows of its cash-flow period-increase total as revenue
#:     and USB another 9, all in the 2012-2016 filings that 3c.1 had just made readable.
INCOME_STATEMENT_ROLE = re.compile(r"operations|income|earnings", re.IGNORECASE)
NOT_INCOME_STATEMENT_ROLE = re.compile(r"comprehensive|cash[\s_-]*flow", re.IGNORECASE)

LINKBASE_TOTAL = "linkbase_total"
LINKBASE_ROOT = "linkbase_root"
LINKBASE_SUM = "linkbase_sum"
STATEMENT_LEAF_SUM = "statement_leaf_sum"
FIELD_SUM = "field_sum"
TAG_PRIMARY = "tag_primary"
TAG_FALLBACK = "tag_fallback"
UNRESOLVED = "unresolved"

#: `dc_code` for a composed field whose roll-up is missing a component the catalogue marks
#: as load-bearing. Distinct from `not_disclosed`: the filer DID report something, we simply
#: cannot make the field out of it without inventing the missing part.
INCOMPLETE_ROLL_UP = "incomplete_roll_up"

#: `dc_code` for the route-3b partial-leaf refusal: the filer declares leaves of this field
#: under the anchor node, but at least one of them is a COMPANY EXTENSION the catalogue
#: cannot classify, so any sum of the rest would be silently short. Distinct from
#: `not_disclosed` -- the amount IS disclosed, in an element no list can name (see
#: `_leaf_sum` guard 3, and the `by_ticker` register that closes it per filer).
PARTIAL_LEAF_SUM = "partial_leaf_sum"

#: `dc_code` for the segment-role refusal: the filer tags this field's concept, but every
#: calculation arc it carries sits on a SEGMENT-INFORMATION role, so the only number on
#: offer is a segment aggregate rather than the consolidated line. Distinct from
#: `not_disclosed`, which would be a false statement about the filing -- the concept IS
#: tagged -- and distinct from `partial_leaf_sum`, where the amount is disclosed and merely
#: unnameable. Here the amount on offer is a DIFFERENT MEASURE, and the null is correct.
SEGMENT_ONLY_CONCEPT = "segment_only_concept"

#: `dc_code` for a field that RESOLVED -- a concept was chosen -- but for which
#: `_materialise` found no period. Nothing upstream calls such a field absent, so without
#: its own code it is the one remaining way to produce a null with no reason. Rare and
#: therefore worth naming: 1 row in 144,131 on the in-sample ledger.
NO_USABLE_PERIOD = "no_usable_period"

#: Routes where the FILER'S OWN declared structure chose the concept.
LINKBASE_METHODS: frozenset[str] = frozenset(
    {LINKBASE_TOTAL, LINKBASE_ROOT, LINKBASE_SUM, STATEMENT_LEAF_SUM})

#: `roll_up.anchor_role` values -> the role-URI pattern that names that statement. An
#: EXCLUSION-free positive test is safe here in a way it is not for `NON_STATEMENT_ROLE`,
#: because the vocabulary for "this is the cash-flow statement" is short and filers do not
#: vary it: measured on 52 tickers x 3,163 filings, every cash-flow role URI on both
#: rosters matches `cash[\s_-]*flow`, including FASB's own
#: `StatementOfCashFlowsIndirectDepositBasedOperations`.
ANCHOR_ROLES: dict[str, re.Pattern[str]] = {
    "cash_flow": re.compile(r"cash[\s_-]*flow", re.IGNORECASE),
    "income_statement": INCOME_STATEMENT_ROLE,
}

#: Standard SUBTOTAL concepts that sit directly above the revenue block on an income
#: statement, nearest-to-revenue first. Route 2 anchors on these: a filer may invent its
#: own revenue element (APA), but subtotals are what the taxonomy is FOR and are extended
#: far less often. `GrossProfit` is first because when it exists its positive child is the
#: revenue line itself (AAPL); the pretax variants are last because more sits between them
#: and revenue.
REVENUE_ANCHORS: tuple[str, ...] = (
    "GrossProfit",
    "OperatingIncomeLoss",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeignCurrencyTransactionsGainLossDerivativeInstrumentsGainLossAndOtherIncomeExpense",
)

#: Concepts that are never a revenue top line even when they sit where one would, because
#: they are non-operating or disposal items a filer parks under the same pretax subtotal
#: (measured on MAA, whose pretax node has four positive children).
NOT_A_TOP_LINE: frozenset[str] = frozenset({
    "OtherNonoperatingIncomeExpense",
    "NonoperatingIncomeExpense",
    "GainLossOnSaleOfPropertyPlantEquipment",
    "GainLossOnSalesOfAssetsAndAssetImpairmentCharges",
    "IncomeLossFromEquityMethodInvestments",
    "InvestmentIncomeInterest",
    # Added after the post-3c re-sweep, each one measured:
    #   VRT 2018-2020 (17 rows) -- the GS Acquisition Holdings SPAC shell's only income is
    #   its trust account's dividends. Genuine, but it is not revenue, and letting it win
    #   is what stopped `zero_only_retained` from keeping the correct 0.
    "InvestmentIncomeDividend",
    #   ETN 2012 (2 rows) -- an unrealised FX line under the pretax node, resolving revenue
    #   to -$149 for a company with a $16 bn top line.
    "ForeignCurrencyTransactionGainLossUnrealized",
    #   GS 2019 (4 rows) -- a bank/broker EXPENSE subtotal that sits legitimately on
    #   `ConsolidatedStatementsOfEarnings`, so no role or period test can reject it. The
    #   bank regime bans it as a CANDIDATE; this bans it as a discovered ROOT, which is
    #   regime-independent -- GS is `broker_dealer`, not `bank`.
    "NoninterestExpense",
    # Added 2026-08-23 by the 52-ticker out-of-sample sweep -- a PRE-EXISTING defect that
    # the in-sample roster could not see, since AXP is out-of-sample only. AXP's 2009-2011
    # linkbases declare these as parentless roots WITH CHILDREN on an income-statement role,
    # and `discover_root`'s widest-aggregation ranking duly took them: **15 rows of revenue
    # at $0.09-$3.40 and 8 rows at AXP's net income**, 23 in all. Every other route-2
    # discovery across the 52 tickers is a genuine revenue line, so this is the whole
    # population.
    #
    # A per-share amount can ALSO be excluded by its unit, and `entity_scope` now does that
    # -- but only for the 5 rows whose `unit_ref` says so. `unit_ref` is a filer-authored ID
    # (`Unit12`, `u000`, `U_iso4217USD`), so the unit oracle cannot carry this and the name
    # list must. A BOTTOM line is monetary and no unit test could ever reject it at all.
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "EarningsPerShareBasicAndDiluted",
    "IncomeLossFromContinuingOperationsPerBasicShare",
    "IncomeLossFromContinuingOperationsPerDilutedShare",
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
})


@dataclass(frozen=True)
class Resolution:
    """How one field resolves for one filing. Period-agnostic on purpose.

    The linkbase describes STRUCTURE, which does not vary across the periods inside one
    filing, while the facts vary per period. Resolving once per (filing, field) and then
    applying the answer to every period is both correct and far less work than resolving
    per period -- the efficiency trap Phase 10 is told to look for.
    """

    field: str
    method: str
    #: The concept whose reported value IS the field, NAMESPACED where the filer used an
    #: extension (`apa:RevenuesAndOther`). None for `linkbase_sum` / `field_sum`.
    concept: str | None = None
    #: (concept, weight) pairs to sum, for `linkbase_sum` and `statement_leaf_sum`.
    #: `linkbase_sum` carries the FILER'S OWN weights; `statement_leaf_sum` carries 1.0,
    #: because there the declared weight is an admission test rather than a coefficient
    #: (see `_leaf_sum` guard 2).
    children: tuple[tuple[str, float], ...] = ()
    #: Catalogue FIELD names to sum, for `field_sum`.
    component_fields: tuple[str, ...] = ()
    #: What the route anchored on: the income-statement subtotal for route 2 (the evidence
    #: for a discovered concept), the cash-flow statement node for route 3b.
    anchor: str | None = None
    #: The statement role the winning arc came from. Corroborates the regime, since FASB's
    #: role URIs name the template (`sfp-dbo` 108000 = bank, `sfp-ibo` 108200 = insurer...).
    role_uri: str | None = None
    #: Concepts to subtract from `concept` (`total_adjustment`), already filtered by the
    #: `_only_when` linkbase condition where the field declares one.
    subtract: tuple[str, ...] = ()
    #: Why the field has no value at all, when it has none.
    dc_code: str | None = None
    #: The field resolved only after a concept the filer reports as ZERO IN EVERY PERIOD
    #: was put back in play -- i.e. the zero is the filer's whole answer, not a tagging
    #: artefact. True for VRT 2018-2020, whose `Revenues = 0` is correct (those filings are
    #: the GS Acquisition Holdings blank-cheque shell, pre-merger, with genuinely no
    #: revenue). Recorded so a zero is never silently either kept or discarded.
    zero_only_retained: bool = False
    #: Concepts a single-concept route (1 or 5) DECLINED because this filer declares them
    #: ONLY outside its face statements -- see `is_note_only`. Carried on whatever route
    #: did answer, so the affected population stays separable and countable: the guard
    #: reorders resolution for the whole universe and "how often did it fire, and on what?"
    #: must be a query rather than an archaeology exercise.
    role_rejected: tuple[str, ...] = ()
    #: Candidates withheld because every calculation arc the filer gives them sits on a
    #: SEGMENT-INFORMATION role -- see `SEGMENT_ROLE`. Unlike `role_rejected` this is
    #: NOT relaxable: a segment aggregate is a different measure, not a narrower basis, so
    #: there is no pass that puts it back. Carried so the population stays countable
    #: (`adjustment::jsonb ? 'segment_rejected'`) rather than vanishing into a null.
    segment_rejected: tuple[str, ...] = ()
    #: The field resolved ONLY after those rejections were put back in play, i.e. the
    #: note-level concept is the filer's whole answer for this field. The value is kept --
    #: a narrow real number beats a null, and 745 correct rows nulled by an over-strict
    #: guard is this repo's own most expensive precedent -- but it is flagged, because a
    #: note-level basis is exactly what Phase 5b's `basis_step` is looking for.
    role_only_retained: bool = False
    #: Candidates route 1 DECLINED because the filer does not declare them in its statement
    #: linkbase at all, while it DOES declare this field's own statement lines -- so route 3b
    #: is reading the filer's structure and route 1 would be reading a bare tag hit.
    #:
    #: The MEASURED half of 4c.1, and the reason the role test alone is not enough. CSCO tags
    #: `us-gaap:DepreciationDepletionAndAmortization` at **exactly $700,000,000 with
    #: `decimals=-8` in fiscal 2023, 2024 AND 2025** -- a rounded narrative figure, the same
    #: number three years running -- while its cash-flow D&A line is
    #: `csco:DepreciationAmortizationAndOther` at **$2,811M**, `decimals=-6`. The bad concept
    #: carries NO calculation arc at all, so no role test can see it; the good one is a
    #: COMPANY EXTENSION, so no candidate list can ever name it. Only "the filer declared its
    #: own lines and did not declare this" separates the two.
    undeclared_rejected: tuple[str, ...] = ()
    #: `(total, leg)` pairs route 1 DECLINED because the filer declares the catalogue's
    #: total as a SIBLING of one of this field's own roll-up legs instead of as its parent
    #: -- so the filer has said they are disjoint lines and its "total" is another leaf.
    #:
    #: Kept apart from `undeclared_rejected` because the two are different epistemic states
    #: and the module keeps its populations countable: that one means "the filer never put
    #: this concept in its statement structure", this one means "the filer put it there and
    #: told us it is not the parent". Measured at 2 of 12 route-1 `capex` filers -- MCD,
    #: whose `PaymentsToAcquireProductiveAssets` is "Purchases of restaurant and other
    #: businesses" ($540.9M FY2019) beside a $2,393.7M "Capital expenditures", and EQIX,
    #: whose is "Purchase of real estate". See `sibling_leg`.
    sibling_rejected: tuple[tuple[str, str], ...] = ()
    #: The field HAS a value, but on a basis the catalogue declares non-comparable -- the
    #: winning concept is named in the field's `dc_code_on_fallback` map. Today's only
    #: entry: `researchAndDevelopment` resolving on
    #: `ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost` instead of the
    #: aggregate, which is a SUBSET, not a substitute (of 21 both-tagged pairs, 0.0% agree
    #: within 1%; mean aggregate/ex-IPR&D ratio 1.675).
    #:
    #: Deliberately NOT `dc_code`: `resolved` is defined as "no dc_code", so a qualifier
    #: riding that column would turn every ex-IPR&D row into an absence and delete a real
    #: number. It rides the `adjustment` JSON instead -- the same seam 4c.1 used for
    #: `undeclared_rejected` -- and the history build lifts it into
    #: `fundamentals_reason_codes`, where a qualified cell and a null cell are both
    #: findable and still distinguishable.
    basis_qualifier: str | None = None

    @property
    def resolved(self) -> bool:
        return self.dc_code is None

    @property
    def is_extension(self) -> bool:
        """Did the filer's own total turn out to be a company extension element? True for
        APA's revenue. Legitimate -- it is the filer's declared total and its components
        are standard -- but worth carrying, because an extension total cannot be compared
        element-for-element with another filer's."""
        return bool(self.concept and ":" in self.concept
                    and not self.concept.startswith("us-gaap:"))

    @property
    def source_concept(self) -> str | None:
        """What to record as the value's origin, for the tag-switch-break check."""
        if self.concept:
            return self.concept
        if self.children:
            return "+".join(c for c, _ in self.children)
        return None


#: The columns `statement_arcs` and `calculation_arcs` guarantee, so a filing with no
#: linkbase returns an empty frame of the right SHAPE rather than one nothing can index.
ARC_COLUMNS = ["concept", "concept_taxonomy", "parent_concept", "parent_taxonomy",
               "weight", "role_uri", "menucat", "is_abstract", ARC_FILTER]


def calculation_arcs(xbrl) -> pd.DataFrame:
    """The filer's calculation linkbase, UNFILTERED, or an empty frame.

    The raw read that `statement_arcs` narrows. Split out because two questions need
    OPPOSITE views of the same linkbase and conflating them is how `is_note_only` came to be
    structurally blind: STRUCTURE must be read off face-statement arcs only, while "where
    does this filer declare this concept?" must be read off ALL of them. Ask the second
    question of the filtered frame and every note-only concept answers "nowhere", which
    `is_note_only` then reads as silence rather than as evidence.
    """
    try:
        arcs = xbrl.calculation_linkbase()
    except Exception:                                   # noqa: BLE001 -- absent linkbase
        return pd.DataFrame(columns=ARC_COLUMNS)
    if arcs is None or arcs.empty:
        return pd.DataFrame(columns=ARC_COLUMNS)
    return arcs


def segment_only_concepts(arcs: pd.DataFrame) -> frozenset[str]:
    """Bare names whose EVERY calculation arc sits on a SEGMENT-INFORMATION role.

    Read off the UNFILTERED linkbase -- `statement_arcs` has already dropped these arcs by
    the time an `ArcGraph` exists, which is exactly why `is_note_only` cannot see them.

    A filing-level property, like `entity_scope.zero_only_concepts`, so resolution stays
    period-agnostic. **Silence is not evidence** here too, and for the same reason: a
    concept carrying no arc at all is absent from the result, because a leaf that can never
    carry a calculation arc (a `dei:` cover-page tag, `goodwill`) would otherwise be
    condemned by having no roles rather than by having only segment ones.

    Indexed on the CHILD end only, unlike `ArcGraph._all_roles` which reads both. In a
    segment RECONCILIATION arc the child is the segment aggregate and the parent is the
    consolidated statement line it reconciles TO, so reading both ends condemns exactly the
    concept the note is reconciling to. Measured: it cost ORCL its `pretaxIncome` for
    2017-08-31 ($2,500M) and 2018-08-31 ($2,540M), because in 10-Q 0001564590-18-023315
    `IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrolling
    Interest` carries no arc except as the parent of the segment note's `GrossProfit`. The
    values were right and the element was the filer's real consolidated pretax line.
    """
    if arcs.empty or "role_uri" not in arcs.columns or "concept" not in arcs.columns:
        return frozenset()
    roles: dict[str, set[str]] = {}
    for concept, role in zip(arcs["concept"], arcs["role_uri"]):
        roles.setdefault(bare(str(concept)), set()).add(str(role))
    return frozenset(concept for concept, seen in roles.items()
                     if seen and all(SEGMENT_ROLE.search(role) for role in seen))


def statement_arcs(xbrl, arcs: pd.DataFrame | None = None) -> pd.DataFrame:
    """The face-statement slice of a filing's calculation linkbase.

    **The UNION of two tests, because each is lossy where the other is not.** Measured on
    26 tickers x 1,544 filings:

      * `menucat == "Statements"` is `None` for 100% of arcs on **418 filings (27.1%)** --
        every filing from 2011 to mid-2015, because edgartools cannot categorise the older
        FilingSummary.xml. The whole linkbase was being discarded and the resolver silently
        degraded to the tag list for four of fifteen years (linkbase share of resolutions:
        **0.9% in 2011-14** against ~70% from 2016). The arcs were there the whole time --
        AAPL's FY2013 10-K carries the complete face statements with correct signed weights.
      * `menucat` is **also lossy where it IS populated**. A filer that splits one
        statement's calculations across several roles gets only the first categorised, so
        `Uncategorized`/`NaN` hides genuine face statements: APA 2022
        `STATEMENTOFCONSOLIDATEDOPERATIONS`, AMT/SO/SPG operations roles, MAA's balance
        sheet. Four of the five found are INCOME-statement roles, which is very likely part
        of why revenue needed structural discovery at all.

    Neither test admitted a junk arc in the measured sample, and the role test was never
    missing an arc `menucat` keeps -- so the union costs nothing and recovers both losses.
    `ARC_FILTER` records which test admitted each arc.

    Returns an empty frame with the expected columns when the filing ships no calculation
    linkbase -- a real and expected case (older filings), and precisely what routes a
    field to `tag_fallback` rather than to an error.

    `arcs` lets a caller that already read the UNFILTERED linkbase hand it in.
    `edgar.xbrl.XBRL.calculation_linkbase` carries no cache, so the fetch path -- which
    needs both views, one for `segment_only_concepts` and one for the graph -- was paying
    for the same parse twice on every filing.
    """
    arcs = calculation_arcs(xbrl) if arcs is None else arcs
    if arcs.empty:
        return pd.DataFrame(columns=ARC_COLUMNS)

    empty = pd.Series(False, index=arcs.index)
    by_menucat = (arcs["menucat"] == STATEMENTS_MENUCAT
                  if "menucat" in arcs.columns else empty)
    by_role = (~arcs["role_uri"].astype(str).str.contains(NON_STATEMENT_ROLE)
               if "role_uri" in arcs.columns else empty)

    keep = by_menucat | by_role
    if not keep.any():
        return pd.DataFrame(columns=ARC_COLUMNS)
    out = arcs[keep].copy()
    out[ARC_FILTER] = [
        "both" if m and r else ("menucat" if m else "role_uri")
        for m, r in zip(by_menucat[keep], by_role[keep])]
    return out.reset_index(drop=True)


def qualify(concept: str, taxonomy: str | None) -> str:
    """`('RevenuesAndOther', 'apa')` -> `apa:RevenuesAndOther`.

    The linkbase splits a concept into name + taxonomy; the facts frame namespaces it into
    one string. Everything downstream keys on the namespaced form, because the bare name
    alone is genuinely ambiguous: `RevenuesAndOther` is APA's own element, and collapsing
    it with a hypothetical standard one of the same name would be a silent basis error.
    """
    taxonomy = (taxonomy or "").strip()
    if not taxonomy or taxonomy.lower() in {"nan", "none"}:
        return concept
    return f"{taxonomy}:{concept}"


@dataclass
class ArcGraph:
    """Parent/child index over one filing's statement arcs.

    Built once per filing, queried once per field. Node keys are BARE names, matching the
    linkbase's own `concept` column; `taxonomy_of` recovers the namespace so a resolved
    concept can be qualified before it is stored or looked up in the facts.
    """

    arcs: pd.DataFrame

    @cached_property
    def _children(self) -> dict[str, list[tuple[str, float]]]:
        out: dict[str, list[tuple[str, float]]] = {}
        if self.arcs.empty:
            return out
        for parent, concept, weight in zip(self.arcs["parent_concept"],
                                           self.arcs["concept"], self.arcs["weight"]):
            out.setdefault(str(parent), []).append((str(concept), float(weight)))
        return out

    @cached_property
    def _children_with_role(self) -> dict[str, list[tuple[str, float, str]]]:
        """parent -> [(child, weight, role_uri)]. The role-carrying twin of `_children`,
        built once per filing because route 3b needs to know WHICH STATEMENT an arc came
        from and `_children` deliberately pools every role.

        The distinction is load-bearing and is the §4b.4 defect: AAPL declares
        `us-gaap:Depreciation` in a PP&E note, where it is a $8.0bn note disclosure, while
        its cash-flow D&A line is `DepreciationDepletionAndAmortization` at $11.7bn --
        31.6% apart. A concept NAME cannot tell those two apart. The role can.
        """
        out: dict[str, list[tuple[str, float, str]]] = {}
        if self.arcs.empty:
            return out
        roles = (self.arcs["role_uri"] if "role_uri" in self.arcs.columns
                 else pd.Series("", index=self.arcs.index))
        for parent, concept, weight, role in zip(
                self.arcs["parent_concept"], self.arcs["concept"],
                self.arcs["weight"], roles):
            out.setdefault(str(parent), []).append(
                (str(concept), float(weight), str(role)))
        return out

    @cached_property
    def _parents(self) -> dict[str, str]:
        """child -> parent. First arc wins: a concept presented on two statements is a
        re-presentation of one line (a cash-flow `Cash` also appearing on the balance
        sheet), not a genuinely different roll-up."""
        out: dict[str, str] = {}
        if self.arcs.empty:
            return out
        for concept, parent in zip(self.arcs["concept"], self.arcs["parent_concept"]):
            out.setdefault(str(concept), str(parent))
        return out

    @cached_property
    def _taxonomy(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.arcs.empty:
            return out
        for concept, taxonomy in zip(self.arcs["concept"], self.arcs["concept_taxonomy"]):
            out.setdefault(str(concept), str(taxonomy))
        for parent, taxonomy in zip(self.arcs["parent_concept"], self.arcs["parent_taxonomy"]):
            out.setdefault(str(parent), str(taxonomy))
        return out

    @cached_property
    def _role_of(self) -> dict[str, str]:
        """concept -> the role its FIRST arc sits on, indexed from BOTH sides of the arc.

        Indexing only the `concept` column -- which this did until the 2026-08-22
        clarification pass -- makes `role_of` return None for **every parentless root**,
        because a root appears only ever in `parent_concept`. That silently disabled the
        role test `discover_root` needs: applied to the old index it would have rejected
        every candidate it was meant to rank, including the correct ones.
        """
        out: dict[str, str] = {}
        if self.arcs.empty or "role_uri" not in self.arcs.columns:
            return out
        for concept, role in zip(self.arcs["concept"], self.arcs["role_uri"]):
            out.setdefault(str(concept), str(role))
        for parent, role in zip(self.arcs["parent_concept"], self.arcs["role_uri"]):
            out.setdefault(str(parent), str(role))
        return out

    @cached_property
    def _all_roles(self) -> dict[str, frozenset[str]]:
        """concept -> EVERY role its arcs sit on, from both sides of the arc.

        The set-valued twin of `_role_of`, which keeps only the FIRST role and therefore
        cannot answer the only question that matters for the statement-role test: is this
        concept declared ANYWHERE on a face statement? A first-arc answer is an arc-order
        lottery -- CSCO declares `DepreciationDepletionAndAmortization` on both a cash-flow
        role and a PP&E-note role, and which one `_role_of` returns depends on insertion
        order.
        """
        out: dict[str, set[str]] = {}
        if self.arcs.empty or "role_uri" not in self.arcs.columns:
            return {}
        for column in ("concept", "parent_concept"):
            for concept, role in zip(self.arcs[column], self.arcs["role_uri"]):
                out.setdefault(str(concept), set()).add(str(role))
        return {concept: frozenset(roles) for concept, roles in out.items()}

    @property
    def is_empty(self) -> bool:
        return self.arcs.empty

    def knows(self, concept: str) -> bool:
        """Does the filer's own statement structure mention this concept at all?"""
        return concept in self._parents or concept in self._children

    def children_of(self, concept: str) -> list[tuple[str, float]]:
        return self._children.get(concept, [])

    def children_on_role(self, concept: str,
                         role: re.Pattern[str]) -> list[tuple[str, float, str]]:
        """Every arc beneath `concept` ON A ROLE matching `role`, NOT de-duplicated.

        Deliberately un-deduplicated, because a duplicate here is signed: NEE declares
        `nee:IndependentPowerInvestments` and `nee:PurchasesOfSecuritiesInSpecialUseFunds`
        twice in the same node, once at +1.0 and once at -1.0. Collapsing on first-arc-wins
        would let arc order decide which sign survives, and the sign is exactly what route
        3b's guard 2 reads. `_leaf_sum` filters by sign FIRST and de-duplicates after, so
        the surviving arc is the one whose direction is the field's.
        """
        return [(child, weight, arc_role)
                for child, weight, arc_role in self._children_with_role.get(concept, [])
                if role.search(arc_role)]

    def parent_of(self, concept: str) -> str | None:
        return self._parents.get(concept)

    def taxonomy_of(self, concept: str) -> str:
        return self._taxonomy.get(concept, "us-gaap")

    def qualified(self, concept: str) -> str:
        return qualify(concept, self.taxonomy_of(concept))

    def role_of(self, concept: str) -> str | None:
        parent = self.parent_of(concept)
        return self._role_of.get(concept) or (self._role_of.get(parent) if parent else None)

    def roles_of(self, concept: str) -> frozenset[str]:
        """Every role this concept is declared on. Empty means the filer's statement
        structure does not mention it at all -- which is NOT the same as mentioning it
        outside the face statements, and the statement-role test turns on that difference."""
        return self._all_roles.get(concept, frozenset())

    def roots_with_children(self) -> list[tuple[str, list[tuple[str, float]]]]:
        """Nodes that aggregate but hang free -- no parent arc anywhere in the statement
        roll-up. A revenue block can legitimately hang like this: DTE declares
        `OperatingIncomeLoss <- -1 CostsAndExpenses` and simply omits the revenue arc, so
        `RegulatedAndUnregulatedOperatingRevenue` has children but no parent."""
        return [(node, kids) for node, kids in self._children.items()
                if kids and self.parent_of(node) is None]

    def has_descendant(self, concept: str, descendant: str) -> bool:
        """Does the filer declare `descendant` anywhere BENEATH `concept`?

        The positive form of the containment question. `has_sibling` answers a different
        one and returns False both for "declared elsewhere" and for "the linkbase says
        nothing", which are not the same epistemic state -- only this one is evidence that
        an amount is actually inside a total.
        """
        seen: set[str] = set()
        stack = [concept]
        while stack:
            for child, _weight in self.children_of(stack.pop()):
                if child == descendant:
                    return True
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return False

    def has_sibling(self, concept: str, sibling: str) -> bool:
        """Do the two concepts share a parent? This is the `ppeNet` disambiguation: a
        finance-lease ROU asset declared as a SIBLING of `PropertyPlantAndEquipmentNet` is
        its own balance-sheet line and was never inside PP&E, so subtracting it would
        remove an amount that is not there. Declared as a descendant, it IS folded in."""
        parent = self.parent_of(concept)
        if parent is None:
            return False
        return any(child == sibling for child, _ in self.children_of(parent))


def sibling_leg(graph: ArcGraph, concept: str, legs: frozenset[str] | set[str],
                available: frozenset[str], magnitudes: dict[str, float]) -> str | None:
    """The field's own declared roll-up LEG that proves `concept` is not this field's
    total -- or None, which is the overwhelmingly common answer.

    `total_concept` is documented as "the concept to prefer when the filer declares it as a
    linkbase PARENT", but route 1 only ever asked `graph.knows` -- *is it in the statement
    structure at all* -- so any filer that tags the superset element on a DIFFERENT line
    handed route 1 that line as the field's total. MCD tags
    `PaymentsToAcquireProductiveAssets` as **"Purchases of restaurant and other
    businesses"** and route 1 stored its franchisee-acquisition line as capex for 67 rows.

    TWO conditions, and BOTH are required. Each is necessary and neither is sufficient:

    1. **The filer declares the leg BESIDE the total, not beneath it.** FASB's own
       calculation linkbase makes `PaymentsToAcquirePropertyPlantAndEquipment` a CHILD of
       `PaymentsToAcquireProductiveAssets`, which is what `capex.roll_up.sum` declares. A
       filer that puts both under one parent at one weight has stated they are disjoint
       lines. `has_descendant` wins over `has_sibling`: a filer that declares the leg both
       beneath and beside really is rolling up, and route 1 keeps it.
    2. **The filer reports the leg LARGER than the total.** A concept FASB defines as
       PP&E + software + intangibles cannot be smaller than the PP&E it contains, so when
       it is, the element is being used for some other line. This is the filer's own
       arithmetic contradicting the filer's own structure, with no appeal to a label.

    **Condition 2 is not decoration -- without it this rule destroys correct data, and the
    measurement is why it is here.** Condition 1 alone also fires on AAPL and SWKS, which
    declare `PaymentsToAcquireIntangibleAssets` beside the total and are entirely correct:
    AAPL's `PaymentsToAcquireProductiveAssets` IS its $9,571M "Payments for acquisition of
    property, plant and equipment" (0001193125-14-383437), with intangibles a separate
    ~$242M line. Refusing there would have handed route 3b the intangibles leg alone and
    cut AAPL's FY2014 capex from **$9,571M to $242M** -- a 97.5% understatement, the same
    class of self-inflicted damage as the 745 correct rows once nulled by over-strict Q4
    guards. Condition 2 keeps both filers on route 1, because their total is the larger
    number, which is exactly what a real superset looks like.

    MEASURED on the 12 roster tickers resolving `capex` by route 1 on
    `PaymentsToAcquireProductiveAssets` (one filing each, calculation linkbase + facts).
    **10 correct, 2 mistaggers**, and both mistaggers declare the leg as a sibling at the
    same -1.0 weight under `NetCashProvidedByUsedInInvestingActivities` AND report it
    several times larger:

      * **MCD** 0000063908-20-000022 -- total "Purchases of restaurant and other
        businesses" **$540.9M** beside leg "Capital expenditures" **$2,393.7M** (FY2019).
        Excluding the restaurant line is DEFINITIONAL, not convenient: it is a business
        acquisition, the same ground on which NEE's
        `PaymentsToAcquireBusinessesGrossAndRelatedCapitalExpenditures` is excluded in
        `fundamentals_exceptions.json`.
      * **EQIX** 0001193125-12-317379 -- "Purchase of real estate" **$24.0M** beside
        "Purchases of property, plant and equipment" **$342.0M**.

    The remaining 10 (AMT, CAT, KR, NVDA, SPG, VLO, LLY, APA on condition 1; AAPL and SWKS
    on condition 2) keep `linkbase_total` untouched.

    Two further guards make a NULL impossible: the leg must be REPORTED (`available`), so
    a refusal always has a real number behind it, and the caller requires route 3b to have
    leaves, so a refusal hands off to an answer rather than to nothing.
    """
    total_peak = magnitudes.get(concept)
    if total_peak is None:
        return None
    for leg in sorted(legs):
        leg_peak = magnitudes.get(leg)
        if leg_peak is None or leg not in available:
            continue
        if leg_peak <= total_peak:
            continue                                    # a real superset -- AAPL, SWKS
        if graph.has_sibling(concept, leg) and not graph.has_descendant(concept, leg):
            return leg
    return None


def bare(concept: str) -> str:
    """`dei:EntityCommonStockSharesOutstanding` -> `EntityCommonStockSharesOutstanding`.

    The catalogue writes most concepts bare but namespaces the few that are NOT us-gaap
    (the cover-page `dei:` tags), while the linkbase and the reported-concept set are keyed
    bare. Matching the two forms without normalising silently skipped
    `dei:EntityCommonStockSharesOutstanding` -- the ONLY summable share tag for a
    multi-class issuer -- and fell through to `CommonStockSharesOutstanding`, which is a
    single class. That is the multi-class NULL defect, and it returns the moment this
    normalisation is dropped.
    """
    return concept.split(":", 1)[-1] if concept else concept


def _candidates(spec: FieldSpec, regime: str | None) -> list[str]:
    """The field's concepts in priority order -- declared total first, then the fallback
    list, with `never_use` removed. De-duplicated, order preserved (the total is usually
    also the head of the fallback list). Returned AS DECLARED, namespace included: callers
    match on `bare()` and store the declared form."""
    banned = {bare(c) for c in spec.never_use(regime)}
    seen: set[str] = set()
    out: list[str] = []
    for concept in [spec.total_concept(regime), *spec.fallback_concepts(regime)]:
        if concept and bare(concept) not in banned and bare(concept) not in seen:
            seen.add(bare(concept))
            out.append(concept)
    return out


#: The two structural tests a field may name in `total_adjustment._only_when_test`. Only the
#: stronger one has a name to declare: the weaker sibling test is what a field gets by NOT
#: naming a test, which is what `ppeNet` has always relied on.
#:
#: They differ because the two standards differ, not as a matter of taste:
#:
#:   * `not_a_declared_sibling` (ppeNet). **ASC 842-20-45-4 explicitly permits** a lessee to
#:     fold a finance-lease ROU asset into an existing line such as PP&E. Folding in is a
#:     real and common presentation, so silence is weak evidence FOR containment and a
#:     separately-declared sibling is strong evidence against.
#:   * `declared_descendant` (shortTermDebt). **ASC 842-20-45-1 requires** operating lease
#:     liabilities to be presented separately from other liabilities, so the prior runs the
#:     other way: an operating lease leg is presumed OUTSIDE the debt total unless the
#:     filer's own linkbase says otherwise.
ONLY_WHEN_DESCENDANT = "declared_descendant"


def _resolve_subtractions(spec: FieldSpec, graph: ArcGraph, available: frozenset[str],
                          concept: str | None) -> tuple[str, ...]:
    """The `total_adjustment.subtract` concepts that actually apply to this filing.

    A field declaring `_only_when` is asking the LINKBASE, not the tag, whether the amount
    is inside the total. Where no condition is declared, every reported concept applies.

    **A declared condition that cannot be evaluated subtracts NOTHING**, in either of the
    two ways it can be unevaluable:

      * no single resolved concept (a `linkbase_sum` of two legs, neither of which contains
        a lease by construction); or
      * the resolved concept is **not in the statement linkbase at all** -- the
        `tag_primary` / `tag_fallback` routes. `has_sibling` returns False there, which the
        first version of this fix read as "not a sibling, therefore subtract". That is
        reading silence as evidence, and it left **75 of the 127 surviving negative
        `shortTermDebt` values** in place: AAPL, CSCO, KR, ETN, SO, DTE, VRT all resolve on
        a tag route and all were still having lease legs removed from a total that never
        contained them.

    Measured on 31 filings spanning every (ticker, concept, route) combination that still
    subtracts: the lease leg is `LEG-NOT-IN-LINKBASE` or `TOTAL-NOT-IN-LINKBASE` in **every
    single case, ppeNet included**, and a declared descendant in none. So the sibling test
    has never actually discriminated anything on this roster -- it is only ever the absence
    of the leg that lets the subtraction through. That is fine for `ppeNet` (0 negatives,
    and ASC 842-20-45-4 says folding in is permitted) and wrong for `shortTermDebt`, which
    is why the test is now named per field rather than assumed.
    """
    adjustment = spec.raw.get("total_adjustment") or {}
    wanted = [bare(c) for c in adjustment.get("subtract", []) if bare(c) in available]
    if not wanted:
        return ()

    # A CONCEPT-scoped adjustment: the element's own name says it contains the amount, so
    # subtracting is definitional rather than a guess -- `LongTermDebtAndCapitalLeaseObligations`
    # cannot not contain capital leases. `longTermDebt` declared this in prose (a `note`
    # key) and nothing read it, so the subtraction fired on `LongTermDebtNoncurrent` too and
    # drove AMT 2021 to -$22.1M. Distinct from `_only_when`, which asks the linkbase about a
    # containment the element name leaves open.
    only_for = {bare(c) for c in adjustment.get("_only_when_concept", [])}
    if only_for and bare(concept or "") not in only_for:
        return ()

    if not adjustment.get("_only_when"):
        return tuple(wanted)
    if concept is None or not graph.knows(concept):
        return ()
    if adjustment.get("_only_when_test") == ONLY_WHEN_DESCENDANT:
        return tuple(c for c in wanted if graph.has_descendant(concept, c))
    return tuple(c for c in wanted if not graph.has_sibling(concept, c))


def is_note_only(graph: ArcGraph, concept: str) -> bool:
    """Is `concept` declared by this filer ONLY outside its face statements?

    The statement-role test, and the whole of 4c.1. A note-level fact can win a
    priority walk over the statement line that is the field, and it does: AMT tags a
    debt-note element at **$1.9M** while its balance-sheet `LongTermDebtNoncurrent` is
    **$21,127M**; CSCO's PP&E-note D&A is $1,200M against a $2,811M cash-flow line
    (2.3x, 45 filings); MCD's note-level capex reads ~12x low over 64 rows and ten
    fiscal years; PG tags a note-level `Revenues` of $28,400M against $83,680M. Route 3b
    has carried this guard since Phase 4b -- which is exactly why it is the safe route --
    and this is it, generalised to the single-concept routes.

    **Silence is not evidence, and that is the load-bearing half of the rule.** §3c.8's
    most expensive lesson was `_only_when` reading "the linkbase says nothing" as licence
    to act. A leaf like `goodwill`, or a `dei:` cover-page tag, can never carry a
    calculation arc at all, and `tag_primary` is its normal and correct home. So the test
    fires only on POSITIVE evidence of note-hood: the concept is declared, and every role
    it is declared on is a non-statement role. Undeclared -> False, always.

    Note that `statement_arcs` admits an arc whose role URI looks like a note whenever
    `menucat == "Statements"`, so those arcs ARE in the graph -- that mis-categorisation is
    how AAPL's PP&E-note `us-gaap:Depreciation` reaches the resolver in the first place.
    `NON_STATEMENT_ROLE` is reused unchanged and must not be loosened: `schedule` is what
    excludes Reg S-X Schedule I/II parent-company-only condensed statements, which look
    exactly like face statements and would corrupt consolidated numbers silently.
    """
    roles = graph.roles_of(concept)
    if not roles:
        return False
    return all(NON_STATEMENT_ROLE.search(role) for role in roles)


def is_income_statement_role(role: str | None) -> bool:
    """Does this role URI name the statement a revenue top line lives on?"""
    if not role:
        return False
    return (bool(INCOME_STATEMENT_ROLE.search(role))
            and not NOT_INCOME_STATEMENT_ROLE.search(role))


def discover_root(graph: ArcGraph, available: frozenset[str],
                  banned: frozenset[str] = frozenset(),
                  duration_concepts: frozenset[str] | None = None,
                  ) -> tuple[str, str] | None:
    """Find the filer's own revenue top line from the statement STRUCTURE alone.

    Returns `(concept, anchor)` in BARE names, or None. Two structural signatures, tried
    in order, both measured against live filings:

      1. **Positive child of the nearest revenue anchor.** Walk `REVENUE_ANCHORS`
         nearest-to-revenue first; the first anchor the filer declares contributes its
         positive-weight, reported children. Catches APA (`apa:RevenuesAndOther` under the
         pretax node) and AAPL (revenue under `GrossProfit`).
      2. **A parentless, reported root that aggregates.** DTE declares
         `RegulatedAndUnregulatedOperatingRevenue` with no parent arc at all -- its
         `OperatingIncomeLoss` carries only the `-1 CostsAndExpenses` side -- so the
         revenue block hangs free. A node with no parent, with positive-weight children,
         that is itself reported, is that block's total.

    `NOT_A_TOP_LINE` filters the non-operating and disposal items a filer parks under the
    same subtotal; MAA's pretax node has four positive children and only one is revenue.

    `duration_concepts` is the set of bare names this filing reports ONLY as duration
    facts. Pass it: without it the instant test is skipped and signature 2 is free to
    return a balance-sheet total again. `None` means "the caller does not know", which
    keeps the pure-structure unit tests callable.
    """
    # An anchor is a margin SUBTOTAL, so it can never itself be the revenue line -- DTE
    # declares `IncomeLoss...BeforeIncomeTaxes <- +1 OperatingIncomeLoss`, and without this
    # exclusion the operating margin would be stored as the utility's revenue.
    excluded = NOT_A_TOP_LINE | set(REVENUE_ANCHORS) | banned
    for anchor in REVENUE_ANCHORS:
        if not graph.knows(anchor):
            continue
        picks = [child for child, weight in graph.children_of(anchor)
                 if weight > 0 and child in available and child not in excluded]
        if len(picks) == 1:
            return picks[0], anchor
        if len(picks) > 1:
            # Prefer a node that is itself an aggregation: a revenue total footing from
            # its own components outranks a bare line parked under the same subtotal.
            aggregating = [p for p in picks if graph.children_of(p)]
            if len(aggregating) == 1:
                return aggregating[0], anchor
            return picks[0], anchor

    # Signature 2. The 26-ticker sweep put **74 revenue rows** on `Assets` (18),
    # `LiabilitiesAndStockholdersEquity` (16), cash-flow period-increase totals (24),
    # `ComprehensiveIncomeNetOfTax` (14) and `NoninterestExpense` (2) -- APA reporting
    # revenue of -$467M. The mechanism is NOT a missing constraint: `roots_with_children`
    # returns roots in ARC INSERTION ORDER and this loop took the first that qualified. On
    # DTE's 2020-04-28 10-Q the correct `RegulatedAndUnregulatedOperatingRevenue` was
    # present, reported and all-positive, and lost purely because the cash-flow root
    # appeared earlier in the arc list. So the repair is two-part: constrain, then RANK.
    #
    # `balance == credit` was tested as a third axis and REJECTED -- the column is empty
    # for GS's `RevenuesNetOfInterestExpense` and DTE's own revenue concept, so requiring
    # it would reject the correct answers.
    survivors: list[tuple[str, list[tuple[str, float]]]] = []
    for node, kids in graph.roots_with_children():
        if node not in available or node in excluded:
            continue
        if not all(weight > 0 for _, weight in kids):
            continue
        if duration_concepts is not None and node not in duration_concepts:
            continue                                    # kills the 34 balance-sheet rows
        if not is_income_statement_role(graph.role_of(node)):
            continue                                    # kills cash-flow / OCI roots
        survivors.append((node, kids))
    if survivors:
        # Deterministic ORDER, replacing the arc-order lottery: the widest aggregation on
        # an income-statement role is the revenue block, and the name breaks ties so the
        # same filing always resolves the same way.
        survivors.sort(key=lambda pair: (-len(pair[1]), pair[0]))
        return survivors[0][0], "linkbase_root_node"
    return None


def resolve_field(spec: FieldSpec, graph: ArcGraph, available: frozenset[str],
                  catalogue: Catalogue, regime: str | None = None,
                  duration_concepts: frozenset[str] | None = None,
                  zero_only: frozenset[str] = frozenset(),
                  magnitudes: dict[str, float] | None = None,
                  ticker: str | None = None,
                  prefer_structure: bool = True,
                  segment_only: frozenset[str] = frozenset()) -> Resolution:
    """Decide how `spec` resolves for one filing, in TWO passes over the zero guard.

    The candidate list is built ONCE here and threaded through every pass: it is a property
    of `(spec, regime)` alone, and the three passes plus route 3b asked for it up to seven
    times per (filing, field).

    `zero_only` is the set of bare names this filing reports as **exactly 0 in every
    period it reports them at all** (`entity_scope.zero_only_concepts`). Measured, that is
    the whole discriminator between a tagging artefact and a real zero -- there is no
    one-bad-quarter-among-good-ones case in 26 tickers x 15 years, so this stays a
    FILING-level property and resolution stays period-agnostic.

    Pass 1 resolves with those concepts withheld. If the field then has no answer at all,
    pass 2 puts them back and flags `zero_only_retained`. The three measured cases each
    get the answer they deserve:

      * **ETN** tags `Revenues = 0` while `SalesRevenueNet` ($20.9-22.6 bn) sits
        undimensioned in the same filing -> pass 1 takes the real top line.
      * **APA** tags `RevenueFromContractWithCustomerIncludingAssessedTax = 0` and declares
        it in its OWN linkbase, so the zero arrives by `linkbase_total`, not by the
        priority walk -- which is why the guard is applied to `available` rather than
        confined to route 5. Pass 1 falls through to route 2 and finds
        `apa:RevenuesAndOther` ($596M-3,874M).
      * **VRT** 2018-2020 has no other revenue-like fact anywhere: those filings are the
        *GS Acquisition Holdings* blank-cheque shell pre-merger ($690M IPO proceeds, G&A of
        $123k), and the zero is CORRECT. Pass 2 keeps it, flagged.

    That last case is why the plan's original acceptance criterion ("no zero-revenue rows")
    was itself wrong, and now reads: *no zero unless the filer reports no other non-zero
    concept for the field anywhere in the filing.*

    `segment_only` is the set of bare names whose every calculation arc sits on a
    segment-information role (`segment_only_concepts`). It is withheld from `available`
    before the first pass and never restored, which is what separates it from the other two
    guards: they refuse a real number on a narrow basis and so earn a relaxation, while this
    one refuses a management aggregate posing as a statement line.

    `prefer_structure=False` disables BOTH halves of 4c.1 -- the statement-role test and the
    declaredness test. It exists as a MEASUREMENT SEAM, not a production knob: 4c.1 reorders
    resolution across the whole universe, and its acceptance is a before/after on the same
    join key with every value change named. Without this flag that comparison needs two
    network sweeps of 3,200 filings; with it, one sweep resolves each filing both ways off one
    `xbrl()` call. Nothing in `src/` ever passes False.
    """
    # Withheld from `available` itself, so no later pass can put them back. The zero and
    # role guards both have a relaxation because their refusals cost a real number; this one
    # has none, because the number it refuses is a DIFFERENT MEASURE and keeping it is the
    # defect. See `SEGMENT_ROLE` for the ORCL measurement.
    candidates = _candidates(spec, regime)
    withheld = frozenset(bare(c) for c in candidates) & segment_only
    available = available - segment_only

    strict = _resolve_once(spec, graph, available - zero_only, available, catalogue,
                           regime, duration_concepts, magnitudes=magnitudes, ticker=ticker,
                           prefer_structure=prefer_structure, candidates=candidates)
    if strict.resolved:
        return _stamp_basis(spec, _segment_stamp(strict, withheld))

    # Relax the statement-role guard before the zero guard, because the two relaxations
    # are not equally palatable: a note-level fact is a real amount on a narrow basis,
    # while a zero-in-every-period fact is usually a tagging artefact. So if both guards
    # are the reason a field has no answer, prefer the narrow number to the zero.
    if strict.role_rejected:
        relaxed = _resolve_once(spec, graph, available - zero_only, available, catalogue,
                                regime, duration_concepts, magnitudes=magnitudes,
                                ticker=ticker, prefer_structure=False,
                                candidates=candidates)
        if relaxed.resolved:
            # Carry the strict pass's ledger through, so the row says BOTH what was
            # withheld and that it had to be put back -- either half alone is unreadable.
            return _stamp_basis(spec, _segment_stamp(
                replace(relaxed, role_only_retained=True,
                        role_rejected=strict.role_rejected), withheld))

    if not zero_only:
        return _segment_stamp(strict, withheld)
    retry = _resolve_once(spec, graph, available, available, catalogue, regime,
                          duration_concepts, magnitudes=magnitudes, ticker=ticker,
                          prefer_structure=False, candidates=candidates)
    return (_stamp_basis(spec, _segment_stamp(replace(retry, zero_only_retained=True),
                                              withheld))
            if retry.resolved else _segment_stamp(strict, withheld))


def _segment_stamp(resolution: Resolution, withheld: frozenset[str]) -> Resolution:
    """Record the segment-role refusal, and make it the REASON where it caused the null.

    Two different rows come out of here. A field that still resolved keeps its value and
    merely carries the withheld names -- the filer tagged a segment aggregate AND a real
    statement line, and we took the statement line. A field that resolved nowhere else gets
    `segment_only_concept` in place of `not_disclosed`, because the concept IS tagged and
    calling it undisclosed would be a false statement about the filing.
    """
    if not withheld:
        return resolution
    resolution = replace(resolution, segment_rejected=tuple(sorted(withheld)))
    return (resolution if resolution.resolved
            else replace(resolution, dc_code=SEGMENT_ONLY_CONCEPT))


def _stamp_basis(spec: FieldSpec, resolution: Resolution) -> Resolution:
    """Flag a resolution that answered on a concept the catalogue declares non-comparable.

    Keyed on the CONCEPT rather than on the route, because the basis is a property of the
    element the filer used and not of how we found it: the ex-IPR&D element is a genuine
    narrower measure whether route 1 or route 5 picks it up.

    Only a field that declares `dc_code_on_fallback` can be stamped, and only one does, so
    this is a dict lookup on a missing key for 52 of 53 fields.
    """
    declared = spec.raw.get("dc_code_on_fallback") or {}
    if not declared or not resolution.concept:
        return resolution
    code = declared.get(bare(resolution.concept))
    return replace(resolution, basis_qualifier=code) if code else resolution


def _resolve_once(spec: FieldSpec, graph: ArcGraph, usable: frozenset[str],
                  available: frozenset[str], catalogue: Catalogue,
                  regime: str | None = None,
                  duration_concepts: frozenset[str] | None = None,
                  magnitudes: dict[str, float] | None = None,
                  ticker: str | None = None, prefer_structure: bool = True,
                  candidates: list[str] | None = None) -> Resolution:
    """One resolution pass.

    `candidates` is `_candidates(spec, regime)`, handed down by `resolve_field` so the three
    passes share one list rather than each rebuilding it.

    `usable` is what a SINGLE-concept route (1, 2, 5) may pick; `available` is everything
    the filing reports. They differ only by the zero guard, and route 3 deliberately reads
    `available`: a genuinely-zero LEG contributes 0 to a weighted sum and must not break
    it, whereas a genuinely-zero TOTAL would be the whole answer.

    `available` is the set of BARE concept names this filing reports at least one usable
    (undimensioned, numeric) fact for -- see `entity_scope.reported_concepts`. It is what
    stops the resolver selecting a node the filer declared structurally but never actually
    reported, which would produce a confident NULL indistinguishable from a coverage
    regression.

    Never raises for an unresolvable field: it returns a `Resolution` carrying a `dc_code`,
    because "this field is absent and here is why" is a first-class outcome of this
    pipeline, not an error.
    """
    regime_block = spec.raw.get("regimes", {}).get(regime or "", {})
    if "dc_code" in regime_block:
        # The regime declares the field meaningless -- bank capex, and bank FCF behind it.
        return Resolution(field=spec.name, method=UNRESOLVED,
                          dc_code=regime_block["dc_code"])

    candidates = _candidates(spec, regime) if candidates is None else candidates
    # bare()d to match `_candidates`, which bares its own copy. Left namespaced, a
    # `never_use` entry written with a prefix would silently fail to ban anything in
    # `discover_root`.
    banned = frozenset(bare(c) for c in spec.never_use(regime))

    # A declared roll-up LEG is not a total, even when it is also listed as a fallback
    # concept. Letting one satisfy route 1 would re-introduce the exact `shortTermDebt`
    # defect this rebuild removes: with `DebtCurrent` unreported, `LongTermDebtCurrent`
    # would win route 1 and `ShortTermBorrowings` -- the larger leg 54.4% of the time --
    # would never be added. Legs stay eligible for route 5, where taking one is the last
    # resort rather than the first answer.
    legs = {bare(c) for c in spec.roll_up(regime) if c not in catalogue.fields}

    # ---- 1. the highest-priority catalogue TOTAL the filer reports.
    #
    # Candidate PRIORITY dominates linkbase presence, not the other way round. Requiring a
    # linkbase hit here let a lower-priority candidate beat a higher-priority one whenever
    # the filer happened to put the lower one in a roll-up: XOM declares
    # `CommonStockSharesOutstanding` under `CommonStockSharesIssued`, so `sharesOutstanding`
    # resolved to a SINGLE SHARE CLASS instead of the cover-page `dei` tag, which is the
    # only summable one for a multi-class issuer. The linkbase's job is to pick between
    # totals and to discover ones no list can name (route 2) -- not to demote the
    # catalogue's explicit first choice. The route label still records whether the
    # structure corroborated the pick.
    #
    # **The one exception, and it is narrow (4c.1).** Where the candidate is not declared
    # ANYWHERE in the statement linkbase and the filer DOES declare this field's own
    # statement lines beneath its anchor node, the tag hit is not the catalogue's explicit
    # first choice competing with a roll-up -- it is a bare tag competing with the filer's
    # own statement. Route 3b wins there. It cannot touch the XOM case that set the general
    # rule: `sharesOutstanding` declares no `roll_up.any_of`, so route 3b never fires for it,
    # and only `capex` and `depAmort` are eligible at all.
    #
    # Route 3b is therefore computed HERE rather than in its own block -- it is needed as
    # evidence during route 1, and `_leaf_sum` returns on its first statement for the 45 of
    # 48 fields that declare no `roll_up.any_of` and carry no `by_ticker` leaf register
    # entry. Only `capex`, `costOfRevenue` and `depAmort` get past it.
    leaves, leaf_refusal, leaf_provenance = _leaf_sum(
        spec, graph, available, regime,
        *catalogue.filer_leaves(ticker, spec.name), candidates=candidates)

    # Candidates withheld by the note-role test. Carried onto whatever route does answer, so
    # the guard's blast radius is measurable rather than invisible. Declared before
    # `_leaf_resolution` because the closure reads it.
    rejected: list[str] = []

    def _leaf_resolution(withheld: tuple[str, ...] = ()) -> Resolution:
        return Resolution(
            field=spec.name, method=STATEMENT_LEAF_SUM, children=leaves,
            # BOTH read off the winning arc, not off the concept. `graph.role_of` and
            # `parent_of` answer across ALL roles and take the first arc, so for UNP's
            # `Depreciation` they return the income statement's `CostsAndExpenses` -- which
            # would make `role_uri` contradict the regime it is meant to corroborate.
            anchor=leaf_provenance[0] if leaf_provenance else None,
            role_uri=leaf_provenance[1] if leaf_provenance else None,
            subtract=_resolve_subtractions(spec, graph, available, None),
            role_rejected=tuple(rejected), undeclared_rejected=withheld)

    for candidate in (c for c in candidates if bare(c) not in legs):
        name = bare(candidate)
        if name not in usable:
            continue
        if prefer_structure and is_note_only(graph, name):
            rejected.append(graph.qualified(name))
            continue
        declared = graph.knows(name)
        if prefer_structure and not declared and leaves:
            return _leaf_resolution(
                (candidate if ":" in candidate else qualify(name, "us-gaap"),))
        # Declared -- but declared BESIDE this field's own leg AND reported smaller than
        # it, which is the filer's structure and the filer's arithmetic agreeing that its
        # "total" is another leaf. Route 3b's sum of the declared lines is the field.
        # `leaves` is required here, so a refusal always hands off to an answer rather than
        # to a NULL. See `sibling_leg` for the 12-filing measurement behind both halves.
        if prefer_structure and declared and leaves:
            beside = sibling_leg(graph, name, legs, available, magnitudes or {})
            if beside is not None:
                return replace(
                    _leaf_resolution(),
                    sibling_rejected=((graph.qualified(name), graph.qualified(beside)),))
        return Resolution(
            field=spec.name,
            method=LINKBASE_TOTAL if declared else TAG_PRIMARY,
            concept=(graph.qualified(name) if declared
                     else (candidate if ":" in candidate else qualify(name, "us-gaap"))),
            role_uri=graph.role_of(name) if declared else None,
            subtract=_resolve_subtractions(spec, graph, available, name),
            role_rejected=tuple(rejected))

    # ---- 2. structural discovery, for the fields that opt in (today: totalRevenue).
    #         This is the APA / DTE repair -- see the module docstring.
    if spec.raw.get("linkbase_root_discovery") and not graph.is_empty:
        found = discover_root(graph, usable, banned, duration_concepts)
        if found:
            concept, anchor = found
            return Resolution(
                field=spec.name, method=LINKBASE_ROOT, concept=graph.qualified(concept),
                anchor=anchor, role_uri=graph.role_of(concept),
                subtract=_resolve_subtractions(spec, graph, available, concept),
                role_rejected=tuple(rejected))

    # ---- 3. no total reported, but the filer declares the legs. Sum them with weights.
    #         This is the `shortTermDebt` fix: `LongTermDebtCurrent` and
    #         `ShortTermBorrowings` are disjoint legs of `DebtCurrent`, and keeping only
    #         one discarded the LARGER leg in 54.4% of the 2,017 cells tagging both.
    concept_children = [bare(c) for c in spec.roll_up(regime)
                        if c not in catalogue.fields]
    if concept_children:
        # The weights are only meaningful against THIS field's own total -- see
        # `_linkbase_weights`.
        totals = {bare(c) for c in
                  [spec.total_concept(regime), *spec.fallback_concepts(regime)] if c}
        weighted = _linkbase_weights(graph, concept_children, available, totals)
        if weighted:
            return Resolution(
                field=spec.name, method=LINKBASE_SUM, children=weighted,
                role_uri=graph.role_of(weighted[0][0]),
                subtract=_resolve_subtractions(spec, graph, available, None),
                role_rejected=tuple(rejected))

    # ---- 3b. the field's constituent STATEMENT LINES, summed. `capex` and `depAmort`:
    #          FASB's own roll-up has ~7 members and every filer reports a different
    #          subset, so route 3's all-or-nothing rule fired for capex zero times in
    #          3,163 filings. See `_leaf_sum` for the three guards.
    if leaves:
        return _leaf_resolution()

    # ---- 4. composed of other catalogue fields (`totalDebt`, `ppeNet`). The caller
    #         completes this once those fields are resolved.
    component_fields = tuple(c for c in spec.roll_up(regime) if c in catalogue.fields)
    if component_fields:
        return Resolution(field=spec.name, method=FIELD_SUM,
                          component_fields=component_fields,
                          role_rejected=tuple(rejected))

    # ---- 5. no linkbase (or none covering this concept): the priority list, used only
    #         where it is genuinely the best available evidence.
    for candidate in candidates:
        name = bare(candidate)
        if name in usable:
            if prefer_structure and is_note_only(graph, name):
                # A LEG can reach route 5 that route 1 skipped, so the guard has to be
                # repeated here rather than hoisted -- the two loops walk different sets.
                if graph.qualified(name) not in rejected:
                    rejected.append(graph.qualified(name))
                continue
            # With no linkbase there is no `concept_taxonomy` to read, so trust the
            # catalogue's own namespace where it gave one (`dei:` cover-page tags) and
            # assume us-gaap otherwise -- never store a bare name, which would not join
            # against the other four routes' output.
            concept = candidate if ":" in candidate else qualify(candidate, "us-gaap")
            return Resolution(
                field=spec.name, method=TAG_FALLBACK, concept=concept,
                subtract=_resolve_subtractions(spec, graph, available, name),
                role_rejected=tuple(rejected))

    # Nothing answered. Prefer the MOST SPECIFIC reason available, in descending order of
    # what it tells a reader:
    #   * `partial_leaf_sum` -- route 3b saw the field's leaves and refused a short sum.
    #     The amount IS disclosed; naming it needs a `by_ticker` register entry.
    #   * a regime's `dc_code_when_absent` -- the regime says an absence here is
    #     STRUCTURAL. Distinct from the regime's `dc_code`, which short-circuits the field
    #     before any resolution is attempted: insurer capex must stay resolvable, because
    #     PGR tags `PaymentsToAcquirePropertyPlantAndEquipment` in 63 of 63 filings
    #     ($65-364M, every duration shape), while AFL, MET and CB tag no capex line at all
    #     in 63 filings each. One regime, two correct answers -- so the code attaches to
    #     the ABSENCE, not to the field.
    #   * `not_disclosed` -- the honest default.
    return Resolution(
        field=spec.name, method=UNRESOLVED,
        dc_code=(leaf_refusal or regime_block.get("dc_code_when_absent")
                 or "not_disclosed"),
        role_rejected=tuple(rejected))


def _roll_up(spec: FieldSpec, regime: str | None) -> dict:
    """The field's `roll_up` block, with the regime's override REPLACING the field-level
    one rather than merging into it.

    Replacement, not merge, matching `FieldSpec.roll_up`: a regime that redeclares the
    roll-up is stating a different composition, and inheriting half of the general one
    would silently mix two bases -- bank `totalRevenue`'s two-leg roll-up must not pick up
    the general revenue leaves.
    """
    block = spec.raw.get("regimes", {}).get(regime or "", {})
    return dict(block.get("roll_up") or spec.raw.get("roll_up") or {})


def _leaf_sum(spec: FieldSpec, graph: ArcGraph, available: frozenset[str],
              regime: str | None, filer_leaves: tuple[tuple[str, ...], ...],
              filer_not_leaves: frozenset[str],
              candidates: list[str] | None = None,
              ) -> tuple[tuple[tuple[str, float], ...], str | None,
                         tuple[str, str] | None]:
    """Route 3b: sum the field's constituent STATEMENT LINES, chosen from the filer's own
    calculation linkbase.

    Returns `(children, refusal, provenance)`, where `provenance` is
    `(anchor, role_uri)`. Exactly one of the first two is ever non-empty:
    either the leaves to sum, or a `dc_code` explaining a refusal; `((), None, None)` means
    "this route does not apply". `anchor` is the node the leaves were actually read from,
    which is NOT recoverable afterwards -- `graph.parent_of` returns the concept's FIRST
    arc's parent across all roles, and for UNP's `Depreciation` that is the income
    statement's `CostsAndExpenses`, not the cash-flow node the guard admitted it under.

    **Why route 3 cannot do this.** `_linkbase_weights` returns () unless EVERY declared
    child is reported, which is right for `shortTermDebt` -- two disjoint legs a filer
    almost always tags together -- and wrong here. FASB's roll-up under
    `PaymentsToAcquireProductiveAssets` has ~7 members (PP&E + Software + Intangibles +
    MineralRights + CryptoAsset + EquipmentOnLease + Other) and every filer reports a
    different subset, so requiring all three of `capex.roll_up.sum` meant route 3 fired for
    capex **zero times in 3,163 filings**. The same is true of D&A: the nine tickers with
    no `DepreciationDepletionAndAmortization` tag `Depreciation` and
    `AmortizationOfIntangibleAssets` as separate cash-flow lines and never the aggregate.

    So the shape is `roll_up.any_of`, a list of GROUPS:

        within a group -> the FIRST reported alternative, never two
        across groups  -> sum
        fires when     -> at least one group hits

    Groups are era-variants or naming-variants of one line; the sum is across genuinely
    disjoint legs. Every grouping on this roster was set by MEASURING co-occurrence: DTE's
    `PlantAndEquipmentExpendituresUtility` and `PaymentsToAcquirePropertyPlantAndEquipment-
    Utility` never appear in the same filing (60 + 3 = 63, exactly its filing count) and so
    are alternatives, while `...Utility` and `...NonUtility` co-occur in all 60 and so are
    legs.

    **Three guards, each earned by a measurement, each with a named test.**

    1. **Statement-role guard.** A leaf counts only where the filer declares it beneath
       `roll_up.anchor` on a role matching `roll_up.anchor_role`. Without it AAPL's
       PP&E-note `us-gaap:Depreciation` is admitted and `depAmort` reads $8.0bn against a
       true $11.7bn -- **-31.6%**. Measured over every (ticker, year) publishing both the
       aggregate and the legs, a name-keyed leg sum reproduced the aggregate in only **14
       of 84** cases.
    2. **Weight-sign guard.** The leaf's declared weight must run in the field's own
       direction (`roll_up.leaf_weight`). PGR, MET and CB each carry a **-1.0**
       `AccretionAmortizationOfDiscountsAndPremiumsInvestments` contra inside the
       operating-activities node; a name filter admits it, the sign excludes it. The weight
       is an ADMISSION test only and never a multiplier -- a payment leaf is tagged
       positive and carries -1.0 because it reduces net investing cash, so multiplying
       would flip a non-negative field negative.
    3. **Partial-leaf guard.** If the anchor node carries a COMPANY-EXTENSION sibling in
       the field's direction that neither the catalogue nor the `by_ticker` register
       classifies, refuse with `partial_leaf_sum` rather than emit a short sum. Extensions
       are the genuinely unclassifiable population: §4b.4 measured and refuted every
       structural rule for them (a negative-weight extension child of the investing node
       admits `apa:EquityMethodInvestmentContribution` at $501M,
       `nee:PurchasesOfSecuritiesInSpecialUseFunds` at $1.4-2.6bn, `dte:ConsolidationOfVIES`
       and `eog:ChangesInComponentsOfWorkingCapital...`). Without the guard, SWKS `depAmort`
       reads $392M against ~$618M (**-37%**), and MAA, APA and PLD lose an extension capex
       leg in 60 of 63, 21 of 22 and 33 filings respectively.

    A REFUSAL IS NOT A NULL: `_resolve_once` carries it forward and still tries routes 4
    and 5, so a filer whose extension is unclassified keeps whatever the tag list can give
    it and the reason code only lands when nothing else answers. That is what makes the
    guard safe to leave strict.
    """
    roll_up = _roll_up(spec, regime)
    declared = roll_up.get("any_of") or ()
    # First, because this route applies to 3 of the 48 fields and the other 45 were paying
    # for the anchor and role reads below only to learn that they do not.
    if not declared and not filer_leaves:
        return (), None, None
    groups = [list(g) for g in declared] + [list(g) for g in filer_leaves]
    # A LIST of anchors, because FASB ships two spellings of the same node and filers split
    # between them: `NetCashProvidedByUsedInInvestingActivities` and
    # `...InvestingActivitiesContinuingOperations`. Their children are unioned rather than
    # tried in order -- a filer with discontinued operations declares both, and its capex
    # can sit under either.
    anchors = roll_up.get("anchor") or []
    anchors = [anchors] if isinstance(anchors, str) else list(anchors)
    role = ANCHOR_ROLES.get(str(roll_up.get("anchor_role", "")))
    if not groups or not anchors or role is None:
        return (), None, None

    declared = [(anchor, *pair) for anchor in anchors
                for pair in graph.children_on_role(bare(anchor), role)]
    if not declared:
        return (), None, None
    want = 1.0 if float(roll_up.get("leaf_weight", -1.0)) >= 0 else -1.0
    # Sign FIRST, de-duplicate after -- see `children_on_role`.
    aligned: dict[str, tuple[float, str, str]] = {}
    for anchor, child, weight, arc_role in declared:
        if weight * want > 0 and child not in aligned:
            aligned[child] = (weight, bare(anchor), arc_role)

    classified = {bare(c) for group in groups for c in group}
    classified |= {bare(c) for c in filer_not_leaves}
    # `never_use` bans a concept as a standalone TOTAL, which is a different claim from
    # banning it as a LEG -- bank `totalRevenue` bans `NoninterestIncome` as a total while
    # keeping it a child of the regime roll-up, and MAA's `PaymentsForCapitalImprovements`
    # and `PaymentsToAcquireInProcessResearchAndDevelopment` are exactly that case for
    # capex. So a `never_use` entry does NOT disqualify a leaf; it is only consulted as
    # evidence that the concept is a KNOWN one, i.e. not an unclassifiable sibling.
    classified |= {bare(c) for c in spec.never_use(regime)}
    classified |= {bare(c) for c in (candidates if candidates is not None
                                     else _candidates(spec, regime))}

    # Weight 1.0, not the filer's declared one, because `_materialise` MULTIPLIES what it
    # is given and the declared weight is an admission test rather than a coefficient: a
    # payment leaf is tagged positive and carries -1.0 only because it reduces net
    # investing cash, so passing -1.0 through would make a non-negative field negative.
    picked: list[tuple[str, float]] = []
    for group in groups:
        for concept in group:
            name = bare(concept)
            if name in aligned and name in available:
                picked.append((name, 1.0))
                break                                   # first reported alternative only
    if not picked:
        return (), None, None

    unclassified = sorted(
        name for name in aligned
        if name not in classified and graph.taxonomy_of(name) != "us-gaap"
        and name in available)
    if unclassified:
        return (), PARTIAL_LEAF_SUM, None
    return tuple(picked), None, aligned[picked[0][0]][1:]


def _linkbase_weights(graph: ArcGraph, children: list[str], available: frozenset[str],
                      totals: frozenset[str] | set[str] = frozenset(),
                      ) -> tuple[tuple[str, float], ...]:
    """Pair each declared child with the weight the FILER gave it -- but ONLY where that
    weight is about this field.

    A calculation weight says how a concept foots into ITS OWN PARENT. The catalogue's
    `roll_up.sum` says how a set of legs foots into THIS FIELD. Those are the same
    statement only when the legs' shared parent IS the field's declared total, and taking
    the weight when they are not is a category error with a sign attached:

      **MSFT** declares `SellingAndMarketingExpense` and `GeneralAndAdministrativeExpense`
      as **-1.0 children of `OperatingIncomeLoss`** -- correct, they reduce operating
      income -- while `sellingGeneralAdmin`'s own total is
      `SellingGeneralAndAdministrativeExpense`. Applying -1.0 made SG&A **-$34.7 bn on 159
      of its 202 rows**, the single largest wrong number in either 26-ticker sweep, and the
      only field in 1,770 `linkbase_sum` rows affected -- because it is the only one whose
      legs hang off a subtotal that is not its own total.

    So the weight is honoured when the legs share one parent and that parent is a concept
    the field itself claims (`shortTermDebt` -> `DebtCurrent`, bank `totalRevenue` ->
    `RevenuesNetOfInterestExpense` both still qualify), and is +1.0 otherwise. The sign
    stays load-bearing where it means something: 22% of Statements arcs carry -1.0.

    Returns () unless every child is reported: a partial sum is not the total, and
    emitting one is exactly how a field silently loses a leg -- the defect this route
    exists to fix. Better a NULL with a reason code than a plausible wrong number.
    """
    if not all(child in available for child in children):
        return ()
    parents = {graph.parent_of(child) for child in children}
    parent = next(iter(parents)) if len(parents) == 1 else None
    trusted = parent is not None and bare(parent) in {bare(t) for t in totals}
    out: list[tuple[str, float]] = []
    for child in children:
        weight = 1.0
        if trusted:
            weight = next((w for c, w in graph.children_of(parent) if c == child), 1.0)
        out.append((child, weight))
    return tuple(out)
