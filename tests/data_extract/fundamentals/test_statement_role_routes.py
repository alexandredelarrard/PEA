"""
Phase 4c.1: the statement-role test on the single-concept resolution routes.

A concept the filer declares ONLY outside its face statements is not the field, however
high it sits in the candidate list. Route 3b (`statement_leaf_sum`) has carried that guard
since Phase 4b -- which is precisely why it is the safe route -- and this is the same guard
generalised to routes 1 (`linkbase_total`) and 5 (`tag_fallback`).

Split per docs/testing.md. The synthetic half proves the THREE-WAY decision -- reject when
every declared role is a note role, keep when any declared role is a face statement, keep
when the concept is undeclared -- because a real filing cannot be made to present the three
cases on demand. The real half proves it fires on the confirmed live instances and, just as
importantly, that it does NOT fire on the two 4b.4 regression guards.

The load-bearing asymmetry: **silence is never evidence.** An undeclared concept is
unaffected, because a leaf (`goodwill`) or a `dei:` cover-page tag can never carry a
calculation arc and `tag_primary` is its normal home. 3c.8 cost four defects to learn that
reading "the linkbase says nothing" as licence to act is the expensive direction.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals import entity_scope as scope
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.data_extract.utils.fundamentals.xbrl_linkbase import (
    LINKBASE_TOTAL, STATEMENT_LEAF_SUM, TAG_PRIMARY, ArcGraph, is_note_only,
    resolve_field, statement_arcs)

CATALOGUE = load_catalogue("./configs")

_ARC_COLS = ["concept", "concept_taxonomy", "parent_concept", "parent_taxonomy",
             "weight", "role_uri", "menucat", "is_abstract", "arc_filter"]

#: A face-statement role URI. Deliberately a real filer spelling -- the test is about role
#: STRINGS, so a placeholder like `.../role/BS` would prove nothing about the pattern.
_BALANCE_SHEET_ROLE = "http://x/role/ConsolidatedBalanceSheets"

#: A footnote role URI. `menucat` is left at "Statements" on purpose: that mis-categorisation
#: is exactly how a note arc reaches the resolver at all (`statement_arcs` is a UNION, so a
#: menucat hit admits an arc whose role URI reads as a note), and a fixture that filtered it
#: out at the door would be testing nothing.
_DEBT_NOTE_ROLE = "http://x/role/DebtDisclosureScheduleOfLongTermDebtDetail"


def _arcs(rows: list[tuple[str, str, str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"concept": c, "concept_taxonomy": tax, "parent_concept": p,
          "parent_taxonomy": "us-gaap", "weight": w, "role_uri": role,
          "menucat": "Statements", "is_abstract": False, "arc_filter": "both"}
         for c, tax, p, w, role in rows],
        columns=_ARC_COLS)


# --------------------------------------------------------------------------- #
# Synthetic known-truth: the three-way decision                               #
# --------------------------------------------------------------------------- #
def test_a_note_only_concept_loses_to_a_lower_priority_statement_line():
    """AMT's defect, in a fixture: `longTermDebt`'s priority-1 concept is declared only in
    the debt note, so it must lose to the priority-3 concept that is on the balance sheet.

    Live, that is **$1.9M against $21,127M** -- four orders of magnitude, and entirely
    invisible to any level or footing check, because $1.9M is a perfectly plausible number.
    """
    graph = ArcGraph(_arcs([
        ("LongTermDebtNoncurrent", "us-gaap", "DebtTotal", 1.0, _DEBT_NOTE_ROLE),
        ("LongTermDebtAndCapitalLeaseObligations", "us-gaap", "Liabilities", 1.0,
         _BALANCE_SHEET_ROLE),
    ]))
    available = frozenset({"LongTermDebtNoncurrent",
                           "LongTermDebtAndCapitalLeaseObligations"})

    resolution = resolve_field(CATALOGUE.field("longTermDebt"), graph, available,
                               CATALOGUE)

    assert is_note_only(graph, "LongTermDebtNoncurrent")
    assert resolution.concept == "us-gaap:LongTermDebtAndCapitalLeaseObligations"
    assert resolution.role_rejected == ("us-gaap:LongTermDebtNoncurrent",)
    assert not resolution.role_only_retained
    print("\n=== SANITY CHECK: note-only concept rejected (routes 1/5) ===")
    print(f"  priority 1 : LongTermDebtNoncurrent, declared ONLY on {_DEBT_NOTE_ROLE}")
    print(f"  resolved to: {resolution.concept} via {resolution.method}")
    print(f"  ledger     : role_rejected={list(resolution.role_rejected)}")
    print("  OK: the statement line won, and the rejection is recorded on the row.")


def test_a_concept_on_both_a_note_and_a_statement_is_kept():
    """One declared face-statement role is enough. A filer re-presenting a balance-sheet
    line inside its own footnote is the NORM, not a defect -- rejecting on "any note role"
    rather than "every role is a note role" would delete most of the balance sheet."""
    graph = ArcGraph(_arcs([
        ("LongTermDebtNoncurrent", "us-gaap", "DebtTotal", 1.0, _DEBT_NOTE_ROLE),
        ("LongTermDebtNoncurrent", "us-gaap", "Liabilities", 1.0, _BALANCE_SHEET_ROLE),
    ]))

    resolution = resolve_field(CATALOGUE.field("longTermDebt"), graph,
                               frozenset({"LongTermDebtNoncurrent"}), CATALOGUE)

    assert not is_note_only(graph, "LongTermDebtNoncurrent")
    assert resolution.method == LINKBASE_TOTAL
    assert resolution.concept == "us-gaap:LongTermDebtNoncurrent"
    assert resolution.role_rejected == ()
    print("\n=== SANITY CHECK: declared on a note AND a statement -> kept ===")
    print(f"  roles declared: {sorted(graph.roles_of('LongTermDebtNoncurrent'))}")
    print(f"  resolved to   : {resolution.concept} via {resolution.method}")
    print("  OK: one face-statement role suffices; a footnote re-presentation is normal.")


def test_an_undeclared_concept_is_unaffected_because_silence_is_not_evidence():
    """The 3c.8 lesson, asserted. `goodwill` is a LEAF: a calculation arc exists only where
    a filer declares a total-and-components relationship, so most balance-sheet leaves and
    every `dei:` cover-page tag are undeclared by construction and `tag_primary` is their
    correct home. A guard that read absence as note-hood would null them all."""
    graph = ArcGraph(_arcs([
        ("Goodwill", "us-gaap", "Assets", 1.0, _BALANCE_SHEET_ROLE),
    ]))

    resolution = resolve_field(CATALOGUE.field("longTermDebt"), graph,
                               frozenset({"LongTermDebtNoncurrent"}), CATALOGUE)

    assert graph.roles_of("LongTermDebtNoncurrent") == frozenset()
    assert not is_note_only(graph, "LongTermDebtNoncurrent")
    assert resolution.method == TAG_PRIMARY
    assert resolution.concept == "us-gaap:LongTermDebtNoncurrent"
    print("\n=== SANITY CHECK: undeclared concept unaffected ===")
    print(f"  LongTermDebtNoncurrent declared on: "
          f"{sorted(graph.roles_of('LongTermDebtNoncurrent')) or 'nothing'}")
    print(f"  resolved to: {resolution.concept} via {resolution.method}")
    print("  OK: silence is not evidence -- the guard fires on positive note-hood only.")


def test_a_note_only_concept_is_kept_when_it_is_the_filers_whole_answer():
    """The guard costs no coverage. Where every candidate is note-only and no other route
    answers, `resolve_field` puts the rejections back and flags `role_only_retained`.

    Deliberate, and the precedent is this repo's own: the 2026-08 audit measured **745
    correct rows nulled** by over-strict Q4 guards. A narrow real number that is flagged
    beats a null, and `basis_step` is the check that reads the flag.
    """
    graph = ArcGraph(_arcs([
        ("LongTermDebtNoncurrent", "us-gaap", "DebtTotal", 1.0, _DEBT_NOTE_ROLE),
    ]))

    resolution = resolve_field(CATALOGUE.field("longTermDebt"), graph,
                               frozenset({"LongTermDebtNoncurrent"}), CATALOGUE)

    assert resolution.resolved
    assert resolution.concept == "us-gaap:LongTermDebtNoncurrent"
    assert resolution.role_only_retained
    assert resolution.role_rejected == ("us-gaap:LongTermDebtNoncurrent",)
    print("\n=== SANITY CHECK: no coverage lost when the note IS the answer ===")
    print(f"  resolved to        : {resolution.concept} via {resolution.method}")
    print(f"  role_only_retained : {resolution.role_only_retained}")
    print("  OK: the value survives, flagged. The guard reorders; it never deletes.")


# --------------------------------------------------------------------------- #
# Real filings: the confirmed instances, and the two regression guards         #
# --------------------------------------------------------------------------- #
#: (ticker, field, why). The four defects the 2026-08-22 audit confirmed by hand, plus the
#: two 4b.4 regression guards whose winning concept is legitimately on a face statement and
#: must NOT move.
_GROUND_TRUTH = [
    ("AMT", "longTermDebt", "must not be the debt note's instrument-level amount"),
    ("CSCO", "depAmort", "must be the cash-flow line, not the PP&E note"),
    ("MCD", "capex", "must be the cash-flow leaf sum for the WHOLE history"),
    ("PG", "totalRevenue", "must be the income statement's top line"),
    ("AAPL", "depAmort", "REGRESSION GUARD: must stay on the cash-flow line"),
    ("VLO", "capex", "REGRESSION GUARD: must stay where 4b.4 put it"),
]

#: GICS for the six, so `regime_for` routes each filing to its own statement template
#: without a DB read. AMT is the deliberate trap: a Telecom Tower REIT files like an
#: industrial (`AssetsCurrent`, `OperatingIncomeLoss`, PP&E capex), so the forced override
#: pulls it OUT of the real_estate industry-group claim.
_GICS = {
    "AMT": ("Real Estate", "Equity Real Estate Investment Trusts (REITs)",
            "Telecom Tower REITs"),
    "CSCO": ("Information Technology", "Technology Hardware & Equipment",
             "Communications Equipment"),
    "MCD": ("Consumer Discretionary", "Consumer Services", "Restaurants"),
    "PG": ("Consumer Staples", "Household & Personal Products", "Household Products"),
    "AAPL": ("Information Technology", "Technology Hardware & Equipment",
             "Technology Hardware, Storage & Peripherals"),
    "VLO": ("Energy", "Energy", "Oil & Gas Refining & Marketing"),
}


@pytest.fixture(scope="module")
def edgar_ready() -> bool:
    if not os.getenv("SEC_USER_AGENT", "").strip():
        pytest.skip("SEC_USER_AGENT unset -- real-filing role tests need EDGAR")
    return True


@pytest.fixture(scope="module")
def latest_annual(edgar_ready) -> dict:
    """Resolve against each ticker's latest 10-K. Module-scoped: one `filing.xbrl()` costs
    1.4-5.8 s and six of them is this file's whole budget."""
    from edgar import Company, set_identity
    set_identity(os.getenv("SEC_USER_AGENT"))

    out: dict[str, dict] = {}
    for ticker, _field, _why in _GROUND_TRUTH:
        if ticker in out:
            continue
        try:
            filing = Company(ticker).latest("10-K")
            xbrl = filing.xbrl()
        except Exception as exc:                                    # noqa: BLE001
            pytest.skip(f"EDGAR unreachable for {ticker}: {exc}")
        facts = scope.consolidated_facts(xbrl.facts.to_dataframe())
        graph = ArcGraph(statement_arcs(xbrl))
        sector, group, sub = _GICS[ticker]
        out[ticker] = {
            "graph": graph, "accession": filing.accession_number,
            "available": scope.reported_concepts(facts),
            "regime": CATALOGUE.regime_for(
                {"sector": sector, "industry_group": group, "sub_industry": sub},
                [str(r) for r in graph.arcs.get("role_uri", pd.Series(dtype=str))]),
        }
    return out


def test_no_field_resolves_onto_a_note_only_concept(latest_annual):
    """The real-data half: every affected field resolves onto a face-statement basis, and
    neither 4b.4 regression guard is disturbed.

    Asserted as a PROPERTY rather than against hard-coded figures, because the figures move
    every year with the filer's own latest 10-K: the resolved concept must not be one this
    filer declares only outside its face statements. That is exactly the invariant 4c.1
    buys, and it is checkable on whatever the latest filing happens to be.
    """
    print("\n=== SANITY CHECK: statement-role test on six real 10-Ks ===")
    print(f"  {'ticker':7s} {'field':14s} {'route':19s} concept")
    failures = []
    for ticker, field, why in _GROUND_TRUTH:
        case = latest_annual[ticker]
        graph = case["graph"]
        resolution = resolve_field(CATALOGUE.field(field), graph, case["available"],
                                   CATALOGUE, case["regime"], ticker=ticker)
        concept = resolution.concept or "+".join(c for c, _ in resolution.children)
        print(f"  {ticker:7s} {field:14s} {resolution.method:19s} {concept}")
        if resolution.role_rejected:
            print(f"          withheld : {list(resolution.role_rejected)}")
        if resolution.role_only_retained:
            print("          note-only retained -- no face-statement alternative existed")
        name = (resolution.concept or "").split(":")[-1]
        if name and is_note_only(graph, name) and not resolution.role_only_retained:
            failures.append(f"{ticker}.{field} resolved to a note-only concept: {why}")
    assert not failures, failures
    print("  OK: no field resolved onto a concept its filer declares only in the notes.")


def test_mcd_capex_resolves_on_the_cash_flow_leaf_sum(latest_annual):
    """MCD is the motivating case, and the one no cross-vintage test can see: MCD tags the
    same narrow concept CONSISTENTLY in its earlier era, so only the route boundary betrays
    it -- a **35.6x** step at the single 2017->2018 filing where route 3b takes over.

    After 4c.1 the note concept loses at route 1 for the whole history, so route 3b's
    cash-flow leaf sum is the basis in both eras and the step has nothing to step across.
    """
    case = latest_annual["MCD"]
    resolution = resolve_field(CATALOGUE.field("capex"), case["graph"], case["available"],
                               CATALOGUE, case["regime"], ticker="MCD")
    print("\n=== SANITY CHECK: MCD capex basis ===")
    print(f"  accession : {case['accession']}")
    print(f"  route     : {resolution.method}")
    print(f"  basis     : {[c for c, _ in resolution.children] or resolution.concept}")
    print(f"  withheld  : {list(resolution.role_rejected) or 'nothing'}")
    assert resolution.method == STATEMENT_LEAF_SUM, (
        "MCD capex must come off the cash-flow statement, not the PP&E note")
    print("  OK: one basis for the whole history -- the 2017->2018 step cannot occur.")


# --------------------------------------------------------------------------- #
# The DECLAREDNESS half of 4c.1 -- the one that actually fires                 #
# --------------------------------------------------------------------------- #
#: The cash-flow role `depAmort`'s anchor lives on. Must read like a real filer's, because
#: `roll_up.anchor_role: cash_flow` matches on `cash[\s_-]*flow` and the arc is only admitted
#: if `NON_STATEMENT_ROLE` does NOT match it.
_CASH_FLOW_ROLE = "http://x/role/ConsolidatedStatementsOfCashFlows"
_OPERATING_NODE = "NetCashProvidedByUsedInOperatingActivities"


def test_an_undeclared_tag_loses_to_the_filers_own_declared_statement_lines():
    """The measured half of 4c.1, and the one the role test cannot reach.

    CSCO tags `us-gaap:DepreciationDepletionAndAmortization` at **exactly $700,000,000 with
    `decimals=-8` in fiscal 2023, 2024 AND 2025** -- the same rounded narrative figure three
    years running -- while its cash-flow D&A line is `csco:DepreciationAmortizationAndOther`
    at **$2,811M**, `decimals=-6`. The bad concept carries NO calculation arc at all, so
    `is_note_only` is silent on it by construction; the good one is a company extension, so
    no candidate list can name it. Only "the filer declares its own lines and does not
    declare this" separates them.
    """
    graph = ArcGraph(_arcs([
        ("Depreciation", "us-gaap", _OPERATING_NODE, 1.0, _CASH_FLOW_ROLE),
        ("AmortizationOfIntangibleAssets", "us-gaap", _OPERATING_NODE, 1.0,
         _CASH_FLOW_ROLE),
    ]))
    available = frozenset({"DepreciationDepletionAndAmortization",   # priority 1, UNdeclared
                           "Depreciation", "AmortizationOfIntangibleAssets"})

    resolution = resolve_field(CATALOGUE.field("depAmort"), graph, available, CATALOGUE,
                               "industrial", ticker="TEST")

    assert not graph.knows("DepreciationDepletionAndAmortization")
    assert resolution.method == STATEMENT_LEAF_SUM
    assert [c for c, _ in resolution.children] == ["Depreciation",
                                                   "AmortizationOfIntangibleAssets"]
    assert resolution.undeclared_rejected == (
        "us-gaap:DepreciationDepletionAndAmortization",)
    print("\n=== SANITY CHECK: an undeclared tag loses to the declared statement lines ===")
    print(f"  priority 1 : DepreciationDepletionAndAmortization -- reported, NO arc anywhere")
    print(f"  the filer declares: Depreciation + AmortizationOfIntangibleAssets on")
    print(f"                      {_CASH_FLOW_ROLE}")
    print(f"  resolved to: {resolution.method} {[c for c, _ in resolution.children]}")
    print(f"  ledger     : undeclared_rejected={list(resolution.undeclared_rejected)}")
    print("  OK: the filer's own statement beats a bare tag hit.")


def test_a_declared_candidate_still_beats_the_leaf_sum():
    """The AAPL / PG / MCD regression guard, and the reason the rule keys on DECLAREDNESS
    rather than on precision or on route order.

    AAPL declares `DepreciationDepletionAndAmortization` on its cash-flow statement at
    $11,698M, and it is also the field's priority-1 concept. Nothing about it is wrong, so
    the filer's declared TOTAL must keep beating a sum of the filer's own leaves -- which is
    the general contract this repo already holds: the legs are what the total is checked
    against, never a substitute for reading it.
    """
    graph = ArcGraph(_arcs([
        ("DepreciationDepletionAndAmortization", "us-gaap", _OPERATING_NODE, 1.0,
         _CASH_FLOW_ROLE),
        ("Depreciation", "us-gaap", _OPERATING_NODE, 1.0, _CASH_FLOW_ROLE),
    ]))
    available = frozenset({"DepreciationDepletionAndAmortization", "Depreciation"})

    resolution = resolve_field(CATALOGUE.field("depAmort"), graph, available, CATALOGUE,
                               "industrial", ticker="TEST")

    assert resolution.method == LINKBASE_TOTAL
    assert resolution.concept == "us-gaap:DepreciationDepletionAndAmortization"
    assert resolution.undeclared_rejected == ()
    print("\n=== SANITY CHECK: a DECLARED candidate keeps winning ===")
    print(f"  resolved to: {resolution.concept} via {resolution.method}")
    print("  OK: the rule fires on absence of an arc, not on the existence of leaves.")


def test_a_field_with_no_leaf_sum_route_is_untouched():
    """The XOM guard, which is why the scope is two fields and not forty-nine.

    Phase 3 set "candidate PRIORITY dominates linkbase presence" because requiring a linkbase
    hit let XOM's `CommonStockSharesOutstanding` -- a SINGLE share class, declared under
    `CommonStockSharesIssued` -- beat the `dei:` cover-page tag, which is the only summable
    one for a multi-class issuer. `sharesOutstanding` and `goodwill` declare no
    `roll_up.any_of`, so route 3b can never fire for them and 4c.1 cannot reach them.
    """
    graph = ArcGraph(_arcs([
        ("Depreciation", "us-gaap", _OPERATING_NODE, 1.0, _CASH_FLOW_ROLE),
    ]))
    resolution = resolve_field(CATALOGUE.field("goodwill"), graph,
                               frozenset({"Goodwill", "Depreciation"}), CATALOGUE,
                               "industrial", ticker="TEST")
    assert resolution.method == TAG_PRIMARY
    assert resolution.concept == "us-gaap:Goodwill"
    assert resolution.undeclared_rejected == ()
    print("\n=== SANITY CHECK: a field with no any_of roll-up is out of scope ===")
    print(f"  goodwill roll_up: {CATALOGUE.field('goodwill').raw.get('roll_up')}")
    print(f"  resolved to: {resolution.concept} via {resolution.method}")
    print("  OK: only capex and depAmort are eligible; sharesOutstanding cannot regress.")
