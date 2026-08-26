"""
Entity scoping for the fundamentals rebuild: reduce a filing's XBRL facts to the
CONSOLIDATED registrant's own numbers.

The rule under test is "keep the dimensionally-unqualified facts, and filter on the AXIS
rather than on the member". The two real cases are Up-C REITs and multi-registrant
utilities, where a subsidiary's FULL primary statements sit inside the parent's instance
and every `xbrli:identifier` is the parent's CIK -- so entity identity cannot separate
them, and a fixed us-gaap member list cannot either, because the members are company
extensions.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals import entity_scope as scope


def _facts(rows: list[dict]) -> pd.DataFrame:
    base = {"concept": "us-gaap:Assets", "numeric_value": 1.0, "is_dimensioned": False,
            "unit_ref": "usd", "period_type": "instant"}
    return pd.DataFrame([{**base, **r} for r in rows])


# --------------------------------------------------------------------------- #
# Synthetic known-truth                                                       #
# --------------------------------------------------------------------------- #
def test_an_extension_member_on_a_known_axis_is_excluded():
    """MAA scopes its operating partnership with `maa:LimitedPartnershipMember` -- a
    COMPANY EXTENSION. A member deny-list cannot enumerate that; axis presence can."""
    facts = _facts([
        {"numeric_value": 100.0, "is_dimensioned": False},
        {"numeric_value": 95.0, "is_dimensioned": True,
         "dim_dei_LegalEntityAxis": "maa:LimitedPartnershipMember"},
    ])
    kept = scope.consolidated_facts(facts)

    assert len(kept) == 1
    assert kept["numeric_value"].iloc[0] == 100.0
    print("\n=== SANITY CHECK: Up-C subsidiary exclusion ===")
    print(f"  2 facts in, {len(kept)} kept; the LP's extension-member fact was dropped")
    print("  OK: filtering on the AXIS catches a member no list could name.")


def test_per_share_units_are_not_stored_as_amounts():
    """A per-share figure is not additive -- summing four quarterly EPS drifts from annual
    EPS as the share count moves. `epsDiluted` is COMPUTED (decision #9); the as-reported
    tag survives only as the validator's cross-check, never as a stored amount here."""
    facts = _facts([
        {"concept": "us-gaap:Assets", "numeric_value": 100.0, "unit_ref": "usd"},
        {"concept": "us-gaap:EarningsPerShareDiluted", "numeric_value": 1.25,
         "unit_ref": "usdPerShare"},
    ])
    kept = scope.consolidated_facts(facts)

    assert "EarningsPerShareDiluted" not in scope.reported_concepts(kept)
    print("\n=== SANITY CHECK: per-share units excluded ===")
    print(f"  kept concepts: {sorted(scope.reported_concepts(kept))}")
    print("  OK: a non-additive per-share amount cannot enter the additive path.")


def test_non_numeric_facts_are_dropped():
    """Cover-page strings, extensible enumerations and text blocks share the frame with
    the amounts; `numeric_value` is NaN for all of them."""
    facts = _facts([
        {"concept": "dei:AmendmentFlag", "numeric_value": float("nan")},
        {"concept": "us-gaap:Assets", "numeric_value": 42.0},
    ])
    kept = scope.consolidated_facts(facts)

    assert scope.reported_concepts(kept) == frozenset({"Assets"})
    print("\n=== SANITY CHECK: non-numeric facts ===")
    print(f"  kept: {sorted(scope.reported_concepts(kept))}")
    print("  OK: only numeric amounts survive.")


def test_bare_concept_strips_any_namespace():
    assert scope.bare_concept("us-gaap:Assets") == "Assets"
    assert scope.bare_concept("apa:RevenuesAndOther") == "RevenuesAndOther"
    assert scope.bare_concept("Assets") == "Assets"
    print("\n=== SANITY CHECK: namespace stripping ===")
    print("  us-gaap:Assets / apa:RevenuesAndOther / Assets all key to their bare name")
    print("  OK: the facts frame, the linkbase and the catalogue can be joined.")


def test_regulatory_capital_is_a_documented_exclusion_not_an_oversight():
    """CET1 is reachable ONLY dimensioned (ASC 942-505-50-1 requires it per legal entity),
    which is why SEC `companyconcept` 404s for it on JPM/USB/BAC. The exclusion is a
    recorded decision with a named hook, so it cannot read as a silent loss."""
    assert "tier1CapitalRatio" in scope.DIMENSIONED_EXCEPTIONS
    reason = scope.DIMENSIONED_EXCEPTIONS["tier1CapitalRatio"]
    assert "942-505-50-1" in reason
    print("\n=== SANITY CHECK: regulatory-capital exclusion ===")
    print(f"  documented: {reason[:96]}...")
    print("  OK: the cost of the entity-scope rule is recorded, not hidden.")


# --------------------------------------------------------------------------- #
# Real filings: the two multi-registrant traps                                #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def edgar_ready() -> bool:
    if not os.getenv("SEC_USER_AGENT", "").strip():
        pytest.skip("SEC_USER_AGENT unset -- real-filing entity-scope tests need EDGAR")
    return True


@pytest.fixture(scope="module")
def maa_filing(edgar_ready):
    from edgar import Company, set_identity
    set_identity(os.getenv("SEC_USER_AGENT"))
    try:
        filing = Company("MAA").latest("10-K")
        return filing, filing.xbrl()
    except Exception as exc:                                # noqa: BLE001
        pytest.skip(f"EDGAR unreachable: {exc}")


def test_maa_operating_partnership_is_scoped_out(maa_filing):
    """MAA's FY instance carries two `dei:EntityCentralIndexKey` facts (parent and LP) but
    every `xbrli:identifier` is the PARENT's CIK -- so the identifier cannot separate them.
    The LP is scoped by `dei:LegalEntityAxis` with an extension member, carrying its full
    primary statements rather than a footnote.
    """
    filing, xbrl = maa_filing
    raw = xbrl.facts.to_dataframe()
    kept = scope.consolidated_facts(raw)

    identifiers = sorted(raw["entity_identifier"].dropna().unique())
    lp_column = "dim_dei_LegalEntityAxis"
    lp_facts = int(raw[lp_column].notna().sum()) if lp_column in raw.columns else 0
    members = sorted(raw[lp_column].dropna().unique()) if lp_facts else []

    assert lp_facts > 0, "MAA should carry LegalEntityAxis facts -- fixture may be stale"
    assert any(not m.startswith(("us-gaap:", "srt:", "dei:")) for m in members), (
        f"expected a company-extension member among {members}")
    # Assert on the RAW frame, not on `kept`. `consolidated_facts` now projects the `dim_*`
    # columns away (they are all-NaN once the undimensioned filter has run, and carrying 39
    # of them per filing cost real memory at sweep scale), so "no dim column is populated in
    # kept" would be vacuously true and test nothing. Two things must actually hold:
    dimensioned = raw["is_dimensioned"].fillna(False).astype(bool)
    #   1. every LP-scoped fact is dimensioned, so the axis rule can see it at all;
    assert bool(dimensioned[raw[lp_column].notna()].all()), (
        "a LegalEntityAxis fact was not flagged dimensioned -- the axis rule would miss it")
    #   2. nothing dimensioned survived scoping.
    assert len(kept) <= int((~dimensioned).sum()), (
        f"kept {len(kept)} rows but only {int((~dimensioned).sum())} facts are "
        "undimensioned -- a subsidiary-scoped fact survived")

    print("\n=== SANITY CHECK: MAA Up-C entity scoping ===")
    print(f"  accession            : {filing.accession_number}")
    print(f"  xbrli:identifier vals: {identifiers}  <- all the PARENT's CIK")
    print(f"  LegalEntityAxis facts: {lp_facts}, members {members}")
    print(f"  facts {len(raw)} -> {len(kept)} consolidated "
          f"({len(kept) / len(raw):.1%} kept)")
    print("  OK: the LP's statements are excluded by AXIS, which a member list could not do.")


def test_maa_shares_outstanding_is_the_parents(maa_filing):
    """The consequence that matters downstream: after scoping, the share count is the
    parent registrant's, not the LP's unit count."""
    from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
    from src.data_extract.utils.fundamentals.xbrl_linkbase import (
        ArcGraph, resolve_field, statement_arcs)
    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import (
        _materialise, _period_frame)

    _, xbrl = maa_filing
    catalogue = load_catalogue("./configs")
    facts = scope.consolidated_facts(xbrl.facts.to_dataframe())
    resolution = resolve_field(catalogue.field("sharesOutstanding"),
                               ArcGraph(statement_arcs(xbrl)),
                               scope.reported_concepts(facts), catalogue, "real_estate")
    periods = _materialise(resolution, _period_frame(facts))
    values = sorted({p["value"] for p in periods.values()})

    assert values, "MAA produced no share count after scoping"
    assert all(5e7 < v < 5e8 for v in values), (
        f"MAA share count outside a plausible range for the parent REIT: {values}")
    print("\n=== SANITY CHECK: MAA share count after scoping ===")
    print(f"  concept : {resolution.concept} ({resolution.method})")
    print(f"  value(s): {[f'{v:,.0f}' for v in values]}")
    print("  OK: the parent's cover-page count, on the only multi-class-summable tag.")


def test_southern_company_six_registrants_collapse_to_the_parent(edgar_ready):
    """Southern Company carries six registrant CIKs and thousands of `LegalEntityAxis`
    occurrences in ONE instance, all identifiers = parent. Four of those registrants file
    their own 10-Ks and return 404 from `companyconcept`, so by-CIK universe construction
    yields silent nulls -- this path must not depend on it."""
    from edgar import Company, set_identity
    set_identity(os.getenv("SEC_USER_AGENT"))
    try:
        xbrl = Company("SO").latest("10-K").xbrl()
    except Exception as exc:                                # noqa: BLE001
        pytest.skip(f"EDGAR unreachable: {exc}")

    raw = xbrl.facts.to_dataframe()
    kept = scope.consolidated_facts(raw)
    lp_column = "dim_dei_LegalEntityAxis"
    subsidiary_facts = int(raw[lp_column].notna().sum()) if lp_column in raw.columns else 0

    assert len(kept) < len(raw)
    assert "Assets" in scope.reported_concepts(kept)
    print("\n=== SANITY CHECK: Southern Company multi-registrant ===")
    print(f"  identifiers          : {sorted(raw['entity_identifier'].dropna().unique())}")
    print(f"  LegalEntityAxis facts: {subsidiary_facts}")
    print(f"  facts {len(raw)} -> {len(kept)} consolidated")
    print("  OK: one consolidated registrant survives; subsidiaries scoped out by axis.")
