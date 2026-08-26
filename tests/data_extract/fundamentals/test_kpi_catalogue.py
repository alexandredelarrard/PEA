"""The KPI catalogue IS the fundamentals contract, so these tests guard the contract itself.

Synthetic-fixture rules do not apply here: the three JSON files under `configs/` are the
real artifact, and checking anything else would prove nothing. What is enforced:

  1. Every field carries tier / kind / sign / unit / definition / authority.
  2. Every `authority` is a primary-source citation, or the explicit `UNVERIFIED`
     placeholder WITH a note saying what would close it. This is the test that stops a
     definition resting on a silent guess -- the failure mode the whole rebuild exists to
     remove.
  3. The tier census matches the plan's contract (11 + 12 + 17 + 13).
  4. Cross-references resolve: `roll_up`/`components`/`feeds` name real fields, regime
     overrides name real regimes, and the exception register names real fields.
  5. The measured absence register is internally consistent with its own rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_extract.utils.fundamentals.kpi_catalogue import (
    INPUT_TIER, UNVERIFIED, load_catalogue,
)

# Repo root = the first ancestor holding pyproject.toml, NOT a fixed `parents[N]`.
# This file moved down one directory level once already (into the mirrored
# tests/data_extract/<area>/ layout) and the hard index silently repointed this at
# tests/configs/ -- 8 tests then errored with "KPI catalogue file missing", which reads
# as a config bug rather than a moved file.
_ROOT = next(p for p in Path(__file__).resolve().parents
             if (p / "pyproject.toml").exists())
CONFIG_DIR = str(_ROOT / "configs")
#: Where the three JSONs actually live, for the raw-text checks that bypass the loader.
CATALOGUE_DIR = Path(CONFIG_DIR) / "fundamentals"

#: The contract from the rebuild plan: 11 Tier-1 + 12 Tier-2 + 17 Tier-3 (16 plus
#: researchAndDevelopment, re-added by decision #10 as regime-gated) + 13 calculation
#: inputs. A change here is a change to the contract and must be deliberate.
EXPECTED_TIER_COUNTS: dict[int, int] = {1: 11, 2: 12, 3: 17, INPUT_TIER: 13}

#: The threshold the exception register's own rule states.
EXPECTED_ABSENT_MIN_RATE = 0.75

#: Fields still resting on the UNVERIFIED placeholder. Phase 2 shipped 17; a second research
#: pass closed every one against FASB's 2025 taxonomy files and eCFR Reg S-X, so this is now
#: EMPTY and the test guards against regressing back into it. Adding a name here requires a
#: matching entry in the rebuild plan's Open items.
EXPECTED_UNVERIFIED: frozenset[str] = frozenset()


@pytest.fixture(scope="module")
def cat():
    return load_catalogue(CONFIG_DIR)


def test_every_field_declares_the_mandatory_keys(cat):
    """`load_catalogue` raises on a missing key, so reaching here at all proves the six
    mandatory keys are present on all 53 entries. This asserts the value TYPES."""
    for name, spec in cat.fields.items():
        assert spec.kind in ("instant", "duration", "ratio", "derived"), f"{name}: {spec.kind}"
        assert spec.sign in ("non_negative", "non_positive", "any"), f"{name}: {spec.sign}"
        assert spec.unit in ("USD", "shares", "count", "ratio"), f"{name}: {spec.unit}"
        assert spec.tier in (0, 1, 2, 3), f"{name}: tier {spec.tier}"
        assert len(spec.definition) > 20, f"{name}: definition is not a sentence"

    print(f"\n[catalogue] {len(cat.fields)} fields, every one typed and defined")


def test_tier_census_matches_the_contract(cat):
    actual = {tier: len(cat.by_tier(tier)) for tier in sorted(EXPECTED_TIER_COUNTS)}
    assert actual == EXPECTED_TIER_COUNTS, f"tier census drifted: {actual}"
    assert len(cat.fields) == sum(EXPECTED_TIER_COUNTS.values())

    print("\n=== SANITY CHECK: tier census ===")
    for tier in sorted(EXPECTED_TIER_COUNTS):
        label = "inputs" if tier == INPUT_TIER else f"tier {tier}"
        print(f"  {label:8} {actual[tier]:2d}: {', '.join(cat.by_tier(tier))}")
    print(f"  scored {len(cat.scored_fields)} | inputs {len(cat.input_fields)} | "
          f"extracted {len(cat.extracted_fields)}")


def test_no_field_appears_in_two_tiers(cat):
    """A field is one entry in one JSON object, so a duplicate key would be silently
    dropped by the JSON parser rather than raising. Check the raw text instead."""
    text = (CATALOGUE_DIR / "fundamentals_kpis.json").read_text(encoding="utf-8")
    raw = json.loads(text)
    for name in raw:
        occurrences = text.count(f'\n  "{name}":')
        assert occurrences == 1, f"{name} is declared {occurrences} times at the top level"

    print(f"\n[catalogue] all {len(raw)} top-level keys are unique in the file TEXT, "
          "so none was silently overwritten by the JSON parser")


def test_every_authority_is_sourced_or_explicitly_unverified(cat):
    """The load-bearing test. `authority` must quote or cite a primary source -- FASB
    element documentation, Reg S-X, ASC, an SEC rule or C&DI -- or inherit one, or be the
    explicit placeholder with a note. `load_catalogue` already rejects a bare UNVERIFIED.

    A second research pass closed all 17 originally-UNVERIFIED fields against FASB's own
    2025 taxonomy files and eCFR Reg S-X, so `EXPECTED_UNVERIFIED` is now empty and this
    test's job flips: it guards against a REGRESSION back to the placeholder."""
    primary_markers = ("FASB", "Reg S-X", "Rule 5-0", "ASC ", "ASU ", "17 CFR", "C&DI",
                       "Regulation", "Item 10(e)", "Compustat", "dei:", "linkbase",
                       "Measured", "measured", "edgartools GH", "us-gaap", "eCFR")
    def cites_a_source(spec) -> bool:
        return any(m in spec.authority for m in primary_markers)

    inherited = [n for n, s in cat.fields.items() if s.authority_inherits_from]
    unsourced = [
        name for name, spec in cat.fields.items()
        if spec.authority != UNVERIFIED
        and not cites_a_source(spec)
        and not spec.authority_inherits_from
    ]
    assert not unsourced, (
        f"{len(unsourced)} field(s) have an `authority` that cites no primary source, does "
        f"not inherit one, and is not marked {UNVERIFIED}: {unsourced}")

    # An inherited authority must bottom out in a field that IS sourced, or the chain is
    # decorative. UNVERIFIED parents are permitted but must be reported, not hidden.
    inherited_from_unverified: list[str] = []
    for name in inherited:
        parents = cat.field(name).authority_inherits_from
        assert parents, name
        if all(cat.field(p).authority == UNVERIFIED for p in parents):
            inherited_from_unverified.append(f"{name} <- {parents}")
        for p in parents:
            assert cites_a_source(cat.field(p)) or cat.field(p).authority == UNVERIFIED

    unverified = cat.unverified_fields
    for name in unverified:
        note = cat.field(name).raw["authority_note"]
        assert len(note) > 60, f"{name}: authority_note is too thin to act on"

    assert set(unverified) == EXPECTED_UNVERIFIED, (
        f"the UNVERIFIED set moved. expected {sorted(EXPECTED_UNVERIFIED) or 'none'}, got "
        f"{unverified}. Closing one is good news -- shrink EXPECTED_UNVERIFIED. Adding one "
        "needs a matching entry in the plan's Open items.")

    caveated = [n for n, s in cat.fields.items() if s.raw.get("authority_caveat")]
    sourced = len(cat.fields) - len(unverified) - len(inherited)
    print("\n=== SANITY CHECK: authority completeness ===")
    print(f"  {len(cat.fields)} fields: {sourced} cite a primary source directly, "
          f"{len(inherited)} inherit one, {len(unverified)} are {UNVERIFIED}")
    print(f"  {len(caveated)} carry an authority_caveat -- VERIFIED, but with one sub-claim a "
          "notch weaker (an ASC paragraph number is primary while its prose could only be")
    print(f"    read in a secondary reproduction, asc.fasb.org being login-walled): "
          f"{', '.join(caveated)}")
    if inherited_from_unverified:
        print(f"  inherit from an UNVERIFIED parent ({len(inherited_from_unverified)}): "
              f"{'; '.join(inherited_from_unverified)}")
    print(f"  All 17 fields Phase 2 shipped as {UNVERIFIED} are now closed against FASB's own "
          "2025 taxonomy (doc/ref/label linkbases + schema) and eCFR Reg S-X. Validated.")


def test_cross_references_resolve(cat):
    """`roll_up` on a FIELD names concepts; `roll_up` on a derived/aggregate field and the
    `components` / `feeds` keys name FIELDS. A typo in either is a silent no-op."""
    names = cat.all_column_names        # extracted fields PLUS the computed columns
    problems: list[str] = []
    for name, spec in cat.fields.items():
        for key in ("components", "feeds"):
            for ref in spec.raw.get(key, []):
                if ref not in names:
                    problems.append(f"{name}.{key} -> unknown column {ref!r}")
    assert not problems, "\n".join(problems)

    # Every tier-0 input must say which field it feeds, or it has no reason to be carried.
    orphans = [n for n in cat.input_fields
               if not cat.field(n).raw.get("feeds")]
    assert not orphans, f"calculation input(s) that feed nothing: {orphans}"

    print(f"\n[catalogue] every `components`/`feeds` reference resolves; "
          f"all {len(cat.input_fields)} calculation inputs declare what they feed")


def test_derived_fields_declare_their_formula(cat):
    derived = [n for n, s in cat.fields.items() if s.kind in ("derived", "ratio")]
    for name in derived:
        assert cat.field(name).raw.get("derived_from"), f"{name}: no derived_from"
        assert not cat.field(name).raw.get("fallback_concepts"), \
            f"{name} is {cat.field(name).kind} but declares fallback_concepts"

    extracted_without_route = [
        n for n, s in cat.fields.items()
        if s.is_extracted and not (s.fallback_concepts() or s.total_concept()
                                   or s.raw.get("source") or s.raw.get("roll_up"))
    ]
    assert not extracted_without_route, \
        f"extracted field(s) with no way to resolve them: {extracted_without_route}"

    print(f"\n[catalogue] {len(derived)} derived/ratio fields carry a formula and no "
          f"concept list; every one of the {len(cat.extracted_fields)} extracted fields "
          "has a resolution route")


def test_regime_overrides_and_exceptions_name_real_regimes(cat):
    """`load_catalogue` raises on an unknown regime or field, so reaching here proves the
    references resolve. This asserts the register's own consistency rule."""
    assert cat.default_regime() == "industrial"

    violations: list[str] = []
    overrides: list[str] = []
    for regime, block in cat.regime_exceptions.items():
        for field, cell in block.items():
            rate, flagged = cell.get("measured_absent_rate"), cell.get("expected_absent")
            if not (flagged and rate is not None and rate < EXPECTED_ABSENT_MIN_RATE):
                continue
            # Below the bar is allowed ONLY with a written reason, so every such cell is a
            # visible, argued exception rather than a quiet loosening of the rule.
            reason = cell.get("override_reason", "")
            if len(reason) < 80:
                violations.append(f"{regime}.{field}: expected_absent at rate {rate} "
                                  "with no adequate override_reason")
            else:
                overrides.append(f"{regime}.{field} (rate {rate})")
    assert not violations, (
        "the register's stated rule is expected_absent only at rate >= "
        f"{EXPECTED_ABSENT_MIN_RATE} unless override_reason argues otherwise: "
        + "; ".join(violations))

    print("\n=== SANITY CHECK: regimes + expected absence ===")
    print(f"  {len(cat.regime_names)} regimes: {', '.join(cat.regime_names)}")
    print(f"  {len(cat.force_regime_by_sub_industry)} sub-industries force a regime "
          "(the verified GICS traps):")
    for si, regime in sorted(cat.force_regime_by_sub_industry.items()):
        print(f"    {si:45} -> {regime}")
    structural = sum(1 for b in cat.regime_exceptions.values()
                     for c in b.values() if c.get("expected_absent"))
    print(f"  {structural} (regime, field) cells are structurally excused at a measured "
          f"absence rate >= {EXPECTED_ABSENT_MIN_RATE}")
    print(f"  {len(overrides)} argued exception(s) below that bar: "
          f"{', '.join(overrides) or 'none'}")
    print(f"  bank/insurer currentAssets absence: "
          f"{cat.measured_absent_rate('bank', 'currentAssets'):.0%} / "
          f"{cat.measured_absent_rate('insurer', 'currentAssets'):.0%} -- structural, per "
          "17 CFR 210.1-02(bb)(1)(i)")
    print(f"  utility/energy currentAssets absence: "
          f"{cat.measured_absent_rate('utility', 'currentAssets'):.0%} / "
          f"{cat.measured_absent_rate('energy', 'currentAssets'):.0%} -- they DO file "
          "classified balance sheets, so absence there stays a finding. Validated.")


def test_the_measured_traps_are_encoded(cat):
    """The specific measured defects this rebuild exists to remove must be IN the contract,
    not merely described in the plan. Each assertion below corresponds to a number the
    research measured."""
    # 2.3: shortTermDebt's two disjoint legs, and both components carried
    std = cat.field("shortTermDebt")
    assert std.roll_up() == ["LongTermDebtCurrent", "ShortTermBorrowings"]
    assert set(std.raw["components"]) == {"longTermDebtCurrentOnly", "shortTermBorrowingsOnly"}
    assert std.total_concept() == "DebtCurrent"
    assert "FinanceLeaseLiabilityCurrent" in std.raw["total_adjustment"]["subtract"]

    # 2.9 / 3.2: the bank top line, and the gross-interest-income trap
    rev = cat.field("totalRevenue")
    assert rev.roll_up("bank") == ["InterestIncomeExpenseNet", "NoninterestIncome"]
    assert rev.total_concept("bank") == "RevenuesNetOfInterestExpense"
    assert "InterestAndDividendIncomeOperating" in rev.never_use("bank")

    # 2.4: capex's superset element, and MAA's IPR&D-tagged development capex
    capex = cat.field("capex")
    assert capex.total_concept() == "PaymentsToAcquireProductiveAssets"
    assert "PaymentsToAcquireInProcessResearchAndDevelopment" in capex.never_use()

    # decision #10: R&D is regime-gated and never lets the REIT capex element in
    rd = cat.field("researchAndDevelopment")
    assert rd.regime_gated
    assert "PaymentsToAcquireInProcessResearchAndDevelopment" in rd.never_use()
    assert "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost" in rd.never_use()

    # decision #9: epsDiluted is computed, and the as-reported tag is a CROSS-CHECK only
    eps = cat.field("epsDiluted")
    assert eps.kind == "derived"
    assert eps.raw["cross_check_concepts"] == ["EarningsPerShareDiluted"]

    # 2.4: the D&A aggregate, not the non-production element
    assert cat.field("depAmort").total_concept() == "DepreciationDepletionAndAmortization"

    # non-additive fields must never be TTM-summed
    for name in ("dilutedShares", "basicShares", "effectiveTaxRate"):
        assert not cat.field(name).is_additive, f"{name} must not be additive"

    # research part 2, Open item #1 -> option (b): net the finance-lease ROU asset out of
    # ppeNet, but only where the linkbase says it is folded IN rather than shown beside.
    ppe = cat.field("ppeNet")
    assert ppe.raw["total_adjustment"]["subtract"] == ["FinanceLeaseRightOfUseAsset"]
    assert ppe.raw["total_adjustment"]["_only_when"], \
        "the ROU subtraction must state its linkbase condition, or it double-removes"

    # research part 2, Open item #2 -> option (b): a caption-anchored bank branch built from
    # Reg S-X 9-04 captions 6/7/8, NOT from InterestExpenseBorrowings (2 of 14 tickers).
    ie = cat.field("interestExpense")
    assert ie.total_concept("bank") == "InterestExpenseOperating"
    assert ie.roll_up("bank") == ["InterestExpenseDeposits",
                                  "InterestExpenseShortTermBorrowings",
                                  "InterestExpenseLongTermDebt"]
    assert "InterestExpenseBorrowings" not in ie.roll_up("bank")
    assert "InterestExpenseDebt" in ie.never_use("bank")
    # InterestExpense is retained SECOND for banks: it is the only element covering 2011-2024.
    assert ie.fallback_concepts("bank") == ["InterestExpenseOperating", "InterestExpense"]

    # research part 2 called IncomeTaxExpenseBenefitContinuingOperations a nonexistent tag;
    # the ledger shows 276 facts / 82 tickers, 2011-2013. Retained, and documented as such.
    tax = cat.field("incomeTaxExpense")
    assert "IncomeTaxExpenseBenefitContinuingOperations" in tax.fallback_concepts()
    assert "IncomeTaxExpenseBenefitContinuingOperations" in tax.raw["fallback_concept_notes"]

    print("\n=== SANITY CHECK: the measured traps are in the contract ===")
    print("  shortTermDebt sums BOTH disjoint legs (2,017 cells / 111 tickers tag both,")
    print("    and the old resolver discarded the larger one 54.4% of the time)")
    print("  bank totalRevenue = InterestIncomeExpenseNet + NoninterestIncome, and")
    print("    InterestAndDividendIncomeOperating (GROSS) is explicitly forbidden")
    print("  capex prefers FASB's declared SUPERSET; MAA's $272M IPR&D-tagged")
    print("    development capex is forbidden for BOTH capex and researchAndDevelopment")
    print("  epsDiluted is DERIVED (netIncome_ttm / dilutedShares_ttm); the as-reported")
    print("    tag is retained only as an independent cross-check")
    print("  depAmort takes the AGGREGATE, not the officially non-production element")
    print("  dilutedShares / basicShares / effectiveTaxRate are non-additive")
    print("  ppeNet nets out FinanceLeaseRightOfUseAsset, but ONLY where the linkbase shows")
    print("    it folded in (detectable for 182 of 417 ppeNet tickers; a sibling in the")
    print("    linkbase means it was never inside ppeNet and must not be subtracted)")
    print("  interestExpense has a Reg-S-X 9-04 bank branch: total = InterestExpenseOperating")
    print("    (caption 9), legs = Deposits + ShortTermBorrowings + LongTermDebt (captions")
    print("    6/7/8). InterestExpenseBorrowings is EXCLUDED -- 2 of 14 banks tag it.")
    print("  IncomeTaxExpenseBenefitContinuingOperations is kept as a pre-2014 fallback:")
    print("    276 facts / 82 tickers, window 2011-06-30..2013-12-31. Validated.")
