"""
test_substrate_contract.py  (tests/validate/fundamentals/)
--------------------------------------------------------------------------------------------
WHICH TABLE EACH CHECK READS, and whether its findings can be acted on.

This file exists because of a measurement. On the 2026-08-24 calibration run, Tier 1 read
`fundamentals_history` for 14 of its 19 checks and produced 1,437 findings -- **0 of which
carried an `edgar_url`**, against 77.8% on Tier 2 and 100% on Tier 3. `Finding.edgar_url` is
built from `(cik, accession_number)` and `fundamentals_history` carries neither, so an agent
handed a Tier-1 finding could not open the filing that caused it. That is the first move the
triage loop requires, so the entire tier was unactionable however well it was ranked.

Eight checks moved to `fundamentals_facts`. Six stayed, and the split is the contract pinned
here:

  * a check that asks about the TABLE reads `history` -- the 69-column ordered contract, a
    null CELL, the reason-code vocabulary, the no-leakage snapshot grain. None of those exist
    in `facts`, so porting them would DELETE them rather than relocate them. They are the
    tripwires for a bug in `build_history`, which is the one defect class genuinely history's
    own -- ETN's 2012 row, `totalLiabilities` of -$8,237,223,652 against `totalAssets` of
    $4,776,348, is the specimen;
  * a check that asks about a NUMBER reads `facts`, where every row carries its accession.

Two properties are pinned, and the second is the one that would rot silently: a future check
added to Tier 1 on the history substrate would look perfectly reasonable in review and would
quietly reintroduce an unactionable finding.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.validate.fundamentals.checks import CHECK_REGISTRY
from tests.validate.fundamentals.conftest import TICKER

#: The six checks that legitimately read `fundamentals_history`. Adding to this set is a
#: DESIGN DECISION -- it means "this check tests a property `fundamentals_facts` cannot
#: express" -- so it must be a deliberate edit with a failing test to update.
CONTRACT_CHECKS = {"grain", "column_contract", "code_vocabulary", "unexplained_null",
                   "pit_leak", "coverage_universe"}

#: Checks whose subject is our CONFIGURATION rather than a filing, so no accession exists to
#: name. They are the only findings allowed to reach a reviewer without a URL.
CONFIG_DIAGNOSTICS = {"catalogue_exclusion_cost", "catalogue_override_coverage",
                      "amendment_ledger", "same_day_collapse"}


def test_only_the_six_contract_checks_read_history() -> None:
    """The split, pinned. A new history-substrate check must be an argued exception."""
    on_history = {name for name, spec in CHECK_REGISTRY.items()
                  if spec.substrate == "history"}

    print(f"\nhistory: {sorted(on_history)}")
    print(f"  facts:   {len(CHECK_REGISTRY) - len(on_history)} checks across all three tiers")
    print("  SANITY: these six test the 69-column contract, a null CELL, the reason-code "
          "vocabulary and the no-leakage grain. `facts` expresses none of them.")
    assert on_history == CONTRACT_CHECKS


def test_every_contract_check_is_a_zero_ceiling_tripwire() -> None:
    """They are not expected to fire at all -- that is what keeps them on history.

    A history check with a non-zero ceiling would be a VALUE check wearing a contract check's
    clothes, and it belongs on facts with the rest.
    """
    ceilings = {name: CHECK_REGISTRY[name].expected_fire_rate_ceiling
                for name in sorted(CONTRACT_CHECKS)}

    print("\n" + "\n".join(f"  {n:<20} ceiling={c:.0%}" for n, c in ceilings.items()))
    print("  SANITY: all zero. They fired 0 findings on the live 54-ticker roster, which is "
          "what a tripwire is supposed to do.")
    assert all(c == 0.0 for c in ceilings.values())


def test_no_tier_1_value_check_reads_history_any_more() -> None:
    """The regression this whole file guards: an unactionable finding is easy to reintroduce."""
    offenders = sorted(name for name, spec in CHECK_REGISTRY.items()
                       if spec.tier == 1 and spec.substrate == "history"
                       and name not in CONTRACT_CHECKS)

    print(f"\nTier-1 value/coverage checks still on history: {offenders or 'none'}")
    print("  SANITY: a finding without (cik, accession_number) has no edgar_url, and a "
          "reviewer cannot open the filing. 0 of 1,437 had one before the move.")
    assert not offenders


@pytest.fixture
def dirty_facts(clean_facts) -> pd.DataFrame:
    """The clean fixture with four INDEPENDENT defects planted, one per ported check.

    The clean base is silent on purpose -- that is what makes every planted-violation test
    meaningful -- so asserting "these findings carry a URL" against it asserts nothing at all.
    Each plant below trips exactly one of the ported checks:

      `cross_identity`     the balance sheet stops footing on the last period
      `coverage_field`     the base filer stops resolving `capex` while its six peers keep
                           doing so, which is the UNIVERSAL verdict
      `filing_lag`         one filing lands 300 days after its own `period_of_report`
      `impossible_value`   a negative top line
    """
    facts = clean_facts.copy()

    last = facts["period_end"].max()
    bent = ((facts["ticker"] == TICKER) & (facts["field"] == "totalAssets")
            & (facts["period_end"] == last))
    facts.loc[bent, "value"] = facts.loc[bent, "value"] * 1.5

    facts = facts[~((facts["ticker"] == TICKER) & (facts["field"] == "capex"))]

    first_accession = facts.loc[facts["ticker"] == TICKER, "accession_number"].iloc[0]
    late = facts["accession_number"] == first_accession
    facts.loc[late, "filing_date"] = (facts.loc[late, "period_of_report"]
                                      + pd.Timedelta(days=300))

    negative = ((facts["ticker"] == TICKER) & (facts["field"] == "totalRevenue")
                & (facts["period_end"] == last))
    facts.loc[negative, "value"] = -abs(facts.loc[negative, "value"])
    return facts.reset_index(drop=True)


@pytest.mark.parametrize("name", sorted(
    n for n, s in CHECK_REGISTRY.items()
    if s.tier == 1 and s.substrate == "facts" and n not in CONFIG_DIAGNOSTICS))
def test_a_tier_1_facts_check_names_the_filing(name, catalogue, dirty_facts) -> None:
    """Every finding a facts-grain Tier-1 check produces carries an accession, or none do.

    Parametrised over the REGISTRY rather than a hand-written list, so a check added to Tier 1
    tomorrow is covered without anyone remembering to add it here -- and it runs against a
    fixture with real defects in it, so the assertions bite.

    A check with no plant of its own still skips rather than passing vacuously, and says which
    it was. The live measurement is in the module docstring: 876 of 886 filing-implicating
    findings carried a URL on the 54-ticker roster, the 10 exceptions all being ticker-grain
    diagnostics with no accession to name.
    """
    from tests.validate.fundamentals.conftest import build_substrates

    substrates = build_substrates(catalogue, dirty_facts.copy())
    found = CHECK_REGISTRY[name].fn(substrates)
    if not found:
        pytest.skip(f"{name} has no plant in `dirty_facts` -- nothing to assert here")

    rows = [f.as_row(pd.Timestamp("2026-08-25")) for f in found]
    without = [r for r in rows if not r["edgar_url"]]

    print(f"\n{name}: {len(rows) - len(without)}/{len(rows)} findings carry an edgar_url")
    if without:
        print(f"  first offender: {without[0]['ticker']} {without[0]['field']}")
    print("  SANITY: a Tier-1 finding a reviewer cannot open is a finding nobody can settle.")
    assert not without
