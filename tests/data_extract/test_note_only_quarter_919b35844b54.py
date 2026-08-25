"""
Cluster `919b35844b54` -- BA `incomeTaxExpense`, 40 findings across 10 agreeing checks.

Boeing's fiscal 2011 10-K (**0001193125-12-048565**) publishes, in Note 23 "Quarterly
Financial Data (Unaudited)", a table of all four quarters of 2011 and 2010 -- revenues, costs,
earnings from operations, net earnings, EPS, dividends, share prices -- and **no income tax
line at all**. Underneath it sits the sentence:

    "During the fourth quarters of 2011 and 2010, we recorded tax benefits of $397 and $371
     as a result of settling the 2004-2006 and 1998-2003 federal audits."

The filer tagged that SENTENCE with `us-gaap:IncomeTaxExpenseBenefit` against Q4 contexts, and
tagged the same magnitudes correctly signed as `us-gaap:TaxAdjustmentsSettlementsAndUnusual
Provisions` (-$397M, -$371M) in the very same contexts. `_values_by_period` selects facts by
concept-name equality with no context or role filter, so both landed in `fundamentals_facts`
as those quarters' income tax expense.

Ground truth read off the filed statements, not `companyfacts`:
  * 0001193125-12-048565 (FY2011 10-K) -- `IncomeTaxExpenseBenefit` has exactly FIVE
    undimensioned facts: FY2009 $396M, FY2010 $1,196M, FY2011 $1,382M, and Q4-2010 $371M,
    Q4-2011 $397M with no Q1/Q2/Q3 sibling in either year. `us-gaap:Revenues` in the SAME
    filing has all four quarters of both years, off the table.
  * 0001193125-11-281613 / 0000012927-12-000024 (Q3 10-Qs) -- ytd9 tax $1,359M (2010) and
    $1,325M (2011), so the true fourth quarters are $1,196M - $1,359M = **-$163M** and
    $1,382M - $1,325M = **+$57M**. Both signs were wrong: a settlement BENEFIT was stored as
    an expense. `holdout_q4` derived exactly those two numbers.

The fix is `fetch_fundamentals_sec._drop_note_only_quarter`. It keys on the note's SHAPE --
the ASC 270 / Item 302 quarterly table is a SERIES, an ASC 270-10-50-2 fourth-quarter
adjustment is a lone SENTENCE -- so it needs no concept list, no role-name matching and no
second request. Every test below is synthetic known-truth (docs/testing.md: parsing math gets
fixtures) built from the numbers above; the real-filing evidence is the measurement recorded
in the function's docstring.

The two tests that matter most are the NEGATIVE ones. `test_asc270_schedule_keeps_all_four_
quarters` pins the case the fix must not break -- BA's own Q4 revenue of $16,550M comes off
the same note in the same filing -- and `test_a_10q_keeps_its_lone_quarter` pins the form
gate, without which every quarter in every 10-Q is "lone" and the whole quarterly grain dies.
"""
from __future__ import annotations

import json

import pandas as pd

from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import (
    _adjustment_json, _drop_note_only_quarter, _period_frame, _values_by_period)
from src.data_extract.utils.fundamentals.xbrl_linkbase import LINKBASE_TOTAL, Resolution

TAX = "us-gaap:IncomeTaxExpenseBenefit"
REVENUES = "us-gaap:Revenues"

#: `IncomeTaxExpenseBenefit`, as BA's FY2011 10-K actually tags it: three annual windows and
#: one lone fourth quarter per year, off the audit-settlement sentence.
_BA_TAX: list[tuple[str, str, str, float]] = [
    (TAX, "2009-01-01", "2009-12-31", 396_000_000.0),
    (TAX, "2010-01-01", "2010-12-31", 1_196_000_000.0),
    (TAX, "2010-10-01", "2010-12-31", 371_000_000.0),
    (TAX, "2011-01-01", "2011-12-31", 1_382_000_000.0),
    (TAX, "2011-10-01", "2011-12-31", 397_000_000.0),
]

#: `Revenues`, from the SAME filing and the SAME note -- but off the table, so all four
#: quarters of 2010 are present. This is the shape the fix must leave alone.
_BA_REVENUE: list[tuple[str, str, str, float]] = [
    (REVENUES, "2010-01-01", "2010-12-31", 64_306_000_000.0),
    (REVENUES, "2010-01-01", "2010-03-31", 15_216_000_000.0),
    (REVENUES, "2010-04-01", "2010-06-30", 15_573_000_000.0),
    (REVENUES, "2010-07-01", "2010-09-30", 16_967_000_000.0),
    (REVENUES, "2010-10-01", "2010-12-31", 16_550_000_000.0),
]


def _periods(facts: list[tuple[str, str, str, float]], concept: str) -> dict[tuple, dict]:
    """`{period key: fact}` for `concept`, through the production framing.

    Deliberately routed through `_period_frame` + `_values_by_period` rather than
    hand-building the dict: `duration_type` is what the guard branches on, and it must come
    from the real day-count bands so a fixture cannot classify a window the pipeline would
    not.
    """
    frame = pd.DataFrame([{"concept": c, "numeric_value": v, "period_type": "duration",
                           "period_start": start, "period_end": end,
                           "fiscal_year": pd.Timestamp(end).year,
                           "fiscal_period": "Q4", "unit_ref": "usd", "decimals": "-6"}
                          for c, start, end, v in facts])
    return _values_by_period(_period_frame(frame), concept)


def _quarter_ends(periods: dict[tuple, dict]) -> set[str]:
    return {str(pd.Timestamp(p["period_end"]).date())
            for p in periods.values() if p["duration_type"] == "quarterly"}


# --------------------------------------------------------------------------- #
# The defect                                                                   #
# --------------------------------------------------------------------------- #
def test_ba_shape_a_lone_quarter_in_a_10k_is_refused():
    """BA: a fourth quarter with no sibling quarter in its own fiscal year is a sentence.

    Both years are refused, and every annual window survives untouched -- the guard must
    never cost the filing the numbers it actually came for.
    """
    before = _periods(_BA_TAX, TAX)
    assert _quarter_ends(before) == {"2010-12-31", "2011-12-31"}

    after = _drop_note_only_quarter(before, form="10-K")

    assert _quarter_ends(after) == set(), "the audit-settlement sentence is not a quarter"
    annual = sorted(p["value"] for p in after.values() if p["duration_type"] == "annual")
    assert annual == [396_000_000.0, 1_196_000_000.0, 1_382_000_000.0]
    print(f"BA shape: {len(before)} periods -> {len(after)}; the $371M/$397M Q4 tax rows are "
          f"gone (true Q4s are -$163M and +$57M by FY-YTD9), all 3 annual windows kept")


def test_the_refusal_is_recorded_on_the_covering_annual():
    """A refusal nobody can count is a silent delete.

    The marker lands on the ANNUAL window that contains the refused quarter -- the period the
    note belongs to -- and rides the `adjustment` JSON, so
    `adjustment::jsonb ? 'note_quarter_rejected'` finds every row the guard acted on with no
    schema change.
    """
    after = _drop_note_only_quarter(_periods(_BA_TAX, TAX), form="10-K")
    hosts = {str(pd.Timestamp(p["period_end"]).date()): p.get("note_quarter_rejected")
             for p in after.values()}

    assert hosts["2010-12-31"] == [{"period_end": "2010-12-31", "value": 371_000_000.0}]
    assert hosts["2011-12-31"] == [{"period_end": "2011-12-31", "value": 397_000_000.0}]
    assert hosts["2009-12-31"] is None, "a year with no refusal carries no marker"

    host = next(p for p in after.values()
                if str(pd.Timestamp(p["period_end"]).date()) == "2011-12-31")
    blob = json.loads(_adjustment_json(
        Resolution(field="incomeTaxExpense", method=LINKBASE_TOTAL, concept=TAX), host))
    assert blob == {"note_quarter_rejected": [{"period_end": "2011-12-31",
                                              "value": 397_000_000.0}]}
    print(f"refusal recorded: FY2011's adjustment blob is {json.dumps(blob)}")


# --------------------------------------------------------------------------- #
# The regressions the defect's obvious fix would have caused                    #
# --------------------------------------------------------------------------- #
def test_asc270_schedule_keeps_all_four_quarters():
    """The ASC 270 TABLE is a legitimate source of quarterly values, in the same filing.

    A blanket "no quarters from a 10-K" rule would have cost BA its published Q4 revenue of
    $16,550M -- a real number, off the same note, three lines above the sentence that caused
    this cluster.
    """
    before = _periods(_BA_REVENUE, REVENUES)
    after = _drop_note_only_quarter(before, form="10-K")

    assert after == before, "four siblings is a series, not a sentence"
    assert _quarter_ends(after) == {"2010-03-31", "2010-06-30", "2010-09-30", "2010-12-31"}
    print(f"ASC 270 schedule: all {len(_quarter_ends(after))} quarters of 2010 kept, "
          f"Q4 revenue still $16,550M")


def test_a_10q_keeps_its_lone_quarter():
    """The form gate. A 10-Q's face statement carries ONE quarterly window per fiscal year.

    So every quarter in every 10-Q is "lone" by this test, and an ungated rule would delete
    the entire quarterly grain of the table rather than two rows of it.
    """
    facts = [(TAX, "2011-01-01", "2011-12-31", 1_382_000_000.0),
             (TAX, "2011-07-01", "2011-09-30", 548_000_000.0)]
    before = _periods(facts, TAX)

    assert _drop_note_only_quarter(before, form="10-Q") == before
    assert _quarter_ends(_drop_note_only_quarter(before, form="10-K")) == set()
    print("form gate: BA's real Q3 2011 $548M survives a 10-Q, is refused only in a 10-K")


def test_a_quarter_with_no_covering_annual_is_not_judged():
    """Silence is not evidence -- the rule `xbrl_linkbase.is_note_only` and D1 both apply.

    With no annual window in the filing there is no fiscal year to count siblings within, so
    the guard declines to judge rather than guessing.
    """
    before = _periods([(TAX, "2011-07-01", "2011-09-30", 548_000_000.0)], TAX)

    assert _drop_note_only_quarter(before, form="10-K") == before
    print("no covering annual: the quarter is kept, not guessed at")


def test_amended_annual_forms_are_gated_too():
    """A 10-K/A restates the same notes, so it carries the same defect."""
    before = _periods(_BA_TAX, TAX)

    assert _quarter_ends(_drop_note_only_quarter(before, form="10-K/A")) == set()
    print("10-K/A is gated with 10-K")
