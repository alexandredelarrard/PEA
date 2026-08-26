"""
Cluster `2603621e89ab` -- ORCL `totalRevenue`, 47 findings across 7 agreeing checks.

Oracle's fiscal 2020, 2021 and 2022 10-Ks tag the **full-year** `us-gaap:Revenues` against a
**91-day fourth-quarter context**. Fiscal 2022 (**0001564590-22-023675**) is the clearest:

    us-gaap:Revenues                                     2022-03-01..2022-05-31   $42,440M
    us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax
                                                         2021-06-01..2022-05-31   $42,440M

The same number, the same end date, one quarter of the window -- so the year is not in doubt
and no inference is needed to date it. Oracle's true Q4 FY2022 revenue is $42,440M -
($9,728M + $10,360M + $10,513M) = **$11,840M**, which is exactly what `holdout_q4` derived.
The SEC's own frame index bought the bad context too, filing that $42.4bn annual figure under
`frame=CY2022Q2`.

Ground truth read off the filings via `companyconcept` (which can prove a concept PRESENT,
never absent -- every claim here is a presence claim):
  * `us-gaap:Revenues` in 0001564590-22-023675 has THREE undimensioned facts, all 91-day:
    2020-03-01..2020-05-31 $39,068M, 2021-03-01..2021-05-31 $40,479M and
    2022-03-01..2022-05-31 $42,440M -- one per fiscal year, never a sibling quarter, and no
    annual window anywhere in the filing.
  * the correct annual figures ARE in the same filing, under
    `RevenueFromContractWithCustomerExcludingAssessedTax` on 2019-06-01..2020-05-31 (365d),
    2020-06-01..2021-05-31 (364d) and 2021-06-01..2022-05-31 (364d).
  * 9 such rows across the three filings, fiscal 2018-2022.

**Why `_drop_note_only_quarter` claimed these rows and never touched them.** Its measurement
read "11 of the 19 are the sole source of their period (BA 2, ORCL 9)", and the ORCL half was
never true. `_lone_quarters` dated a quarter by a covering annual window of THE SAME FIELD,
via `_annual_windows(periods)`; `Revenues` has none in those filings, so the function hit
`if not windows: return {}` and judged nothing at all. The rows survived the fix that claimed
them. Each filing does carry 3 annual windows -- 36 annual rows across other fields -- and
they cover the bad quarters exactly, so passing the FILING's calendar in as a fallback dates
them off the filer's own contexts.

`periods._drop_annual_masquerading_as_quarter` (D1b) already refused these rows downstream,
which is why `fundamentals_history_sec` never showed $42,440M as a quarter. But D1b runs on the
way to HISTORY, and `fundamentals_facts` -- the substrate all 47 findings read -- kept
asserting the bad quarter. Refusing at the facts layer is what closes the cluster.

**Where the refusal is RECORDED.** Emptying `Revenues` for those three filings leaves the
field resolved with no period, which is `no_usable_period`'s shape but not its meaning --
that code says `_materialise` found none, and here we found three and refused them. So the
stub carries `ambiguous_duration` instead, the code D1b used to write for this exact defect,
which keeps the diagnostic when the refusal moves layers. `test_periods_q4.py::test_orcls_
fiscal_2020_fourth_quarter_is_refused_and_no_other_year_is` pins that end-to-end against the
real filings (3 stubs, one per 10-K, all value-less); `test_the_filings_own_calendar_refuses
_all_three` below pins its precondition, that the guard empties the field outright.

Every test below is synthetic known-truth built from the numbers above (docs/testing.md:
parsing math gets fixtures); the real-filing evidence is the measurement in the function's
docstring. The NEGATIVE tests carry the weight:
`test_fallback_is_scoped_to_fields_with_no_annual_of_their_own` pins the scoping decision that
keeps the blast radius at 9 rows rather than 16, and
`test_asc270_table_survives_a_filing_wide_calendar` pins the sibling rule, which is the only
thing standing between this fallback and four filers' quarterly `grossProfit`.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import (
    _drop_note_only_quarter, _filing_annual_windows, _period_frame, _values_by_period)
from src.data_extract.utils.fundamentals.periods import ANNUAL, QUARTERLY

REVENUES = "us-gaap:Revenues"
ASC606 = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"

#: `us-gaap:Revenues` exactly as ORCL's fiscal 2022 10-K tags it: three full YEARS, each in a
#: 91-day fourth-quarter context, and not one annual window.
_ORCL_REVENUES: list[tuple[str, str, str, float]] = [
    (REVENUES, "2020-03-01", "2020-05-31", 39_068_000_000.0),
    (REVENUES, "2021-03-01", "2021-05-31", 40_479_000_000.0),
    (REVENUES, "2022-03-01", "2022-05-31", 42_440_000_000.0),
]

#: The SAME three years, correctly windowed, from the SAME filing under the ASC 606 element.
#: This is where the filing's fiscal calendar comes from -- and it is the proof that the
#: 91-day contexts hold years: $42,440M appears in both lists.
_ORCL_ANNUAL: list[tuple[str, str, str, float]] = [
    (ASC606, "2019-06-01", "2020-05-31", 39_068_000_000.0),
    (ASC606, "2020-06-01", "2021-05-31", 40_479_000_000.0),
    (ASC606, "2021-06-01", "2022-05-31", 42_440_000_000.0),
]


def _periods(facts: list[tuple[str, str, str, float]], concept: str) -> dict[tuple, dict]:
    """`{period key: fact}` for `concept`, through the production framing.

    Routed through `_period_frame` + `_values_by_period` rather than hand-built: the guard
    branches on `duration_type`, so it must come from the real day-count bands or a fixture
    could classify a window the pipeline would not. ORCL is the reason that matters -- 91 and
    364 days are what separate the bad context from the good one.
    """
    frame = pd.DataFrame([{"concept": c, "numeric_value": v, "period_type": "duration",
                           "period_start": start, "period_end": end,
                           "fiscal_year": pd.Timestamp(end).year,
                           "fiscal_period": "Q4", "unit_ref": "usd", "decimals": "-6"}
                          for c, start, end, v in facts])
    return _values_by_period(_period_frame(frame), concept)


def _quarter_ends(periods: dict[tuple, dict]) -> set[str]:
    return {str(pd.Timestamp(p["period_end"]).date()) for p in periods.values()
            if p["duration_type"] == QUARTERLY}


def _orcl_filing() -> dict[str, dict[tuple, dict]]:
    """The filing as `rows_from_xbrl` holds it: `{field: periods}`, two fields.

    `totalRevenue` resolved to `Revenues`, which is the defect's precondition: resolution is
    period-agnostic by design, so a concept present only in broken contexts still wins.
    """
    return {"totalRevenue": _periods(_ORCL_REVENUES, REVENUES),
            "netIncome": _periods(_ORCL_ANNUAL, ASC606)}


# --------------------------------------------------------------------------- #
# The defect                                                                   #
# --------------------------------------------------------------------------- #
def test_orcl_full_years_in_quarterly_contexts_survived_the_field_local_rule():
    """The bug, pinned: with only the field's own windows there is nothing to judge against.

    This was never a threshold set too loose -- the guard returned before reading a single
    quarter, because `Revenues` has no annual window in the filing. Pinned so the fallback
    cannot later be deleted as apparently-dead code.
    """
    before = _periods(_ORCL_REVENUES, REVENUES)

    assert _quarter_ends(before) == {"2020-05-31", "2021-05-31", "2022-05-31"}
    assert not [p for p in before.values() if p["duration_type"] == ANNUAL], (
        "the premise: ORCL publishes no annual-window `Revenues` in these filings")
    assert _drop_note_only_quarter(before, form="10-K") == before
    print("field-local rule: all 3 mislabelled years kept -- 0 of 9 rows caught, which is "
          "the defect the docstring's 'ORCL 9' claimed to have fixed")


@pytest.mark.parametrize("form", ["10-K", "10-K/A"])
def test_the_filings_own_calendar_refuses_all_three(form):
    """With the FILING's annual windows, each mislabelled year is a lone quarter and goes.

    Parametrised over `10-K/A` because an amendment restates the same statements and so
    carries the same broken contexts.
    """
    values = _orcl_filing()
    windows = _filing_annual_windows(values)

    after = _drop_note_only_quarter(values["totalRevenue"], form=form,
                                    filing_windows=windows)

    assert _quarter_ends(after) == set(), "a full year is not a fourth quarter"
    assert after == {}, "nothing else was in `Revenues` to keep"
    print(f"{form}: 3 mislabelled years refused ($39,068M/$40,479M/$42,440M); true Q4 FY2022 "
          f"is $11,840M by FY-YTD9, not $42,440M")


def test_the_annual_windows_of_other_fields_are_left_alone():
    """The fallback reads other fields' calendars; it must never edit them.

    `netIncome` carries the three annual windows this whole fix leans on, and the guard runs
    on every field in the filing -- so a rule that touched them would cost the filing the
    numbers it came for.
    """
    values = _orcl_filing()
    windows = _filing_annual_windows(values)

    after = _drop_note_only_quarter(values["netIncome"], form="10-K",
                                    filing_windows=windows)

    assert after == values["netIncome"]
    assert sorted(p["value"] for p in after.values()) == [
        39_068_000_000.0, 40_479_000_000.0, 42_440_000_000.0]
    print("the 3 annual windows that date the refusal are themselves untouched")


def test_filing_annual_windows_dedupes_by_span():
    """One filing states one fiscal calendar, however many fields tag it.

    Deduplicated on `(start, end)` and not on the period key, which carries the field: a full
    catalogue would otherwise repeat each year ~50 times and lengthen every
    `_covering_annual` scan for no extra evidence.
    """
    values = _orcl_filing()
    values["operatingIncome"] = _periods(_ORCL_ANNUAL, ASC606)   # the same three years again

    windows = _filing_annual_windows(values)

    assert [(str(lo.date()), str(hi.date())) for _, lo, hi in windows] == [
        ("2019-06-01", "2020-05-31"),
        ("2020-06-01", "2021-05-31"),
        ("2021-06-01", "2022-05-31")]
    print(f"2 fields x 3 identical years -> {len(windows)} windows")


# --------------------------------------------------------------------------- #
# The regressions this fallback could have caused                               #
# --------------------------------------------------------------------------- #
def test_asc270_table_survives_a_filing_wide_calendar():
    """The sibling rule is the whole safety margin, and it now carries more weight.

    Before the fallback, a field with no annual window of its own was never judged; now it is
    judged against the filing's calendar, which is exactly the population DE, CAT, EQIX, TMO
    and AFL sit in -- 388 legitimate ASC 270 rows between them. Four siblings in a fiscal
    year is a SERIES and must survive.
    """
    table = [(REVENUES, "2021-09-01", "2021-11-30", 10_360_000_000.0),
             (REVENUES, "2021-12-01", "2022-02-28", 10_513_000_000.0),
             (REVENUES, "2022-03-01", "2022-05-31", 11_840_000_000.0),
             (REVENUES, "2021-06-01", "2021-08-31", 9_728_000_000.0)]
    before = _periods(table, REVENUES)
    values = {"totalRevenue": before, "netIncome": _periods(_ORCL_ANNUAL, ASC606)}

    after = _drop_note_only_quarter(before, form="10-K",
                                    filing_windows=_filing_annual_windows(values))

    assert after == before, "four siblings in FY2022 is a series, not a sentence"
    assert _quarter_ends(after) == {"2021-08-31", "2021-11-30", "2022-02-28", "2022-05-31"}
    print("ASC 270: all 4 real FY2022 quarters kept against the filing-wide calendar, "
          "including the true Q4 of $11,840M")


def test_fallback_is_scoped_to_fields_with_no_annual_of_their_own():
    """The scoping decision, and it is a blast-radius decision rather than a taste one.

    Consulting the filing's calendar only where the field has NO window of its own drops 9
    rows table-wide, all ORCL. Unioning it in unconditionally drops 16 across 7 (ticker,
    field) pairs -- 7 further rows on DTE, EQIX, META and VLO that share the prose-aside
    shape but have not been read at filing level. So a field that declares its own calendar
    is judged on that alone, even where the filing's is wider.
    """
    own = [(REVENUES, "2021-06-01", "2022-05-31", 42_440_000_000.0),   # its own FY2022
           (REVENUES, "2022-03-01", "2022-05-31", 11_840_000_000.0),   # a real lone Q4
           (REVENUES, "2020-03-01", "2020-05-31", 39_068_000_000.0)]   # outside its calendar
    before = _periods(own, REVENUES)
    wider = {"totalRevenue": before, "netIncome": _periods(_ORCL_ANNUAL, ASC606)}

    after = _drop_note_only_quarter(before, form="10-K",
                                    filing_windows=_filing_annual_windows(wider))

    assert "2020-05-31" in _quarter_ends(after), (
        "FY2020 is outside this field's own calendar, so the wider one must not reach it")
    assert "2022-05-31" not in _quarter_ends(after), (
        "the field's own FY2022 window still judges its own lone quarter")
    print("scoping: the field's own calendar judges FY2022 and the filing's does NOT reach "
          "FY2020 -- 9 rows, not 16")


def test_a_10q_is_still_gated_out():
    """The form gate survives the new argument.

    A 10-Q's face statement carries one quarterly window per fiscal year, so every quarter in
    every 10-Q is "lone" -- and a 10-Q now always has a filing-wide calendar to be judged
    against, which makes the gate load-bearing rather than merely tidy.
    """
    values = _orcl_filing()
    windows = _filing_annual_windows(values)

    assert _drop_note_only_quarter(values["totalRevenue"], form="10-Q",
                                   filing_windows=windows) == values["totalRevenue"]
    print("form gate: the 3 ORCL windows are refused in a 10-K and kept in a 10-Q")


def test_no_annual_anywhere_in_the_filing_is_still_not_judged():
    """Silence is still not evidence. With no calendar at all, the guard declines."""
    before = _periods([(REVENUES, "2022-03-01", "2022-05-31", 42_440_000_000.0)], REVENUES)

    assert _drop_note_only_quarter(before, form="10-K", filing_windows=[]) == before
    assert _drop_note_only_quarter(before, form="10-K") == before
    print("no calendar anywhere: the quarter is kept, not guessed at")
