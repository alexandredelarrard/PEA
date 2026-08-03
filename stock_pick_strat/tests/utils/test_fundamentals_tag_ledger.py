"""
Tests for src/utils/fundamentals_tag_ledger.py: collapsing `fundamentals_facts`
into contiguous `source_tag` eras, and scoring the switch between eras so a
benign US-GAAP taxonomy migration is separated from a real measure splice.

Pure-synthetic, no network / no DB.
"""
from __future__ import annotations

import pandas as pd

from src.constants.constants import TAG_SWITCH_LEVEL_BREAK_RATIO
from src.utils.fundamentals_tag_ledger import (
    TAG_SWITCH_LEVEL_BREAK, build_tag_ledger, detect_tag_switch_breaks,
)


def _row(period_end, value, source_tag="us-gaap:X", *, filed=None, derived=0.0,
        ticker="ZZZ", field="shortTermDebt", duration_type=None):
    period_end = pd.Timestamp(period_end)
    return {"ticker": ticker, "field": field, "period_end": period_end,
           "value": value, "source_tag": source_tag, "derived": derived,
           "duration_type": duration_type,
           "filing_date": pd.Timestamp(filed) if filed else period_end + pd.Timedelta(days=40),
           "accession_number": f"acc-{period_end.date()}"}


def _quarters(start_year: int, n: int) -> list[pd.Timestamp]:
    """n consecutive calendar quarter-ends starting at Q1 of `start_year`."""
    return list(pd.date_range(f"{start_year}-03-31", periods=n, freq="QE"))


def _flat(tag: str, ends: list[pd.Timestamp], value: float, **kw) -> list[dict]:
    return [_row(e, value, tag, **kw) for e in ends]


# --------------------------------------------------------------------------- #
# build_tag_ledger                                                            #
# --------------------------------------------------------------------------- #
def test_alternating_tags_collapse_into_one_era_per_run():
    """DTE `shortTermDebt`'s real shape: `ShortTermBorrowings` in the three 10-Qs,
    `DebtCurrent` in the 10-K, every year. That is genuinely SIX eras over two years,
    not two -- collapsing it to "two tags, used at some point" would hide that the swap
    recurs annually, which is the whole finding."""
    ends = _quarters(2012, 8)
    rows = []
    for i, e in enumerate(ends):
        rows.append(_row(e, 100.0 + i, "us-gaap:DebtCurrent" if (i + 1) % 4 == 0
                         else "us-gaap:ShortTermBorrowings"))
    ledger = build_tag_ledger(pd.DataFrame(rows))
    assert len(ledger) == 4                       # STB(3), DC(1), STB(3), DC(1)
    assert ledger["n_periods"].tolist() == [3, 1, 3, 1]
    assert ledger["era_index"].tolist() == [1, 2, 3, 4]
    assert ledger["n_eras"].unique().tolist() == [4]


def test_derived_rows_do_not_split_an_era():
    """A flow field's Q4 is UNCONDITIONALLY derived (`fundamentals_periods.
    decumulate_quarterly_flow`) and so carries no source_tag -- 9.5% of the live table.
    Counting those as their own era would shatter every flow field's history into
    three-quarter fragments around each Q4."""
    ends = _quarters(2020, 8)
    rows = []
    for i, e in enumerate(ends):
        if (i + 1) % 4 == 0:
            rows.append(_row(e, 400.0, None, derived=1.0))
        else:
            rows.append(_row(e, 100.0, "us-gaap:Revenues"))
    ledger = build_tag_ledger(pd.DataFrame(rows))
    assert len(ledger) == 1
    assert ledger.iloc[0]["n_periods"] == 6


def test_latest_filing_wins_for_a_restated_period():
    """`fundamentals_facts` is accession-grain, so a later filing (or an amendment)
    legitimately restates the same period_end under a different concept. The ledger must
    describe the CURRENT state of the series, i.e. the most recent filing."""
    rows = [
        _row("2015-03-31", 50.0, "us-gaap:Old", filed="2015-05-01"),
        _row("2015-03-31", 75.0, "us-gaap:New", filed="2016-02-01"),
    ]
    ledger = build_tag_ledger(pd.DataFrame(rows))
    assert len(ledger) == 1
    assert ledger.iloc[0]["source_tag"] == "us-gaap:New"
    assert ledger.iloc[0]["last_value"] == 75.0


def test_single_tag_history_is_one_era():
    ledger = build_tag_ledger(pd.DataFrame(_flat("us-gaap:Assets", _quarters(2019, 12), 900.0)))
    assert len(ledger) == 1
    assert ledger.iloc[0]["n_eras"] == 1


def test_empty_and_untagged_input_pass_through():
    assert build_tag_ledger(pd.DataFrame()).empty
    assert build_tag_ledger(None).empty
    untagged = pd.DataFrame(_flat(None, _quarters(2019, 4), 10.0, derived=1.0))
    assert build_tag_ledger(untagged).empty


def test_annual_and_quarterly_duration_types_do_not_manufacture_a_switch():
    """`depAmort`'s real shape for several utilities/med-tech filers: the annual bucket
    resolves `DepreciationDepletionAndAmortization` while the quarterly bucket resolves
    the narrower `Depreciation` -- by DESIGN (different candidate-list priority per
    duration_type), not a filer defect. Pooling both duration_types into one
    chronological series by period_end alone made FY alternate with the surrounding
    quarters' tag every single year, manufacturing a "recurring swap" that was actually
    just the annual/quarterly split (confirmed on the live table: SYK `depAmort` showed
    25 fake boundaries this way). Scoping eras by duration_type must keep each bucket a
    single, clean era."""
    quarterly = _flat("us-gaap:Depreciation", _quarters(2019, 12), 40.0,
                      field="depAmort", duration_type="quarterly")
    annual_ends = [pd.Timestamp(f"{y}-12-31") for y in (2019, 2020, 2021)]
    annual = _flat("us-gaap:DepreciationDepletionAndAmortization", annual_ends, 480.0,
                   field="depAmort", duration_type="annual")
    facts = pd.DataFrame(quarterly + annual)
    ledger = build_tag_ledger(facts)
    assert ledger.groupby("duration_type")["era_index"].max().to_dict() == {
        "annual": 1, "quarterly": 1}
    assert detect_tag_switch_breaks(ledger, facts).empty


def test_fy_and_q4_sharing_a_period_end_are_not_dropped_as_duplicates():
    """A calendar-year filer's FY (`duration_type='annual'`) and Q4
    (`duration_type='instant'`/`'quarterly'`) share one `period_end` (Dec 31). Without
    duration_type in the de-dup subset, `drop_duplicates(["ticker","field","period_end"])`
    kept only whichever was filed later and silently discarded the other duration_type's
    fact entirely."""
    same_end = pd.Timestamp("2020-12-31")
    facts = pd.DataFrame([
        _row(same_end, 500.0, "us-gaap:Assets", field="totalAssets",
            duration_type="instant", filed="2021-02-01"),
        _row(same_end, 5000.0, "us-gaap:Revenues", field="totalAssets",
            duration_type="annual", filed="2021-02-15"),
    ])
    ledger = build_tag_ledger(facts)
    assert set(ledger["duration_type"]) == {"instant", "annual"}
    assert ledger.set_index("duration_type")["last_value"].to_dict() == {
        "instant": 500.0, "annual": 5000.0}


# --------------------------------------------------------------------------- #
# detect_tag_switch_breaks                                                    #
# --------------------------------------------------------------------------- #
def test_benign_taxonomy_cutover_is_not_flagged():
    """The dominant real pattern, and the one that must stay quiet: ASC 842 renamed the
    lease-maturity elements (`OperatingLeasesFutureMinimumPaymentsDueCurrent` ->
    `LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths`) for EVERY filer at once.
    Same measure, continuous level -- a detector that fires here is useless, because
    measured on the live table this shape accounts for most switches there are."""
    old = _flat("us-gaap:OperatingLeasesFutureMinimumPaymentsDueCurrent",
                _quarters(2017, 8), 500.0, field="leaseMaturity1y")
    new = _flat("us-gaap:LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths",
                _quarters(2019, 8), 510.0, field="leaseMaturity1y")
    facts = pd.DataFrame(old + new)
    breaks = detect_tag_switch_breaks(build_tag_ledger(facts), facts)
    assert breaks.empty


def test_level_break_across_a_switch_is_flagged_and_scored():
    """The defect worth finding: the concept changes AND the level steps, so the column
    now splices two different measures. Shaped on DTE `shortTermDebt` (~$240M under the
    balance-sheet `ShortTermBorrowings` vs ~$694M under the long-term-debt FOOTNOTE's
    `DebtCurrent`)."""
    before = _flat("us-gaap:ShortTermBorrowings", _quarters(2011, 8), 240.0)
    after = _flat("us-gaap:DebtCurrent", _quarters(2013, 8), 694.0)
    facts = pd.DataFrame(before + after)
    breaks = detect_tag_switch_breaks(build_tag_ledger(facts), facts)
    assert len(breaks) == 1
    row = breaks.iloc[0]
    assert row["check"] == TAG_SWITCH_LEVEL_BREAK
    assert row["from_tag"] == "us-gaap:ShortTermBorrowings"
    assert row["to_tag"] == "us-gaap:DebtCurrent"
    assert row["level_ratio"] > TAG_SWITCH_LEVEL_BREAK_RATIO
    assert round(row["level_ratio"], 2) == round(694.0 / 240.0, 2)


def test_a_shared_transition_is_counted_across_tickers():
    """`n_tickers_same_switch` is what separates the two ROOT CAUSES behind an identical
    symptom: the same from_tag -> to_tag transition made by many filers is a taxonomy
    migration, so a level break there indicts the FIELD's candidate list (global fix),
    while a transition only one filer makes is that filer's own mis-tagging (a
    `FIELD_TAG_DENYLIST` entry). Severity is deliberately NOT downgraded by a high count:
    a shared migration can still change the measure for a subset (the ASC-606 contract
    element captures only fee income for insurers, MetLife ~48x too small)."""
    rows = []
    for ticker in ("AAA", "BBB", "CCC"):
        rows += _flat("us-gaap:Old", _quarters(2017, 8), 100.0, ticker=ticker)
        rows += _flat("us-gaap:New", _quarters(2019, 8), 900.0, ticker=ticker)
    rows += _flat("us-gaap:Old", _quarters(2017, 8), 100.0, ticker="LONE")
    rows += _flat("us-gaap:Odd", _quarters(2019, 8), 900.0, ticker="LONE")
    facts = pd.DataFrame(rows)
    breaks = detect_tag_switch_breaks(build_tag_ledger(facts), facts)
    shared = breaks[breaks["to_tag"] == "us-gaap:New"]
    lone = breaks[breaks["to_tag"] == "us-gaap:Odd"]
    assert shared["n_tickers_same_switch"].unique().tolist() == [3]
    assert lone["n_tickers_same_switch"].tolist() == [1]
    assert (breaks["severity"] == "warning").all()


def test_a_volatile_level_inside_one_tag_does_not_trigger_a_break():
    """`shortTermDebt` is a revolver balance: DTE really does go $0 -> $1,131M quarter to
    quarter with no concept change. Pooling a median over `TAG_SWITCH_BASELINE_PERIODS`
    either side is what keeps that volatility from reading as a measure splice when a
    switch happens to land next to a swing."""
    swings = [0.0, 1131.0, 52.0, 988.0, 38.0, 914.0, 69.0, 758.0]
    before = [_row(e, v, "us-gaap:ShortTermBorrowings")
              for e, v in zip(_quarters(2019, 8), swings)]
    after = [_row(e, v, "us-gaap:LongTermDebtCurrent")
             for e, v in zip(_quarters(2021, 8), swings)]
    facts = pd.DataFrame(before + after)
    breaks = detect_tag_switch_breaks(build_tag_ledger(facts), facts)
    assert breaks.empty


def test_a_switch_across_a_long_gap_is_left_unscored():
    """With years missing between the two eras the level difference is confounded by the
    absent periods, so nothing can be attributed to the tag change. Reporting it anyway
    would put unfalsifiable rows at the top of the queue."""
    before = _flat("us-gaap:Old", _quarters(2011, 4), 100.0)
    after = _flat("us-gaap:New", _quarters(2018, 4), 900.0)
    facts = pd.DataFrame(before + after)
    breaks = detect_tag_switch_breaks(build_tag_ledger(facts), facts)
    assert breaks.empty


def test_a_zero_level_never_scores_as_an_infinite_break():
    """A revolver drawn to exactly $0 is a real, common reading. Dividing by it would put
    an infinite ratio at the top of a queue that is sorted by ratio."""
    before = _flat("us-gaap:Old", _quarters(2019, 4), 0.0)
    after = _flat("us-gaap:New", _quarters(2020, 4), 500.0)
    facts = pd.DataFrame(before + after)
    breaks = detect_tag_switch_breaks(build_tag_ledger(facts), facts)
    assert breaks.empty


def test_breaks_are_ranked_worst_first():
    """The existing `detect_source_tag_misalignment` output carries no magnitude, so its
    675 rows all looked equally urgent. Ranking is the point of this pass."""
    mild = (_flat("us-gaap:A", _quarters(2011, 8), 100.0, field="fieldMild")
            + _flat("us-gaap:B", _quarters(2013, 8), 200.0, field="fieldMild"))
    severe = (_flat("us-gaap:A", _quarters(2011, 8), 100.0, field="fieldSevere")
              + _flat("us-gaap:B", _quarters(2013, 8), 5000.0, field="fieldSevere"))
    facts = pd.DataFrame(mild + severe)
    breaks = detect_tag_switch_breaks(build_tag_ledger(facts), facts)
    assert breaks["field"].tolist() == ["fieldSevere", "fieldMild"]
    assert breaks["level_ratio"].is_monotonic_decreasing
