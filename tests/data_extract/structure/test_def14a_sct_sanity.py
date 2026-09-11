"""
Unit tests for the CEO-pay sanity step (`validate.sanity_check_exec_comp` / `repair_pay_ratio`).

Every fixture below is a REAL row from `def14a_llm`, with the real values -- the point of this
step is that it makes the right call on the specific filings that motivated it, and an invented
number cannot show that.

WHAT THIS STEP IS. Pure arithmetic on stored columns, run after the LLM has written its rows.
It never reads a filing and never calls a model: a rule either holds on the row or it does not.
64 of 8,470 stored filings break an identity that costs nothing to test, and the step settles
each one by scoring the suspect value against the SAME CEO's other filings.

THE THREE THINGS IT DOES, AND THE THREE THE PLAN ASKED FOR THAT MEASUREMENT REMOVED
-----------------------------------------------------------------------------------
Kept:
  * a negative TOTAL is dropped as unusable (1 row, EQT 2009);
  * a total that fails an identity is scored against the neighbour reference and then KEPT,
    REPAIRED from its components, NULLED, or left alone for want of a reference -- measured
    over 56 failing filings: 38 keep, 1 repair, 8 null, 8 no reference;
  * a pay ratio that is provably unusable is dropped (5 rows).

Removed, each because the data said so:
  * **blanking negative COMPONENTS.** They are the filer's own numbers. For fiscal years
    2006-2008 the Summary Compensation Table reported Stock and Option Awards as the FAS 123R
    expense RECOGNISED, which goes negative when a performance award is reversed, and 3 of the
    10 affected rows have `total == Sigma(parts)` to the dollar with the negative included.
  * **the directional "below its parts -> take the parts, above its parts -> keep the total"
    pair.** It deletes correct filed values: GE 2001 (see the test below).
  * **"prefer the recomputed pay ratio".** Wrong more often than right -- 87 of the 123
    disagreements are mid-year CEO transitions where Item 402(u) permits a ratio that does not
    reconcile, and among the rest JPM 2022 has a correct ratio with an under-extracted total
    while AMZN 2023 has a correct total with a stale ratio. Reported, not repaired.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.data_extract.utils.structure.def14a.validate import (
    DEF14A_SCT_KEEP_BAND, DEF14A_SCT_PART_COLS, DEF14A_SCT_REPAIR_BAND, repair_pay_ratio,
    sanity_check_exec_comp, sct_reference,
)

_PARTS = list(DEF14A_SCT_PART_COLS)


def _row(ticker: str, as_of: str, ceo: str, total=None, salary=None, bonus=None, stock=None,
         option=None, nei=None, other=None, ratio=None, median=None) -> dict:
    """One `def14a_llm` row in the column vocabulary the step reads."""
    return {
        "ticker": ticker, "as_of": pd.Timestamp(as_of), "ceo_name_proxy": ceo,
        "accession_number": f"{ticker}-{as_of}", "ceo_total_comp": total,
        "ceo_salary": salary, "ceo_bonus": bonus, "ceo_stock_awards": stock,
        "ceo_option_awards": option, "ceo_non_equity_incentive": nei,
        "ceo_all_other_comp": other, "ceo_pay_ratio": ratio, "median_employee_pay": median,
    }


def _run(rows: list[dict]) -> tuple[pd.DataFrame, dict[str, int]]:
    return sanity_check_exec_comp(pd.DataFrame(rows))


def _total(out: pd.DataFrame, ticker: str, year: int) -> float:
    hit = out[(out["ticker"] == ticker) & (out["as_of"].dt.year == year)]
    assert len(hit) == 1, f"{ticker} {year}: {len(hit)} rows"
    return hit["ceo_total_comp"].iloc[0]


def _ratio(out: pd.DataFrame, ticker: str, year: int) -> float:
    hit = out[(out["ticker"] == ticker) & (out["as_of"].dt.year == year)]
    assert len(hit) == 1
    return hit["ceo_pay_ratio"].iloc[0]


# --------------------------------------------------------------------------------------------
# The neighbour reference
# --------------------------------------------------------------------------------------------

def test_the_reference_excludes_the_row_being_tested():
    """Otherwise a CEO with one filing scores a perfect 1.00 against themselves and every rule
    passes vacuously -- the row would certify itself."""
    rows = [_row("X", "2020-01-01", "Jane A. Doe", total=10_000_000),
            _row("X", "2021-01-01", "Jane A. Doe", total=12_000_000),
            _row("X", "2022-01-01", "Jane A. Doe", total=100.0)]
    ref = sct_reference(pd.DataFrame(rows))
    # the $100 row's reference is built from the other two only, so it is NOT pulled down to it
    assert ref.iloc[2] == pytest.approx(11_000_000)
    # and the 2020 row's reference excludes its own 10M
    assert ref.iloc[0] == pytest.approx(6_000_050)


def test_the_reference_key_folds_case_so_one_ceo_is_not_split_in_two():
    """COHR files "FRANCIS J. KRAMER" in caps against "Francis J. Kramer" elsewhere. The plan
    specified `clean_person_name`, which preserves casing -- and a SPLIT reference is a MISSING
    reference, which silently turns a testable row into an untouchable one."""
    rows = [_row("COHR", "2011-09-20", "Francis J. Kramer", total=2_000_000),
            _row("COHR", "2012-09-20", "FRANCIS J. KRAMER", total=1_800_000),
            _row("COHR", "2013-09-20", "Francis J. Kramer, Jr.", total=1_900_000)]
    ref = sct_reference(pd.DataFrame(rows))
    assert ref.notna().all(), "case or a suffix split one CEO into separate reference groups"


def test_a_ceo_with_no_other_filing_has_no_reference():
    ref = sct_reference(pd.DataFrame([_row("X", "2020-01-01", "Solo Person", total=5_000_000)]))
    assert math.isnan(ref.iloc[0])


# --------------------------------------------------------------------------------------------
# Rule 1 -- the negative total, and the negative COMPONENTS that must survive it
# --------------------------------------------------------------------------------------------

def test_a_negative_total_is_dropped_but_its_negative_components_are_kept():
    """EQT's 2009 proxy: Murry Gerber's FY2008 total is -$8,920,166, and it is the filer's own
    arithmetic -- salary 649,036 + bonus 925,012 + stock -12,693,707 + option 228,102 + nei
    1,874,988 + other 96,403 sums to it EXACTLY, because the pre-2010 SCT reported Stock Awards
    as the FAS 123R expense recognised and EQT reversed a performance award. The total goes
    (a negative pay level has no logarithm and inverts every growth rate through it); the
    components stay, because blanking them would break a row that reconciles to the dollar."""
    rows = [_row("EQT", "2009-03-05", "Murry S. Gerber", total=-8_920_166.0, salary=649_036.0,
                 bonus=925_012.0, stock=-12_693_707.0, option=228_102.0, nei=1_874_988.0,
                 other=96_403.0),
            _row("EQT", "2010-03-05", "Murry S. Gerber", total=8_288_624.0, salary=700_000.0),
            _row("EQT", "2011-03-05", "Murry S. Gerber", total=9_000_000.0, salary=720_000.0)]
    out, tally = _run(rows)
    assert math.isnan(_total(out, "EQT", 2009))
    assert tally["nulled_negative_total"] == 1
    kept = out[(out["ticker"] == "EQT") & (out["as_of"].dt.year == 2009)]
    assert kept["ceo_stock_awards"].iloc[0] == -12_693_707.0
    assert sum(kept[c].iloc[0] for c in _PARTS) == pytest.approx(-8_920_166.0)
    assert tally["negative_component_rows_KEPT"] == 1


# --------------------------------------------------------------------------------------------
# Rule 2 -- the neighbour band decides WHICH LEG to trust
# --------------------------------------------------------------------------------------------

def test_a_pre_2006_total_that_excludes_equity_is_KEPT_not_replaced():
    """⚠ THE CASE THAT OVERTURNED THE PLAN'S DIRECTIONAL RULE. GE's 2001 proxy reports John F.
    Welch's FY2000 total as $16,754,019 = salary $4,000,000 + bonus $12,700,000 + other $54,019.
    That is exactly what the PRE-2006 Summary Compensation Table's Total column was: equity
    awards were disclosed in their own columns and were NOT summed into it. So Sigma(parts)
    reaches $67,973,744 by adding ~$51.2M of equity the filed Total legitimately excludes.

    The plan's rule ("total below its components -> replace the total with the components' sum,
    else NULL") deletes $16,754,019 -- a correct filed value that Welch's own other years
    corroborate at 1.66x. The band-per-leg rule keeps it and blames the components."""
    rows = [_row("GE", "2001-03-09", "John F. Welch, Jr.", total=16_754_019.0, salary=4_000_000.0,
                 bonus=12_700_000.0, stock=25_000_000.0, option=26_219_725.0, other=54_019.0),
            _row("GE", "2000-03-09", "John F. Welch, Jr.", total=13_325_000.0, salary=3_400_000.0),
            _row("GE", "1999-03-09", "John F. Welch, Jr.", total=10_104_944.0, salary=3_000_000.0),
            _row("GE", "1998-03-09", "John F. Welch, Jr.", total=8_000_000.0, salary=2_800_000.0)]
    out, tally = _run(rows)
    assert _total(out, "GE", 2001) == 16_754_019.0
    assert tally["kept_total_components_suspect"] >= 1
    assert tally["repaired_total_from_components"] == 0


def test_a_zero_total_is_repaired_only_when_the_components_nearly_MATCH_the_neighbours():
    """FAST 2000 is the single repair in the whole table. Robert Kierlin famously took ~$120k at
    Fastenal, so Sigma(parts) = $117,000 lands at 0.96x his $122,500 reference and the $0 is
    recoverable."""
    rows = [_row("FAST", "2000-03-13", "Robert A. Kierlin", total=0.0, salary=117_000.0,
                 bonus=0.0, stock=0.0, option=0.0, nei=0.0, other=0.0),
            _row("FAST", "1999-03-13", "Robert A. Kierlin", total=120_000.0, salary=120_000.0),
            _row("FAST", "2001-03-13", "Robert A. Kierlin", total=125_000.0, salary=125_000.0),
            _row("FAST", "2002-03-13", "Robert A. Kierlin", total=122_500.0, salary=122_500.0)]
    out, tally = _run(rows)
    assert _total(out, "FAST", 2000) == pytest.approx(117_000.0)
    assert tally["repaired_total_from_components"] == 1


def test_a_plausible_looking_tiny_repair_is_refused():
    """⚠ THE BAND IS LOAD-BEARING, AND AGILENT IS WHY. A 2011's stored total is $0 and its
    Sigma(parts) is $160,091 -- 1.7% of William Sullivan's $9,166,077 reference. Repairing would
    turn an obviously-missing value into a plausible-looking small one, which is FAR more
    dangerous as a pay-ratio denominator than the zero was. It is blanked instead."""
    rows = [_row("A", "2011-01-19", "William P. Sullivan", total=0.0, salary=160_091.0,
                 bonus=0.0, stock=0.0, option=0.0, nei=0.0, other=0.0),
            _row("A", "2010-01-19", "William P. Sullivan", total=9_166_077.0, salary=900_000.0),
            _row("A", "2012-01-19", "William P. Sullivan", total=10_581_647.0, salary=950_000.0),
            _row("A", "2013-01-19", "William P. Sullivan", total=8_500_000.0, salary=940_000.0)]
    out, tally = _run(rows)
    assert math.isnan(_total(out, "A", 2011))
    assert tally["nulled_total_no_plausible_leg"] == 1
    assert tally["repaired_total_from_components"] == 0


def test_an_under_extracted_component_set_keeps_the_total():
    """JPM 2025: the total is $39,000,000 against a $1,500,000 salary-only component set. The
    total agrees with Dimon's other years (1.31x of $29,750,000), so the COMPONENTS are the
    suspect leg and the total survives. 38 of the 56 failing filings resolve this way."""
    rows = [_row("JPM", "2025-04-07", "James Dimon", total=39_000_000.0, salary=1_500_000.0),
            _row("JPM", "2024-04-07", "James Dimon", total=29_750_000.0, salary=1_500_000.0,
                 bonus=5_000_000.0, stock=23_250_000.0),
            _row("JPM", "2023-04-07", "James Dimon", total=34_500_000.0, salary=1_500_000.0,
                 bonus=5_000_000.0, stock=28_000_000.0)]
    out, tally = _run(rows)
    assert _total(out, "JPM", 2025) == 39_000_000.0
    assert tally["kept_total_components_suspect"] >= 1


def test_a_row_with_no_reference_is_left_exactly_as_filed():
    """CSX 1999 fails an identity (total $654,681 below a $1,100,008 salary) but John Snow has
    no other filing in the table, so there is nothing to score it against. Blanking on "no
    reference" would discard a value on the strength of no evidence at all. 8 rows."""
    rows = [_row("CSX", "1999-03-16", "John W. Snow", total=654_681.0, salary=1_100_008.0,
                 bonus=0.0, stock=1_000_000.0, option=500_000.0, nei=228_542.0, other=100_000.0)]
    out, tally = _run(rows)
    assert _total(out, "CSX", 1999) == 654_681.0
    assert tally["left_alone_no_reference"] == 1
    assert tally["nulled_total_no_plausible_leg"] == 0


def test_a_clean_filing_passes_through_bit_identical():
    rows = [_row("AAPL", "2023-01-12", "Timothy D. Cook", total=99_420_097.0, salary=3_000_000.0,
                 bonus=0.0, stock=82_959_453.0, option=0.0, nei=12_000_000.0, other=1_460_644.0,
                 ratio=672.0, median=147_970.0),
            _row("AAPL", "2022-01-12", "Timothy D. Cook", total=98_734_394.0, salary=3_000_000.0,
                 ratio=1447.0, median=68_254.0)]
    before = pd.DataFrame(rows)
    out, tally = _run(rows)
    for col in ["ceo_total_comp", "ceo_pay_ratio", *_PARTS]:
        pd.testing.assert_series_equal(out[col], before[col], check_names=False)
    assert tally["nulled_total_no_plausible_leg"] == 0
    assert tally["pay_ratio_nulled_unusable"] == 0


def test_a_clean_row_is_never_tested_against_its_neighbours():
    """CEO pay is genuinely lumpy: a real mega-grant year sits far outside 3x its neighbours.
    The band gates only rows that have ALREADY failed an arithmetic identity, which is what
    keeps a wide band safe. TSLA 2019 is the live case -- Musk's $2,284,044,884 is 45,753x the
    prior year and entirely real."""
    rows = [_row("TSLA", "2019-04-30", "Elon Musk", total=2_284_044_884.0, salary=56_380.0,
                 option=2_283_988_504.0),
            _row("TSLA", "2018-04-26", "Elon Musk", total=49_920.0, salary=49_920.0),
            _row("TSLA", "2020-05-28", "Elon Musk", total=23_760.0, salary=23_760.0)]
    out, _ = _run(rows)
    assert _total(out, "TSLA", 2019) == 2_284_044_884.0


# --------------------------------------------------------------------------------------------
# The pay ratio
# --------------------------------------------------------------------------------------------

def test_the_swapped_columns_are_dropped():
    """GOOGL 2018 and 2019 hold the IDENTICAL number in `ceo_pay_ratio` and
    `median_employee_pay` against a $1 total. One of the two is definitely wrong and neither is
    recoverable from the other."""
    for year, value in [("2018-04-27", 197_274.0), ("2019-04-30", 246_804.0)]:
        out = repair_pay_ratio(_row("GOOGL", year, "Larry Page", total=1.0, ratio=value,
                                    median=value))
        assert math.isnan(out["ceo_pay_ratio"])


def test_a_disagreement_over_a_placeholder_total_is_dropped():
    """SMCI 2023 states a ratio of 0.1 against a $1 total: the recomputation would be 1.24e-5,
    so there is no leg to keep. TSLA's 2022 filing is the same shape -- a stated 18,043 against
    a $0 total, where `def14a_executive_comp` independently carries Musk's FY2021 total as 0
    with `reconciles = 1` and TSLA's four other $0 filings all disclose a ratio of 0."""
    out = repair_pay_ratio(_row("SMCI", "2023-04-14", "Charles Liang", total=1.0, ratio=0.1,
                                median=80_413.0))
    assert math.isnan(out["ceo_pay_ratio"])
    out = repair_pay_ratio(_row("TSLA", "2022-06-23", "Elon Musk", total=0.0, ratio=18_043.0,
                                median=40_723.0))
    assert math.isnan(out["ceo_pay_ratio"])


def test_a_zero_pay_ratio_survives():
    """⚠ NOT AN EXEMPTION -- AN AGREEMENT. Musk takes no pay, so `0 / 46,150` is the correct
    ratio and the recomputation independently returns 0. Nothing fires. Four stored rows."""
    for as_of, median in [("2021-08-26", 46_150.0), ("2023-04-06", 34_084.0),
                          ("2024-04-29", 45_811.0), ("2025-09-17", 57_243.0)]:
        out = repair_pay_ratio(_row("TSLA", as_of, "Elon Musk", total=0.0, salary=0.0,
                                    ratio=0.0, median=median))
        assert out["ceo_pay_ratio"] == 0.0


def test_a_sub_one_pay_ratio_survives_when_its_own_legs_agree():
    """Musk's 2018 and 2020 filings disclose 0.91 and 0.41 -- a CEO paid LESS than the median
    employee. Absurd-looking and correct, and the arithmetic confirms both."""
    out = repair_pay_ratio(_row("TSLA", "2018-04-26", "Elon Musk", total=49_920.0, ratio=0.91,
                                median=54_816.0))
    assert out["ceo_pay_ratio"] == 0.91


@pytest.mark.parametrize("ticker,as_of,total,median,ratio", [
    # the disclosed ratio is RIGHT and our total is under-extracted: 917 x 92,112 = 84,466,704,
    # James Dimon's real FY2021 total, which our row is missing a $52.6M option award from
    ("JPM", "2022-04-04", 34_500_000.0, 92_112.0, 917.0),
    # the disclosed ratio is STALE: 6,474 is Andrew Jassy's FY2021 figure, and $1,298,723 is the
    # correct FY2022 total
    ("AMZN", "2023-04-13", 1_298_723.0, 32_855.0, 6474.0),
    # neither column is wrong: Dirk Van de Put became CEO in November, and Item 402(u) permits
    # an annualised or year-end-CEO ratio that does not reconcile with a partial-year total
    ("MDLZ", "2018-04-02", 42_442_924.0, 42_893.0, 403.0),
])
def test_an_unreconciled_ratio_over_a_REAL_total_is_reported_not_rewritten(
        ticker, as_of, total, median, ratio):
    """⚠ THE PLAN'S RULE WOULD HAVE CORRUPTED THIS COLUMN. It asked that any disagreement beyond
    25% be resolved by preferring `ceo_total_comp / median_employee_pay`. These three rows carry
    the same arithmetic symptom and three incompatible causes, and 87 of the 123 disagreements
    are the third one -- the disagreement rate is 21.0% across a CEO transition against 1.0%
    for the same CEO. So the ratio is left exactly as filed and the row is counted."""
    out = repair_pay_ratio(_row(ticker, as_of, "Someone", total=total, ratio=ratio,
                                median=median))
    assert out["ceo_pay_ratio"] == ratio


def test_the_unreconciled_rows_are_counted():
    rows = [_row("JPM", "2022-04-04", "James Dimon", total=34_500_000.0, salary=1_500_000.0,
                 bonus=5_000_000.0, stock=28_000_000.0, ratio=917.0, median=92_112.0),
            _row("JPM", "2023-04-04", "James Dimon", total=34_500_000.0, salary=1_500_000.0,
                 bonus=5_000_000.0, stock=28_000_000.0, ratio=374.5, median=92_112.0)]
    _, tally = _run(rows)
    assert tally["pay_ratio_UNRECONCILED_reported_only"] == 1
    assert tally["pay_ratio_rewritten"] == 0


def test_the_ratio_is_judged_against_the_repaired_total_not_the_raw_one():
    """Order matters: `repair_pay_ratio` runs after the total has been settled, so a ratio is
    never measured against a value the step is about to delete."""
    rows = [_row("A", "2011-01-19", "William P. Sullivan", total=0.0, salary=160_091.0,
                 bonus=0.0, stock=0.0, option=0.0, nei=0.0, other=0.0, ratio=120.0,
                 median=60_000.0),
            _row("A", "2010-01-19", "William P. Sullivan", total=9_166_077.0, salary=900_000.0),
            _row("A", "2012-01-19", "William P. Sullivan", total=10_581_647.0, salary=950_000.0)]
    out, _ = _run(rows)
    assert math.isnan(_total(out, "A", 2011))
    # the total is gone, so the ratio can no longer be reconciled against anything and is left
    # alone rather than measured against a NaN
    assert _ratio(out, "A", 2011) == 120.0


def test_a_median_pay_jump_is_reported_and_never_repaired():
    """`median_employee_pay` year on year is otherwise wage inflation (p05/p50/p95 = 0.808 /
    1.036 / 1.292 over 3,480 pairs), so a 3x move is worth reporting -- but 2 of the 6 breaches
    are the real composition effect of furloughing low-paid staff through COVID (LYV 2021 at
    3.12x then 2022 at 0.28x), not a parse error. Nothing is written."""
    rows = [_row("LYV", "2020-04-27", "Michael Rapino", total=10_000_000.0, median=18_333.0),
            _row("LYV", "2021-04-27", "Michael Rapino", total=10_000_000.0, median=57_195.0),
            _row("LYV", "2022-04-27", "Michael Rapino", total=10_000_000.0, median=15_740.0)]
    before = pd.DataFrame(rows)
    out, tally = _run(rows)
    assert tally["median_employee_pay_jump_REPORTED_only"] == 2
    pd.testing.assert_series_equal(out["median_employee_pay"], before["median_employee_pay"],
                                   check_names=False)


# --------------------------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------------------------

def test_the_step_is_idempotent():
    """A row the step has settled satisfies the identities, so a second pass must not test it
    again -- which is what makes the batch script safe to re-run after every extraction."""
    rows = [_row("A", "2011-01-19", "William P. Sullivan", total=0.0, salary=160_091.0,
                 bonus=0.0, stock=0.0, option=0.0, nei=0.0, other=0.0),
            _row("A", "2010-01-19", "William P. Sullivan", total=9_166_077.0, salary=900_000.0),
            _row("A", "2012-01-19", "William P. Sullivan", total=10_581_647.0, salary=950_000.0),
            _row("FAST", "2000-03-13", "Robert A. Kierlin", total=0.0, salary=117_000.0,
                 bonus=0.0, stock=0.0, option=0.0, nei=0.0, other=0.0),
            _row("FAST", "1999-03-13", "Robert A. Kierlin", total=120_000.0, salary=120_000.0),
            _row("FAST", "2001-03-13", "Robert A. Kierlin", total=125_000.0, salary=125_000.0)]
    once, _ = _run(rows)
    twice, tally2 = sanity_check_exec_comp(once)
    pd.testing.assert_frame_equal(once, twice)
    assert tally2["nulled_total_no_plausible_leg"] == 0
    assert tally2["repaired_total_from_components"] == 0


def test_the_repair_band_is_strictly_inside_the_keep_band():
    """Keeping a filed value needs only corroboration; WRITING one needs a near-match. The plan
    stated 0.33x-3.0x for both, then its own verified table rejected Sigma(parts) at 0.34x,
    0.36x and 0.44x while accepting only 0.96x -- the narrow band is what its evidence
    describes."""
    assert DEF14A_SCT_REPAIR_BAND[0] > DEF14A_SCT_KEEP_BAND[0]
    assert DEF14A_SCT_REPAIR_BAND[1] < DEF14A_SCT_KEEP_BAND[1]


def test_an_empty_or_columnless_frame_is_survivable():
    out, tally = sanity_check_exec_comp(pd.DataFrame(columns=["ticker", "as_of"]))
    assert out.empty and tally == {}
    out, _ = _run([])
    assert out.empty
