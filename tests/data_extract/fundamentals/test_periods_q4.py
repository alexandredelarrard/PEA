"""
Phase 4: does the period engine produce discrete quarters that are actually discrete?

Two halves, per `docs/testing.md`'s split.

**Synthetic known-truth** for the arithmetic, because every one of these is a case where
the right answer is a number I can state in advance and the wrong answer is *plausible*:
the two Q4 routes disagreeing, the sign guard that once nulled 745 correct rows, the
share-day derivation, the split-straddling window, the 52/53-week calendar.

**Real data** for the two figures the plan names (AAPL's fiscal-2025 Q4 revenue and
Skyworks' 97-day fiscal-2020 Q4) and for the defect the whole phase exists to remove -- the
frozen TTM. Those cannot be faked: a synthetic fixture proving `FY - YTD9` works says
nothing about whether real filers publish a YTD9.

Real-data tests need EDGAR and are skipped without `SEC_USER_AGENT`.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data_extract.utils.fundamentals import periods as P
from src.data_extract.utils.fundamentals.kpi_catalogue import FieldSpec, load_catalogue

CATALOGUE = load_catalogue("./configs")

#: Stated here rather than read from `configs.yml`, so a test failure means the ENGINE
#: changed and not that somebody retuned a knob. The config values are asserted to match.
GUARDS = P.PeriodGuards(max_opposite_sign_ratio=3.0, concept_switch_scale_max=2.0,
                        share_basis_max_ratio=1.5)


def _spec(name: str = "totalRevenue", **overrides) -> FieldSpec:
    """A catalogue spec, optionally with one attribute bent for the case under test.

    `dataclasses.replace`, not `FieldSpec(**real.__dict__)`: the spec memoises `never_use`
    per regime in its instance dict, so `__dict__` carries a non-field key as soon as any
    resolution has asked for it and the constructor rejects it.
    """
    return replace(CATALOGUE.field(name), **overrides)


def _facts(rows: list[tuple], ticker: str = "TEST", field: str = "totalRevenue",
           concept: str = "us-gaap:Revenues") -> pd.DataFrame:
    """(start, end, value[, filing_date[, concept]]) -> a facts frame the engine can read.
    `duration_type` is derived by the same `period_shape` production uses, so a fixture
    cannot accidentally declare a shape the real pipeline would not."""
    out = []
    for row in rows:
        start, end, value = row[0], row[1], row[2]
        filed = row[3] if len(row) > 3 else end
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        days = (end - start).days
        out.append({"ticker": ticker, "field": field, "period_start": start,
                    "period_end": end, "period_days": days, "value": float(value),
                    "filing_date": pd.Timestamp(filed),
                    "source_concept": row[4] if len(row) > 4 else concept,
                    "duration_type": P.period_shape("duration", days)})
    return pd.DataFrame(out)


#: One ordinary calendar year of the three shapes a filer publishes.
_Q1 = ("2020-01-01", "2020-03-31")
_Y6 = ("2020-01-01", "2020-06-30")
_Y9 = ("2020-01-01", "2020-09-30")
_FY = ("2020-01-01", "2020-12-31")


# ------------------------------------------------------------------ the Q4 ladder ---

def test_the_ytd9_route_wins_when_the_two_q4_routes_disagree():
    """The plan's headline synthetic: build a year where `FY - YTD9` and
    `FY - (Q1+Q2+Q3)` give DIFFERENT answers, and assert the YTD9 one is taken and named.

    They can disagree because YTD9 is one as-reported number while Q1+Q2+Q3 is three, and
    a filer that restates a quarter without restating the nine-month total leaves the two
    inconsistent. Preferring the YTD9 is what makes the validator's `Q1+Q2+Q3+Q4 == FY`
    check a real test instead of an identity -- all 203,798 legacy Q4 rows were derived
    from that identity, so it passed 99.73% by construction.
    """
    facts = _facts([
        (*_Q1, 100.0), ("2020-04-01", "2020-06-30", 100.0), ("2020-07-01", "2020-09-30", 100.0),
        (*_Y6, 200.0), (*_Y9, 290.0),          # the nine-month total says 290, not 300
        (*_FY, 400.0),
    ])
    quarters = P.quarterize(facts, _spec(), GUARDS)
    q4 = quarters[quarters["period_end"] == pd.Timestamp("2020-12-31")].iloc[0]

    assert q4["basis"] == P.FY_MINUS_YTD9
    assert q4["value"] == pytest.approx(110.0)          # 400 - 290, NOT 400 - 300
    print(f"\nSANITY: FY 400 with YTD9 290 and Q1+Q2+Q3 300 -> Q4 = {q4['value']:.0f} "
          f"by {q4['basis']}. The two routes differ by 10 and the as-reported nine-month "
          f"total wins, so Q1+Q2+Q3+Q4 = 410 != FY and the footing check has something "
          f"real to find.")


def test_the_quarter_sum_route_is_the_fallback_and_says_so():
    """With no YTD9 the engine still derives Q4, but records that it used the identity --
    which is exactly the flag `q4_footing` must exclude on."""
    facts = _facts([
        (*_Q1, 100.0), ("2020-04-01", "2020-06-30", 100.0), ("2020-07-01", "2020-09-30", 100.0),
        (*_FY, 400.0),
    ])
    quarters = P.quarterize(facts, _spec(), GUARDS)
    q4 = quarters[quarters["period_end"] == pd.Timestamp("2020-12-31")].iloc[0]
    assert q4["basis"] == P.FY_MINUS_QUARTERS
    assert q4["value"] == pytest.approx(100.0)
    testable = quarters[quarters["basis"].isin([P.AS_REPORTED, P.FY_MINUS_YTD9])]
    print(f"\nSANITY: no YTD9 -> Q4 = {q4['value']:.0f} by {q4['basis']}, and it is "
          f"correctly EXCLUDED from the {len(testable)} independently-derived quarters "
          f"the footing check may use.")


def test_an_as_reported_quarter_is_never_replaced_by_a_derived_one():
    """A filer that publishes its own fourth quarter has said something we should not
    overwrite with arithmetic on two other numbers."""
    facts = _facts([
        (*_Q1, 100.0), ("2020-04-01", "2020-06-30", 100.0), ("2020-07-01", "2020-09-30", 100.0),
        ("2020-10-01", "2020-12-31", 90.0),                 # the filer's own Q4
        (*_Y9, 300.0), (*_FY, 400.0),                       # arithmetic would say 100
    ])
    quarters = P.quarterize(facts, _spec(), GUARDS)
    q4 = quarters[quarters["period_end"] == pd.Timestamp("2020-12-31")]
    assert len(q4) == 1
    assert q4.iloc[0]["basis"] == P.AS_REPORTED
    assert q4.iloc[0]["value"] == pytest.approx(90.0)
    print(f"\nSANITY: the filer's own Q4 of 90 is kept over the derived 100, and only one "
          f"row survives for the window -- {q4.iloc[0]['basis']}.")


def test_the_97_day_fact_a_filer_labels_fy_is_not_the_annual_anchor():
    """Skyworks FY2020 tags BOTH a 370-day and a 97-day fact as `fp='FY'`. Nothing here
    reads that label, so the 97-day fact is classified a QUARTER by its own duration and
    the annual anchor is unambiguous."""
    facts = _facts([
        ("2019-09-28", "2020-10-02", 3356.0),          # 370 days -- the real fiscal year
        ("2020-06-27", "2020-10-02", 956.8),           # 97 days -- also tagged fp='FY'
        ("2019-09-28", "2020-06-26", 2399.2),          # YTD9
    ])
    shapes = dict(zip(facts["period_days"], facts["duration_type"]))
    assert shapes[370] == P.ANNUAL and shapes[97] == P.QUARTERLY
    quarters = P.quarterize(facts, _spec(), GUARDS)
    q4 = quarters[quarters["period_end"] == pd.Timestamp("2020-10-02")]
    assert len(q4) == 1 and q4.iloc[0]["value"] == pytest.approx(956.8)
    print(f"\nSANITY: 370d -> {shapes[370]}, 97d -> {shapes[97]}. The 97-day fp='FY' fact "
          f"becomes Q4 = {q4.iloc[0]['value']:.1f} and never poses as the year.")


# --------------------------------------------------------------------- the guards ---

def test_a_negative_quarter_on_a_non_negative_field_is_refused():
    """The sharpest guard, and the one needing no threshold: a cost line cannot be
    negative, so a negative derived value proves the two inputs measured different things.
    This is CBRE fiscal 2016's -$6.4bn cost of revenue and KeyCorp's -$152M D&A in each of
    eight consecutive years.

    Note which fields it protects. The catalogue declares `totalRevenue` as
    **`sign: "any"`** -- an insurer's top line carries realized investment losses -- so JPM's
    -$63bn derived revenue quarter is NOT caught here. It is caught by the concept and
    scale tests instead, and whatever survives those belongs to the Phase-7 validator.
    """
    spec = _spec("costOfRevenue")
    assert spec.sign == "non_negative"
    facts = _facts([(*_Y9, 300.0), (*_FY, 250.0)], field="costOfRevenue")
    quarters = P.quarterize(facts, spec, GUARDS)
    assert quarters.empty or (quarters["period_end"] != pd.Timestamp("2020-12-31")).all()
    print(f"\nSANITY: FY 250 against YTD9 300 would give Q4 = -50 on costOfRevenue "
          f"(sign={spec.sign}); {len(quarters)} quarters emitted, none for the year end. "
          f"A reason-coded gap beats a negative cost line.")


def test_one_loss_quarter_does_not_destroy_the_years_fourth_quarter():
    """The regression that nulled **745 correct rows**. The legacy rule demanded the derived
    quarter match the sign of EVERY sibling, so a single loss-making quarter anywhere in the
    year threw away that year's Q4 for every income-statement field at once.

    GLW fiscal 2016 verbatim: -368 / +2,207 / +284 against an FY of 3,695 has a perfectly
    correct Q4 of +1,572.
    """
    signed = _spec("netIncome")
    assert signed.sign == "any"
    facts = _facts([
        (*_Q1, -368.0), ("2020-04-01", "2020-06-30", 2207.0),
        ("2020-07-01", "2020-09-30", 284.0), (*_Y9, 2123.0), (*_FY, 3695.0),
    ], field="netIncome")
    quarters = P.quarterize(facts, signed, GUARDS)
    q4 = quarters[quarters["period_end"] == pd.Timestamp("2020-12-31")].iloc[0]
    assert q4["value"] == pytest.approx(1572.0)
    print(f"\nSANITY: quarters -368 / +2,207 / +284 and FY 3,695 -> Q4 = "
          f"{q4['value']:,.0f}. One loss quarter no longer nulls the year.")


def test_a_concept_switch_is_allowed_when_the_two_legs_are_the_same_size():
    """Filers rename a line far more often than they change what it means -- ATO tagged D&A
    two ways for NINE consecutive years. Requiring the legs to share a concept made 107
    real quarters underivable, so a switch is recorded rather than refused, and only the
    scale test decides."""
    facts = _facts([
        (*_Y9, 300.0, "2020-11-01", "us-gaap:DepreciationAndAmortization"),
        (*_FY, 400.0, "2021-02-01", "us-gaap:DepreciationDepletionAndAmortization"),
    ], field="depAmort")
    quarters = P.quarterize(facts, _spec("depAmort"), GUARDS)
    q4 = quarters.iloc[0]
    assert q4["value"] == pytest.approx(100.0)
    assert bool(q4["concept_switch"]) is True
    print(f"\nSANITY: D&A tagged two ways across the year still derives Q4 = "
          f"{q4['value']:.0f}, flagged concept_switch={q4['concept_switch']} so the "
          f"validator can see it.")


def test_a_concept_switch_to_a_different_sized_line_is_refused():
    """The other half of the same rule. Compared as per-day RATES, so the twelve-month and
    nine-month legs are commensurable without a count-based annualisation."""
    facts = _facts([
        (*_Y9, 300.0, "2020-11-01", "us-gaap:OperatingIncomeLoss"),
        (*_FY, 5000.0, "2021-02-01", "us-gaap:Revenues"),      # ~12x the nine-month rate
    ])
    quarters = P.quarterize(facts, _spec(), GUARDS)
    assert quarters.empty
    print(f"\nSANITY: an FY running at 12x the nine-month rate under a different concept "
          f"is refused -- {len(quarters)} quarters emitted.")


# ----------------------------------------------------------------- share counts ---

def test_a_weighted_average_share_count_is_differenced_in_share_days():
    """A share count is not additive, but `average x days` IS -- that product is share-days
    outstanding, and share-days accumulate. So `Q4 = (FY.avg*FY.days - YTD9.avg*YTD9.days)
    / Q4.days` is exact.

    Refusing the derivation instead (the plan's text, and edgartools'
    `_is_additive_concept`) leaves `dilutedShares_ttm` computable at 8% of points, which
    does not protect decision #9's `epsDiluted` -- it deletes it.
    """
    spec = _spec("dilutedShares")
    assert spec.is_additive is False
    # 1,000 shares for the first three quarters, 2,000 in the fourth.
    # Day counts INCLUSIVE of both endpoints, which is what the share-day arithmetic
    # uses: Jan 1 - Sep 30 is 274 days, Oct 1 - Dec 31 is 92, and they foot to 366.
    ytd9, fy = 1_000.0, (1_000.0 * 274 + 2_000.0 * 92) / 366
    facts = _facts([("2020-01-01", "2020-09-30", ytd9), ("2020-01-01", "2020-12-31", fy)],
                   field="dilutedShares",
                   concept="us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding")
    quarters = P.quarterize(facts, spec, GUARDS)
    q4 = quarters.iloc[0]
    assert q4["value"] == pytest.approx(2_000.0, rel=1e-9)
    print(f"\nSANITY: 1,000 shares for 274 days then 2,000 for 92 gives an annual weighted "
          f"average of {fy:,.1f}; differencing in share-days recovers Q4 = "
          f"{q4['value']:,.1f} exactly, not the {fy - ytd9:,.1f} a naive subtraction gives.")


def test_a_stock_split_inside_the_window_refuses_the_trailing_average():
    """A split retroactively rescales every prior share count, and the four quarters of a
    trailing window come from four different filings -- so a window straddling one averages
    two incompatible units. Measured: 45 `dilutedShares` windows across 8 tickers.

    Refused rather than repaired: picking a basis would be guessing which one the consumer
    wants, and the number it feeds (`epsDiluted`) is wrong by an exact integer factor and
    looks entirely plausible.
    """
    spec = _spec("dilutedShares")
    quarters = pd.DataFrame([
        {"ticker": "TEST", "field": "dilutedShares", "period_start": pd.Timestamp(s),
         "period_end": pd.Timestamp(e), "period_days": 91, "value": v,
         "basis": P.AS_REPORTED, "known_from": pd.Timestamp(e), "source_concept": "x",
         "concept_switch": False, "fiscal_year": 2020, "fiscal_quarter": q}
        for q, (s, e, v) in enumerate([
            ("2019-10-01", "2019-12-31", 1_000.0), ("2020-01-01", "2020-03-31", 1_000.0),
            ("2020-04-01", "2020-06-30", 1_000.0), ("2020-07-01", "2020-09-30", 7_000.0),
        ], start=1)])
    ttm = P.trailing_twelve(quarters, spec, guards=GUARDS)
    last = ttm.iloc[-1]
    assert pd.isna(last["value"])
    assert last["dc_code"] == P.SPLIT_BASIS_MISMATCH
    print(f"\nSANITY: three quarters at 1,000 shares then one at 7,000 (a 7:1 split) -> "
          f"TTM is NULL with dc_code={last['dc_code']}, not the 2,500 a mean would give.")


# ------------------------------------------------------------------------- TTM ---

def test_a_ttm_needs_four_contiguous_quarters_and_never_a_carried_forward_annual():
    """The staircase fix. The legacy fallback carried the last annual forward up to four
    quarters and froze **1,622 of 26,242 consecutive `totalRevenue` pairs (6.2%)**, making
    `revenueGrowth` exactly 0 for three quarters in four. Coverage drops here on purpose."""
    spec = _spec()
    rows = [("2020-01-01", "2020-03-31", 100.0), ("2020-04-01", "2020-06-30", 110.0),
            ("2021-01-01", "2021-03-31", 120.0), ("2021-04-01", "2021-06-30", 130.0)]
    quarters = pd.DataFrame([
        {"ticker": "TEST", "field": "totalRevenue", "period_start": pd.Timestamp(s),
         "period_end": pd.Timestamp(e), "period_days": (pd.Timestamp(e) - pd.Timestamp(s)).days,
         "value": v, "basis": P.AS_REPORTED, "known_from": pd.Timestamp(e),
         "source_concept": "x", "concept_switch": False, "fiscal_year": 2021,
         "fiscal_quarter": 1} for s, e, v in rows])
    ttm = P.trailing_twelve(quarters, spec, guards=GUARDS)
    # Four quarters exist, but they span two years with a two-quarter hole in between.
    assert ttm["value"].isna().all()
    assert set(ttm["dc_code"].dropna()) == {P.INSUFFICIENT_QUARTERS}
    print(f"\nSANITY: four quarters spanning 18 months with a hole -> "
          f"{int(ttm['value'].isna().sum())}/{len(ttm)} NULL with "
          f"{P.INSUFFICIENT_QUARTERS}. No annual is carried forward to fill it.")


# ------------------------------------------------------------- the fiscal calendar ---

def test_a_four_four_five_retail_calendar_labels_its_own_quarters():
    """Kroger's fiscal Q1 is 16 weeks and Q2-Q4 are 12. A fixed day-count divisor puts the
    111-day Q1 in two buckets; dividing the offset by THAT YEAR's own quarter length and
    rounding is exact."""
    year_ends = [pd.Timestamp("2025-02-01"), pd.Timestamp("2026-01-31")]
    windows = [("2025-02-02", "2025-05-24"), ("2025-05-25", "2025-08-16"),
               ("2025-08-17", "2025-11-08"), ("2025-11-09", "2026-01-31")]
    quarters = pd.DataFrame([
        {"ticker": "KR", "field": "totalRevenue", "period_start": pd.Timestamp(s),
         "period_end": pd.Timestamp(e),
         "period_days": (pd.Timestamp(e) - pd.Timestamp(s)).days, "value": 1.0,
         "basis": P.AS_REPORTED, "known_from": pd.Timestamp(e), "source_concept": "x",
         "concept_switch": False} for s, e in windows])
    labelled = P.label_fiscal_periods(quarters, year_ends)
    assert list(labelled["fiscal_quarter"]) == [1, 2, 3, 4]
    assert set(labelled["fiscal_year"]) == {2026}
    print(f"\nSANITY: a 111/83/83/83-day retail year labels as "
          f"{list(labelled['fiscal_quarter'])} in FY{labelled['fiscal_year'].iloc[0]}, "
          f"with the 16-week Q1 in exactly one bucket.")


def test_the_year_the_filer_is_still_inside_labels_from_its_start():
    """The three quarters filed since the last 10-K are the ones a model trades on. Ranking
    from the END labelled them Q2/Q3/Q4; anchoring on the year's own start gives Q1/Q2/Q3."""
    year_ends = [pd.Timestamp("2025-09-27"), pd.Timestamp("2026-09-26")]
    windows = [("2025-09-28", "2025-12-27"), ("2025-12-28", "2026-03-28"),
               ("2026-03-29", "2026-06-27")]
    quarters = pd.DataFrame([
        {"ticker": "AAPL", "field": "totalRevenue", "period_start": pd.Timestamp(s),
         "period_end": pd.Timestamp(e), "period_days": 90, "value": 1.0,
         "basis": P.AS_REPORTED, "known_from": pd.Timestamp(e), "source_concept": "x",
         "concept_switch": False} for s, e in windows])
    labelled = P.label_fiscal_periods(quarters, year_ends)
    assert list(labelled["fiscal_quarter"]) == [1, 2, 3]
    print(f"\nSANITY: three quarters into an unfinished fiscal year label as "
          f"{list(labelled['fiscal_quarter'])}, not the 2/3/4 a rank-from-the-end gives.")


def test_a_missing_annual_report_does_not_swallow_three_years_of_quarters():
    """A gap in the annual facts is a gap in what survived entity scoping, not in the
    calendar. Unfilled, MAA's 2013 quarters were all labelled FY2017 Q1 -- 69 collisions
    across 11 tickers."""
    facts = _facts([("2013-01-01", "2013-12-31", 1.0), ("2017-01-01", "2017-12-31", 1.0)])
    ends = P.fiscal_year_ends(facts)
    years = [e.year for e in ends]
    assert years[:5] == [2013, 2014, 2015, 2016, 2017]
    print(f"\nSANITY: annual facts for 2013 and 2017 only -> a calendar of {years}. "
          f"The three missing years are interpolated and the next one extrapolated, so no "
          f"quarter falls into the wrong bucket.")


def test_the_same_quarter_tagged_twice_a_day_apart_collapses():
    """Filers nudge the boundary day between filings: GS tags Q1-2013 as both `-> 03-30`
    and `-> 03-31`. Keyed on the exact window both survive and the fiscal label lands on
    two rows at once."""
    facts = _facts([
        ("2013-01-01", "2013-03-30", 100.0, "2013-05-01"),
        ("2013-01-01", "2013-03-31", 101.0, "2014-05-01"),
    ])
    quarters = P.quarterize(facts, _spec(), GUARDS)
    assert len(quarters) == 1
    assert quarters.iloc[0]["value"] == pytest.approx(101.0)      # the later filing wins
    print(f"\nSANITY: the same quarter tagged `-> 03-30` and `-> 03-31` collapses to "
          f"{len(quarters)} row at {quarters.iloc[0]['value']:.0f}, the later filing's.")


def test_the_configured_guards_are_the_ones_this_file_asserts_against():
    """A knob move in `configs.yml` must fail loudly here rather than quietly changing what
    the engine accepts."""
    live = P.load_guards("./configs")
    assert live == GUARDS
    print(f"\nSANITY: configs/configs.yml carries {live}, matching the values every guard "
          f"test above is written against.")


# --------------------------------------------------------------------- real data ---

_REAL_TICKERS = {
    "AAPL": ("Information Technology", "Technology Hardware & Equipment",
             "Technology Hardware, Storage & Peripherals"),
    "SWKS": ("Information Technology", "Semiconductors & Semiconductor Equipment",
             "Semiconductors"),
    "XOM": ("Energy", "Energy", "Integrated Oil & Gas"),
}


@pytest.fixture(scope="module")
def real_periods() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`(facts, quarters, ttm)` for three tickers, straight through the production path.

    AAPL for the September fiscal year and the plan's named Q4 figure, SWKS for the 97-day
    fiscal-2020 Q4, XOM for the frozen-TTM baseline the staircase fix exists to remove
    (36% of its legacy consecutive revenue pairs were identical).
    """
    if not os.getenv("SEC_USER_AGENT", "").strip():
        pytest.skip("SEC_USER_AGENT unset -- the real-data checks need EDGAR")
    from edgar import Company, set_identity

    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import filing_rows

    set_identity(os.environ["SEC_USER_AGENT"])
    rows: list[dict] = []
    for ticker, (sector, group, sub) in _REAL_TICKERS.items():
        gics = {"sector": sector, "industry_group": group, "sub_industry": sub}
        company = Company(ticker)
        filings = [f for f in company.get_filings(form=["10-K", "10-Q"])
                   if pd.Timestamp(f.filing_date) >= pd.Timestamp("2018-01-01")
                   and not str(f.form).upper().endswith("/A")]
        for filing in filings:
            rows.extend(filing_rows(ticker, str(company.cik), filing, CATALOGUE, gics))
    facts = pd.DataFrame(rows)
    quarters, ttm = [], []
    for _, group in facts.groupby("ticker"):
        q, t, _ = P.build_periods(group, CATALOGUE)
        quarters.append(q)
        ttm.append(t)
    return facts, pd.concat(quarters, ignore_index=True), pd.concat(ttm, ignore_index=True)


def test_apples_fiscal_2025_fourth_quarter_revenue(real_periods):
    """$102.466 bn, the edgartools-verified figure, and it must arrive by the PRIMARY
    ladder -- if it comes from `FY - (Q1+Q2+Q3)` the phase has not done its job."""
    _, quarters, _ = real_periods
    q4 = quarters[(quarters.ticker == "AAPL") & (quarters.field == "totalRevenue")
                  & (quarters.period_end == pd.Timestamp("2025-09-27"))]
    assert len(q4) == 1
    assert q4.iloc[0]["value"] == pytest.approx(102.466e9, rel=1e-4)
    assert q4.iloc[0]["basis"] == P.FY_MINUS_YTD9
    assert q4.iloc[0]["fiscal_quarter"] == 4
    print(f"\nSANITY: AAPL fiscal-2025 Q4 revenue = ${q4.iloc[0]['value'] / 1e9:.3f} bn by "
          f"{q4.iloc[0]['basis']}, labelled FY{q4.iloc[0]['fiscal_year']} "
          f"Q{q4.iloc[0]['fiscal_quarter']} -- matching the published figure.")


def test_skyworks_ninety_seven_day_fiscal_2020_fourth_quarter(real_periods):
    """$956.8M over 97 days. The fact is tagged `fp='FY'` and nothing here reads that."""
    _, quarters, _ = real_periods
    q4 = quarters[(quarters.ticker == "SWKS") & (quarters.field == "totalRevenue")
                  & (quarters.period_end == pd.Timestamp("2020-10-02"))]
    assert len(q4) == 1
    assert q4.iloc[0]["value"] == pytest.approx(956.8e6, rel=1e-3)
    assert q4.iloc[0]["period_days"] == 97
    assert q4.iloc[0]["fiscal_quarter"] == 4
    print(f"\nSANITY: SWKS fiscal-2020 Q4 = ${q4.iloc[0]['value'] / 1e6:.1f}M over "
          f"{int(q4.iloc[0]['period_days'])} days, labelled Q{q4.iloc[0]['fiscal_quarter']} "
          f"-- the filer's own published number, from a fact it labels fp='FY'.")


def test_the_ttm_no_longer_repeats_itself(real_periods):
    """XOM's legacy `totalRevenue` TTM was frozen on **36%** of consecutive pairs, because
    the annual figure was carried forward for up to four quarters."""
    _, _, ttm = real_periods
    revenue = ttm[(ttm.field == "totalRevenue") & ttm.value.notna()].sort_values(
        ["ticker", "period_end"])
    report = {}
    for ticker, group in revenue.groupby("ticker"):
        pairs = max(len(group) - 1, 0)
        report[ticker] = (int((group.value.diff() == 0).sum()), pairs)
    assert all(frozen == 0 for frozen, _ in report.values())
    print("\nSANITY: frozen consecutive TTM revenue pairs, against a 6.2% universe-wide "
          "legacy baseline (XOM 36%): "
          + ", ".join(f"{t} {f}/{n}" for t, (f, n) in sorted(report.items())))


def test_a_derived_year_foots_to_the_number_the_filer_published(real_periods):
    """The strongest available check on the three rungs below Q4: sum the four discrete
    quarters and compare with the filer's OWN annual fact, which the engine never reads
    when it derives Q1-Q3."""
    facts, _, ttm = real_periods
    annual = (facts[(facts.duration_type == P.ANNUAL) & facts.value.notna()]
              .sort_values("filing_date")
              .drop_duplicates(["ticker", "field", "period_start", "period_end"], keep="last"))
    sums = ttm[ttm.basis == P.TTM_FOUR_QUARTERS]
    joined = sums.merge(annual[["ticker", "field", "period_end", "value"]].rename(
        columns={"value": "as_filed"}), on=["ticker", "field", "period_end"])
    joined = joined[joined.as_filed.abs() > 0]
    relative = (joined.value - joined.as_filed).abs() / joined.as_filed.abs()
    within = (relative < 0.005).mean()
    assert len(joined) > 100
    assert within > 0.90
    print(f"\nSANITY: {len(joined):,} (ticker, field, fiscal year-end) points where the "
          f"four derived quarters can be compared with the filer's own annual figure -- "
          f"{within:.1%} agree within 0.5%, median error {relative.median():.4%}. "
          f"The residual is restatement and basis change, not derivation.")


def test_the_q4_derivation_survives_a_holdout_against_the_filers_own_quarter(real_periods):
    """The Phase-4 HOLD-OUT proof, folded in as a standing assertion.

    Take every (ticker, field, fiscal year) where the filer published **all three** of the
    FY fact, the YTD9 fact **and its own discrete Q4**. The engine prefers the as-reported
    quarter and never derives these, so forcing the derivation there is a genuine hold-out
    with ground truth -- the only such test available, because everywhere else the
    derivation IS the answer and there is nothing independent to check it against.

    `FY - YTD9 == reported Q4` is an IDENTITY whenever the filer's own three numbers are
    mutually consistent, so the result splits cleanly and both halves matter:

      * where the filer FOOTS, a miss is our arithmetic. Measured on both 26-ticker
        rosters: 591 and 752 cases, 94.5% / 92.4% footing, and of those **98.7% / 99.0%
        within 1%**. The 14 residuals are the filer's own rounding measured against a small
        Q4 denominator -- NEE fiscal 2017 is typical, a $2M gap on a $5,173M year being
        0.04% of the year and 1.1% of the $186M quarter.
      * where it does NOT foot, no method can match both its Q4 and its FY, and the
        disagreement measures the FILER. VLO fiscal 2012 operating income: YTD9 $2,426M +
        Q4 $1,584M = $4,010M against a published FY of $5,044M.

    The bar here is deliberately below the measured 98.7% so the test tracks a REGRESSION
    in the derivation rather than drifting with each roster's restatement history.
    """
    facts, _, _ = real_periods
    d = facts[facts.value.notna() & facts.period_start.notna()].copy()
    for column in ("period_start", "period_end", "filing_date"):
        d[column] = pd.to_datetime(d[column])

    def latest(shape: str) -> pd.DataFrame:
        sub = d[d.duration_type == shape]
        return (sub.sort_values(["period_end", "filing_date"])
                .drop_duplicates(["ticker", "field", "period_end"], keep="last"))

    annual, ytd9, quarterly = latest(P.ANNUAL), latest(P.YTD9), latest(P.QUARTERLY)
    held = annual.merge(
        ytd9[["ticker", "field", "period_start", "period_end", "value"]],
        on=["ticker", "field", "period_start"], suffixes=("_fy", "_y9"))
    held = held[held.period_end_y9 < held.period_end_fy]
    held = held.merge(
        quarterly[["ticker", "field", "period_end", "value"]].rename(
            columns={"period_end": "period_end_fy", "value": "q4_reported"}),
        on=["ticker", "field", "period_end_fy"])
    held = held[held.q4_reported.abs() > 0]
    # The floor is what THIS fixture can supply, not what the roster-wide sweep supplies.
    # `_REAL_TICKERS` is three companies and only SWKS publishes a discrete Q4 at all, so
    # the hold-out population is 24 cases; the 1,596-point in-sample and 1,800-point
    # out-of-sample versions of this same measurement live in the Phase 4b log.
    assert len(held) >= 20, f"only {len(held)} hold-out cases -- too few to prove anything"

    # The ENGINE's arithmetic, not a naive subtraction. A non-additive field is differenced
    # in SHARE-DAYS with BOTH endpoints counted; scoring a plain subtraction would measure a
    # method the engine deliberately does not use, and dropping the +1 cost the share-count
    # median error a factor of 50 when it was measured.
    additive = held.field.map(lambda f: CATALOGUE.field(f).is_additive)
    days_fy = (held.period_end_fy - held.period_start).dt.days + 1
    days_y9 = (held.period_end_y9 - held.period_start).dt.days + 1
    days_q4 = (days_fy - days_y9).replace(0, pd.NA).astype(float)
    derived = np.where(
        additive, held.value_fy - held.value_y9,
        (held.value_fy * days_fy - held.value_y9 * days_y9) / days_q4)
    relative = (derived - held.q4_reported).abs() / held.q4_reported.abs()

    # Does the FILER's own trio foot? Where it does not, the gap is not ours to close.
    filer_gap = np.where(
        additive,
        (held.value_y9 + held.q4_reported - held.value_fy).abs() / held.value_fy.abs(),
        ((held.value_y9 * days_y9 + held.q4_reported * days_q4) / days_fy
         - held.value_fy).abs() / held.value_fy.abs())
    foots = filer_gap < 0.005

    within_1pc = (relative[foots] < 0.01).mean()
    assert foots.mean() > 0.80, f"only {foots.mean():.1%} of the filers' own trios foot"
    assert within_1pc > 0.95
    print(f"\nSANITY: {len(held):,} hold-out cases where the filer published the FY fact, "
          f"the YTD9 fact AND its own discrete Q4. The filer's own three numbers foot in "
          f"{foots.mean():.1%}; on those, the derivation is exact to the dollar in "
          f"{(relative[foots] < 1e-9).mean():.1%}, within 0.1% in "
          f"{(relative[foots] < 0.001).mean():.1%} and within 1% in {within_1pc:.1%} "
          f"(median {relative[foots].median():.6%}). Where the filer does NOT foot, no "
          f"method can match both its Q4 and its FY.")


# ------------------------------------------- the annual-masquerading-as-a-quarter guard ---


def _duration_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(start, end, value) triples -> the duration frame `quarterize` consumes."""
    out = []
    for start, end, value in rows:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        days = (e - s).days
        out.append({"ticker": "TEST", "field": "totalRevenue", "period_start": s,
                    "period_end": e, "period_days": days,
                    "duration_type": P.period_shape("duration", days),
                    "value": value, "filing_date": e + pd.Timedelta(days=30),
                    "source_concept": "us-gaap:Revenues"})
    return pd.DataFrame(out)


def _revenue_spec():
    return load_catalogue("./configs").field("totalRevenue")


def test_an_annual_tagged_against_a_q4_context_does_not_become_q4():
    """ORCL's real shape: the fiscal-year figure carried on a ~92-day fourth-quarter window.

    Left alone it outranks `FY - YTD9`, because an as-reported quarter always beats a
    derived one, and Q4 reads the whole year. Fiscal 2022: $42,440M against a true $11,840M.
    """
    frame = _duration_frame([
        ("2021-06-01", "2021-08-31", 9_728e6),      # Q1, genuine
        ("2021-06-01", "2021-11-30", 20_087e6),     # YTD6
        ("2021-06-01", "2022-02-28", 30_600e6),     # YTD9
        ("2022-03-01", "2022-05-31", 42_440e6),     # the ANNUAL value on a Q4 window
        ("2021-06-01", "2022-05-31", 42_440e6),     # the annual itself
    ])
    q = P.quarterize(frame, _revenue_spec(), P.load_guards("./configs"))
    q4 = q[q.period_end == pd.Timestamp("2022-05-31")]
    assert len(q4) == 1
    assert q4.iloc[0].basis == P.FY_MINUS_YTD9
    assert q4.iloc[0].value == pytest.approx(11_840e6)
    assert q.value.sum() == pytest.approx(42_440e6)      # the four quarters now foot


def test_a_fourth_quarter_that_really_is_the_whole_year_is_kept():
    """The condition-3 case: nine months of zero, so Q4 == FY is the truth.

    Without the interim-accumulation test the guard would delete a legitimate value, which
    is a worse failure than the one it is fixing.
    """
    frame = _duration_frame([
        ("2021-06-01", "2021-08-31", 0.0),
        ("2021-06-01", "2021-11-30", 0.0),
        ("2021-06-01", "2022-02-28", 0.0),
        ("2022-03-01", "2022-05-31", 500e6),
        ("2021-06-01", "2022-05-31", 500e6),
    ])
    q = P.quarterize(frame, _revenue_spec(), P.load_guards("./configs"))
    q4 = q[q.period_end == pd.Timestamp("2022-05-31")]
    assert len(q4) == 1
    assert q4.iloc[0].basis == P.AS_REPORTED
    assert q4.iloc[0].value == pytest.approx(500e6)


def test_a_normal_fourth_quarter_is_untouched():
    """The overwhelmingly common shape, asserted so the guard cannot regress it."""
    frame = _duration_frame([
        ("2021-06-01", "2021-08-31", 9_728e6),
        ("2021-06-01", "2021-11-30", 20_087e6),
        ("2021-06-01", "2022-02-28", 30_600e6),
        ("2022-03-01", "2022-05-31", 11_840e6),     # the real Q4
        ("2021-06-01", "2022-05-31", 42_440e6),
    ])
    q = P.quarterize(frame, _revenue_spec(), P.load_guards("./configs"))
    q4 = q[q.period_end == pd.Timestamp("2022-05-31")]
    assert q4.iloc[0].basis == P.AS_REPORTED
    assert q4.iloc[0].value == pytest.approx(11_840e6)


# --------------------------------------- D1b: the same defect with NO annual to compare ---
# 4c.8 / decision 24. D1's conditions 1 and 2 both need the filer's own annual figure, so
# they cannot run when the mislabelled fact is the ONLY place the year appears. The plan
# proposed gating this on the WINDOW LENGTH -- "a ~365-day fact in a quarterly slot" -- but
# measurement refutes the premise: ORCL's window really is 91 days, so `duration_type` is
# `quarterly` and there is no length anomaly to see. The non-circular evidence that IS
# available is the filer's own nine-month cumulative; both facts are as-filed and neither is
# a derived quarter.


def test_an_annual_on_a_q4_window_with_no_annual_fact_is_refused():
    """ORCL fiscal 2020: all three vintages stamp the full-year `us-gaap:Revenues` with a
    Q4 window and NONE of them publishes an annual-window fact for that year. So `FY - YTD9`
    cannot run either, and the as-reported row would win by default at **$39,068M** against
    a true ~$10,439M -- a $39bn Q4 propagating into four TTM windows, `revenueGrowth`, and
    every peer z-score built on them.

    Refused, not reclassified. Reclassifying would be inference: the only test for "is this
    really the year?" would use the very quarters being derived.
    """
    frame = _duration_frame([
        ("2019-06-01", "2019-08-31", 9_218e6),      # Q1, genuine
        ("2019-06-01", "2019-11-30", 18_832e6),     # YTD6
        ("2019-06-01", "2020-02-29", 28_629e6),     # YTD9 -- the only comparison available
        ("2020-03-01", "2020-05-31", 39_068e6),     # the YEAR, on a Q4 window
    ])                                              # and NO annual fact anywhere
    refusals: list[dict] = []
    q = P.quarterize(frame, _revenue_spec(), P.load_guards("./configs"),
                     refusals=refusals)
    q4 = q[q.period_end == pd.Timestamp("2020-05-31")]
    print("\n=== SANITY CHECK: D1b, an annual on a Q4 window with no annual fact ===")
    print(f"  the fact      : $39,068M on a 91-day window ending 2020-05-31")
    print(f"  nine months   : $28,629M -- SMALLER than the 'quarter', which no real Q4 is")
    print(f"  rows kept for that window: {len(q4)}")
    print(f"  refusals      : {[(r['dc_code'], r['value']) for r in refusals]}")
    assert q4.empty, "the mislabelled annual must not survive as a fourth quarter"
    assert [r["dc_code"] for r in refusals] == [P.AMBIGUOUS_DURATION]
    assert refusals[0]["value"] == pytest.approx(39_068e6)
    print("  OK: refused and reason-coded, rather than stored as a $39bn quarter.")


def test_d1b_keeps_a_fourth_quarter_that_really_is_the_whole_year():
    """The condition-3 analogue, and the reason D1b is not simply "quarter > nine months".

    A capex programme that only spends in Q4 is unusual but not wrong, and with nine months
    of zero the quarter legitimately IS the year. The materiality floor -- the cumulative
    must exceed 1% of the quarter -- is what keeps that row, exactly as it does in D1.
    """
    frame = _duration_frame([
        ("2019-06-01", "2019-08-31", 0.0),
        ("2019-06-01", "2019-11-30", 0.0),
        ("2019-06-01", "2020-02-29", 0.0),
        ("2020-03-01", "2020-05-31", 500e6),
    ])
    refusals: list[dict] = []
    q = P.quarterize(frame, _revenue_spec(), P.load_guards("./configs"),
                     refusals=refusals)
    q4 = q[q.period_end == pd.Timestamp("2020-05-31")]
    print("\n=== SANITY CHECK: D1b does not eat a legitimate Q4-equals-FY ===")
    print(f"  nine months $0, Q4 $500M -> rows kept {len(q4)}, refusals {len(refusals)}")
    assert len(q4) == 1 and q4.iloc[0].value == pytest.approx(500e6)
    assert not refusals
    print("  OK: silence in the cumulative is not evidence of a mislabelled year.")


def test_d1b_leaves_a_normal_fourth_quarter_alone_when_no_annual_exists():
    """A filer that simply never tags the annual window is common and is not a defect. The
    ordinary shape -- a fourth quarter roughly a third of the nine months before it -- must
    pass untouched, or D1b would delete a quarter for every such filer."""
    frame = _duration_frame([
        ("2019-06-01", "2019-08-31", 9_218e6),
        ("2019-06-01", "2019-11-30", 18_832e6),
        ("2019-06-01", "2020-02-29", 28_629e6),
        ("2020-03-01", "2020-05-31", 10_439e6),     # the real Q4
    ])
    refusals: list[dict] = []
    q = P.quarterize(frame, _revenue_spec(), P.load_guards("./configs"),
                     refusals=refusals)
    q4 = q[q.period_end == pd.Timestamp("2020-05-31")]
    print("\n=== SANITY CHECK: D1b on the ordinary no-annual filer ===")
    print(f"  nine months $28,629M, Q4 $10,439M -> kept {len(q4)}, refusals {len(refusals)}")
    assert len(q4) == 1 and q4.iloc[0].basis == P.AS_REPORTED
    assert not refusals
    print("  OK: the guard needs the quarter to EXCEED the nine months, not merely to exist.")


@pytest.fixture(scope="module")
def orcl_quarters() -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """ORCL's real revenue quarters, the D1b refusals, and the FACTS they were built from.

    The facts frame is returned because the refusal moved layers (cluster `2603621e89ab`):
    `fetch_fundamentals_sec._drop_note_only_quarter` now drops the mislabelled years while
    the filing is being read, so `build_periods` never sees them and D1b has nothing left to
    refuse. The evidence is therefore in the fact rows, not in the `refusals` list.

    Since `_retry_without`, those three filings no longer produce a value-less stub either:
    withholding `us-gaap:Revenues` lets the ASC 606 element resolve and the annual windows
    come back, so the fixture now carries ORCL's real fiscal 2020-2022 top line.

    Scoped to filings from 2017-06 rather than the full history, but no tighter -- and the
    lower bound is load-bearing in a way worth recording, because getting it wrong changes
    the answer. ORCL stamps the full year into a Q4 context in fiscal 2018 through 2022, and
    the ANNUAL-window fact that lets D1 handle fiscal 2021 and 2022 arrives in the FY2023 and
    FY2024 10-Ks. Truncate the window at 2022 and D1b fires on three years instead of one --
    correctly, since all three facts really are mislabelled annuals, but not the outcome
    production sees. `quarterize` is handed the ticker's WHOLE stored history, so the fixture
    has to be wide enough to contain the later vintages or it tests a different question.
    """
    if not os.getenv("SEC_USER_AGENT", "").strip():
        pytest.skip("SEC_USER_AGENT unset -- the real-data checks need EDGAR")
    from edgar import Company, set_identity

    from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import filing_rows

    set_identity(os.environ["SEC_USER_AGENT"])
    gics = {"sector": "Information Technology", "industry_group": "Software & Services",
            "sub_industry": "Systems Software"}
    company = Company("ORCL")
    rows: list[dict] = []
    for filing in company.get_filings(form=["10-K", "10-Q"]):
        filed = pd.Timestamp(filing.filing_date)
        if filed < pd.Timestamp("2017-06-01"):
            continue
        rows.extend(filing_rows("ORCL", str(company.cik), filing, CATALOGUE, gics))
    facts = pd.DataFrame(rows)
    refusals: list[dict] = []
    quarters, _ttm, _instants = P.build_periods(facts, CATALOGUE, refusals=refusals)
    return quarters, refusals, facts


def test_orcls_mislabelled_years_never_become_quarters(orcl_quarters):
    """The real-data pairing for cluster `2603621e89ab`, and the fire-rate check.

    ORCL stamps its full-year `us-gaap:Revenues` into a 91-day Q4 context in fiscal 2018,
    2019, 2020, 2021 AND 2022 -- nine facts across three 10-Ks, fiscal 2022 carrying $42,440M
    where the quarter is $11,840M.

    This test used to assert that fiscal 2020's Q4 was REFUSED, because D1b was the only guard
    that could reach it and refusing was the best it could do. That is no longer the outcome
    and it was never a good one: a refused year leaves `Q4 = FY - YTD9` uncomputable, and the
    point-in-time quarter then carries the PRIOR quarter with nothing to say so. What the
    guards owe is stronger -- **no window ever carries a year's value as a quarter, and every
    year that can be derived is** -- so that is what is pinned here.

    All five years now come out at $10-12bn against the $39-42bn the mislabelled facts carry,
    fiscal 2020 included, and fiscal 2020 is the load-bearing one: no other vintage reaches
    back to it, so it derives only because `_retry_without` recovered its annual from the
    fiscal 2020 10-K itself.
    """
    quarters, refusals, facts = orcl_quarters
    revenue = quarters[quarters["field"] == "totalRevenue"]
    coded = [r for r in refusals if r["dc_code"] == P.AMBIGUOUS_DURATION]
    years = ("2018-05-31", "2019-05-31", "2020-05-31", "2021-05-31", "2022-05-31")

    print("\n=== SANITY CHECK: ORCL's Q4-windowed annuals, fiscal 2018-2022 ===")
    for year_end in years:
        got = revenue[revenue["period_end"] == pd.Timestamp(year_end)]
        if got.empty:
            print(f"  {year_end}  REFUSED -- no Q4 row")
        else:
            row = got.iloc[0]
            print(f"  {year_end}  ${row.value / 1e9:6.3f}bn via {row.basis}")
    print(f"  D1b refusals: {len(coded)} (expected 0 -- the facts layer refused first)")

    assert not coded, (
        "D1b saw a mislabelled year -- `_drop_note_only_quarter` should have refused it "
        "while the filing was read, before `fundamentals_facts` could assert it")
    for year_end in years:
        row = revenue[revenue["period_end"] == pd.Timestamp(year_end)]
        assert len(row) == 1, f"{year_end}: expected exactly one Q4 row"
        assert row.iloc[0].basis == P.FY_MINUS_YTD9, (
            f"{year_end}: a fourth quarter here can only be derived, never as-reported -- "
            f"the only as-reported candidate is the mislabelled year")
        assert 8e9 < row.iloc[0].value < 20e9, (
            f"{year_end}: ${row.iloc[0].value / 1e9:.3f}bn is not a fourth quarter")

    as_filed = facts[(facts["field"] == "totalRevenue")
                     & (facts["duration_type"] == P.QUARTERLY)
                     & (facts["period_end"].isin([pd.Timestamp(y) for y in years]))]
    assert as_filed.empty, (
        f"{len(as_filed)} mislabelled year(s) still stored as a quarter in "
        f"fundamentals_facts -- the substrate every Tier-2/3 check reads")
    print("  OK: 0 of 9 mislabelled years survive as quarters; all 5 fourth quarters derive")


def test_the_retry_recovers_the_annual_the_filer_tagged_under_the_other_element(
        orcl_quarters):
    """The second half of cluster `2603621e89ab`: refusing the lie is not recovering the truth.

    Dropping the nine mislabelled years left `totalRevenue` resolving to NOTHING in three
    10-Ks, and that is not a fixed field -- it is a quieter broken one. Fiscal 2020's annual
    existed in NO filing we stored, so `Q4 = FY - YTD9` could not run for three years and the
    point-in-time fourth quarter silently carried the PRIOR quarter instead: 9,796 / 10,085 /
    10,513 $M at `as_of` 2020-06-22, 2021-06-21 and 2022-06-21, against true quarters of
    ~10,440 / ~11,259 / 11,840. No check fires on a carried-forward quarter, because
    `revenue_q` has no period of its own to disagree with.

    Oracle tags the correct figure in those same filings under the ASC 606 element, on proper
    364/365-day windows, and the catalogue already ranks it second in `fallback_concepts`.
    `_retry_without` withholds the concept whose every period was refused and re-resolves, so
    the second candidate finally gets asked.

    Fiscal 2020 is the assertion that matters: its annual is recoverable ONLY from the
    fiscal 2020 10-K itself, because the later vintages that rescue 2021 and 2022 do not
    reach back that far.
    """
    quarters, _refusals, facts = orcl_quarters
    revenue = facts[facts["field"] == "totalRevenue"]
    annual = revenue[revenue["duration_type"] == P.ANNUAL]
    broken = {"0001564590-20-030125", "0001564590-21-033616", "0001564590-22-023675"}

    recovered = annual[annual["accession_number"].isin(broken)]
    print("\n=== SANITY CHECK: annuals recovered from the three broken 10-Ks ===")
    for r in recovered.sort_values(["accession_number", "period_end"]).itertuples():
        print(f"  {r.accession_number}  {str(r.period_end)[:10]}  "
              f"${r.value / 1e9:6.3f}bn  {r.source_concept.split(':')[-1]}")

    assert len(recovered) == 9, "3 fiscal years x 3 filings, every one an annual window"
    assert set(recovered["source_concept"]) == {
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"}, (
        "the retry must land on the ASC 606 element, not back on `Revenues`")
    assert recovered["value"].notna().all(), "a recovered annual is not a value-less stub"

    fy2020 = annual[annual["period_end"] == pd.Timestamp("2020-05-31")]
    assert not fy2020.empty, (
        "fiscal 2020's annual is in NO other vintage -- if the retry misses it, Q4 2020 is "
        "unrecoverable and the point-in-time row keeps carrying Q3")
    assert fy2020["value"].min() == fy2020["value"].max() == 39_068_000_000.0
    assert str(fy2020["accession_number"].min()) == "0001564590-20-030125", (
        "recovered from the fiscal 2020 10-K itself, not from a later restatement")

    q4_2020 = quarters[(quarters["field"] == "totalRevenue")
                       & (quarters["period_end"] == pd.Timestamp("2020-05-31"))]
    assert len(q4_2020) == 1 and q4_2020.iloc[0].basis == P.FY_MINUS_YTD9
    assert 10.3e9 < q4_2020.iloc[0].value < 10.6e9, (
        f"Q4 2020 is ${q4_2020.iloc[0].value / 1e9:.3f}bn, not the ~$10.44bn "
        f"$39,068M - YTD9 implies")
    print(f"  fiscal 2020 Q4 now DERIVES at ${q4_2020.iloc[0].value / 1e9:.3f}bn "
          f"(was: no row at all, and the PIT quarter carried Q3's $9.796bn)")


def test_d1b_keeps_a_genuine_loss_quarter_bigger_than_the_nine_months():
    """LLY fiscal 2017, and the reason D1b compares SIGNED values rather than magnitudes.

    LLY's Q4 2017 net income is **-$1,656.9M** -- the Tax Cuts and Jobs Act charge -- against
    a nine-month **+$1,452.8M**. A magnitude-only test says "the quarter exceeds the nine
    months, so it must be the year" and deletes a correct quarter. The four as-filed quarters
    foot to LLY's real FY2017 net loss of $204.1M, which is exactly how the annual-footing
    report surfaced it.

    The catalogue cannot rescue this: `netIncome` AND `totalRevenue` both declare
    `sign: any`, so gating the guard on the field's sign would have disabled the ORCL case it
    exists for. Direction of the cumulative is the discriminator.
    """
    frame = _duration_frame([
        ("2017-01-01", "2017-03-31", -110.8e6),
        ("2017-01-01", "2017-06-30", 897.2e6),
        ("2017-01-01", "2017-09-30", 1_452.8e6),      # nine months POSITIVE
        ("2017-10-01", "2017-12-31", -1_656.9e6),     # Q4 a bigger NEGATIVE
    ])                                                # and no annual fact anywhere
    refusals: list[dict] = []
    q = P.quarterize(frame, CATALOGUE.field("netIncome"), P.load_guards("./configs"),
                     refusals=refusals)
    q4 = q[q.period_end == pd.Timestamp("2017-12-31")]
    print("\n=== SANITY CHECK: D1b keeps a genuine loss quarter ===")
    print(f"  nine months +$1,452.8M, Q4 -$1,656.9M -> kept {len(q4)}, refusals {len(refusals)}")
    print(f"  four quarters sum to {q['value'].sum() / 1e6:,.1f}M "
          f"(LLY's reported FY2017 net loss was -204.1M)")
    assert len(q4) == 1 and q4.iloc[0].value == pytest.approx(-1_656.9e6)
    assert not refusals, "a loss quarter is not a mislabelled year"
    assert q["value"].sum() == pytest.approx(-204.1e6, rel=1e-6)
    print("  OK: opposite-signed cumulative means the year turned, not a tagging defect.")
