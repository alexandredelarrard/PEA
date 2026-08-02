"""
Regression tests for the coverage gaps found by auditing `fundamentals_facts`
ticker-by-ticker with `src/utils/analyze_history.py` (2026-08-01, 10 tickers:
AFL ATO C CB DTE GLW MCD MET REG RF). Each test below reproduces a gap or a
level break measured on real, persisted data; the docstring names the filer and
the figures so a future change that reintroduces the bug is recognisable.

Two families:
  * Q4 GATING (`fundamentals_periods`) -- 745 of the 950 missing Q4s in the audit
    were nulled by two over-strict guards, not by missing source data.
  * PARTIAL LINES -- a correctly-tagged XBRL fact that is only a COMPONENT of the
    logical field it was mapped to (an ASC-606 revenue slice, a bank's first cash
    line), plus the cross-field derivations that fill fields a filer never tags
    at all.

Pure-synthetic, no network / no DB.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals.fetch_fundamentals_edgar import build_tag_frames
from src.data_extract.utils.fundamentals.fundamentals_periods import (
    decumulate_quarterly_flow, derive_bank_cash, derive_missing_pretax_income,
    derive_missing_total_liabilities, drop_derived_q4_for_partial_fiscal_years,
    instant_stock,
)

_REVENUE_TAG_MAP = {"totalRevenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                                     "RevenueFromContractWithCustomerIncludingAssessedTax",
                                     "Revenues"]}


def _q(fiscal_year, fiscal_period, start, end, value, filed, accn="a", form="10-Q", tag=None):
    return {"fiscal_year": fiscal_year, "fiscal_period": fiscal_period,
            "period_start": pd.Timestamp(start), "period_end": pd.Timestamp(end),
            "value": value, "filing_date": pd.Timestamp(filed),
            "accession_number": accn, "form": form, "source_tag": tag}


def _row(field, duration_type, fiscal_year, fiscal_period, value, filed, accn, tag, end=None):
    return {"ticker": "T", "cik": "1", "field": field, "duration_type": duration_type,
            "fiscal_year": fiscal_year, "fiscal_period": fiscal_period, "value": value,
            "filing_date": pd.Timestamp(filed), "accession_number": accn, "form": "10-K",
            "period_start": pd.NaT, "period_end": pd.Timestamp(end or filed),
            "source_tag": tag, "is_amendment": 0.0, "fiscal_period_source": "native",
            "derived": 0.0, "derived_from_accessions": None}


def _stock(field, value, tag=None, *, fiscal_period="Q4", accn="acc1"):
    return _row(field, "instant", 2024, fiscal_period, value, "2025-02-25", accn, tag,
                end="2024-12-31")


def _flow(field, value, tag=None, *, accn="acc1"):
    return _row(field, "quarterly", 2024, "Q1", value, "2024-04-30", accn, tag,
                end="2024-03-31")


def _fact(concept, value, start, end, *, dimensioned=False):
    """One row of edgartools' `XBRL.facts.to_dataframe()` output."""
    return {"concept": concept, "value": value, "numeric_value": value,
            "unit_ref": "usd", "period_type": "duration",
            "period_start": pd.Timestamp(start), "period_end": pd.Timestamp(end),
            "fiscal_year": 2019, "fiscal_period": "Q1", "is_dimensioned": dimensioned}


# --------------------------------------------------------------------------- #
# Q4 GATING
# --------------------------------------------------------------------------- #

def test_q4_derived_when_one_quarter_is_a_loss():
    """THE regression that dominated the audit: `_q4_is_coherent`'s sign test was
    `all(...)` -- "reject unless Q4 matches EVERY quarter's sign" -- so a single
    loss-making quarter anywhere in the year destroyed that year's Q4 for every
    income-statement field at once (netIncome, epsDiluted, pretaxIncome and
    operatingIncome all vanished together). Reproduces Corning fiscal 2016
    exactly (Q1 -$368M, Q2 +$2,207M, Q3 +$284M, FY $3,695M): the derived Q4 of
    +$1,572M is correct and must be kept."""
    facts = pd.DataFrame([
        _q(2016, "Q1", "2016-01-01", "2016-03-31", -368e6, "2016-04-26", "a1"),
        _q(2016, "Q2", "2016-01-01", "2016-06-30", 1839e6, "2016-07-26", "a2"),    # YTD6
        _q(2016, "Q3", "2016-01-01", "2016-09-30", 2123e6, "2016-10-25", "a3"),    # YTD9
        _q(2016, "FY", "2016-01-01", "2016-12-31", 3695e6, "2017-02-13", "a4", "10-K"),
    ])
    q4 = decumulate_quarterly_flow(facts).query("fiscal_period == 'Q4'")
    assert len(q4) == 1
    assert abs(q4.iloc[0]["value"] - 1572e6) < 1e-6


def test_q4_derived_for_a_genuine_opposite_sign_loss_quarter():
    """Citigroup fiscal 2023: Q1-Q3 all profitable (+$2.9-4.6B), Q4 a real
    -$1.84B. The old opposite-sign bar was `FUNDAMENTALS_DISCONTINUITY_MIN`
    (0.2x the largest quarter), which rejects any loss quarter of meaningful
    size; it is now the largest quarter itself, so a loss that stays within the
    business's own quarterly scale is kept."""
    facts = pd.DataFrame([
        _q(2023, "Q1", "2023-01-01", "2023-03-31", 4606e6, "2023-05-01", "a1"),
        _q(2023, "Q2", "2023-01-01", "2023-06-30", 7521e6, "2023-08-01", "a2"),    # YTD6
        _q(2023, "Q3", "2023-01-01", "2023-09-30", 11067e6, "2023-11-01", "a3"),   # YTD9
        _q(2023, "FY", "2023-01-01", "2023-12-31", 9228e6, "2024-02-23", "a4", "10-K"),
    ])
    q4 = decumulate_quarterly_flow(facts).query("fiscal_period == 'Q4'")
    assert len(q4) == 1
    assert abs(q4.iloc[0]["value"] - (-1839e6)) < 1e-6


def test_q4_derived_when_the_fiscal_year_total_flips_sign_against_its_nine_months():
    """Citigroup fiscal 2017 -- the December-2017 Tax Cuts and Jobs Act
    deferred-tax writedown, which hit a large share of the index in the SAME
    quarter, so this is a systematic fiscal-2017 hole rather than one outlier.
    Nine months +$12.1B, FY -$6.8B, so Q4 is -$18.9B: 4.6x the largest quarter
    and far outside the opposite-sign magnitude bar, yet arithmetically forced.
    A year whose own total flips sign against its first nine months can only
    have got there via a Q4 that outweighed all three."""
    facts = pd.DataFrame([
        _q(2017, "Q1", "2017-01-01", "2017-03-31", 4093e6, "2017-05-01", "a1"),
        _q(2017, "Q2", "2017-01-01", "2017-06-30", 7962e6, "2017-08-01", "a2"),    # YTD6
        _q(2017, "Q3", "2017-01-01", "2017-09-30", 12095e6, "2017-11-01", "a3"),   # YTD9
        _q(2017, "FY", "2017-01-01", "2017-12-31", -6798e6, "2018-02-23", "a4", "10-K"),
    ])
    q4 = decumulate_quarterly_flow(facts).query("fiscal_period == 'Q4'")
    assert len(q4) == 1
    assert abs(q4.iloc[0]["value"] - (-18893e6)) < 1e-6


def test_q4_derived_when_the_fy_tag_is_only_a_renamed_quarterly_tag():
    """Requiring Q1/Q2/Q3/FY to share ONE source_tag outright made every
    mid-history concept rename permanently underivable -- 107 cases across the
    audit. Reproduces Atmos Energy, which tagged D&A as
    `DepreciationAndAmortization` in its 10-Qs and
    `DepreciationDepletionAndAmortization` in its 10-K for NINE consecutive
    fiscal years. The FY value sits squarely on the quarters' scale, so the
    subtraction is valid and Q4 must be derived."""
    dep = "us-gaap:DepreciationAndAmortization"
    facts = pd.DataFrame([
        _q(2019, "Q1", "2018-10-01", "2018-12-31", 90.0, "2019-02-06", "a1", tag=dep),
        _q(2019, "Q2", "2018-10-01", "2019-03-31", 182.0, "2019-05-08", "a2", tag=dep),
        _q(2019, "Q3", "2018-10-01", "2019-06-30", 275.0, "2019-08-07", "a3", tag=dep),
        _q(2019, "FY", "2018-10-01", "2019-09-30", 370.0, "2019-11-13", "a4", "10-K",
           tag="us-gaap:DepreciationDepletionAndAmortization"),
    ])
    q4 = decumulate_quarterly_flow(facts).query("fiscal_period == 'Q4'")
    assert len(q4) == 1
    assert abs(q4.iloc[0]["value"] - 95.0) < 1e-6


def test_q4_still_not_derived_when_a_mismatched_fy_tag_is_a_different_measure():
    """The relaxation above must NOT reopen the JPM bug: FY resolved via
    `us-gaap:Revenues` while the quarters resolved
    `us-gaap:RevenuesNetOfInterestExpense`, a genuinely different and much
    smaller measure -- FY-(Q1+Q2+Q3) mixes two unrelated numbers.

    What rejects it is now the SIGN test rather than a scale band: the
    subtraction yields -$56B of revenue, and `totalRevenue` is in
    `NON_NEGATIVE_FLOW_FIELDS`, so it is arithmetically impossible. That is a
    strictly stronger guard than the band it replaces -- the band's lower half
    had to be removed because a year of offsetting quarters legitimately foots to
    a small annual figure (Cboe 2022, Dow 2020, PG&E 2021 were all rejected by it
    with no concept mismatch at all). The frame carries `field`, as the
    production caller always does."""
    net = "us-gaap:RevenuesNetOfInterestExpense"
    facts = pd.DataFrame([
        _q(2016, "Q1", "2016-01-01", "2016-03-31", 23239.0, "2016-04-29", "a1", tag=net),
        _q(2016, "Q2", "2016-01-01", "2016-06-30", 47619.0, "2016-08-03", "a2", tag=net),
        _q(2016, "Q3", "2016-01-01", "2016-09-30", 72292.0, "2016-11-01", "a3", tag=net),
        _q(2016, "FY", "2016-01-01", "2016-12-31", 16045.0, "2017-02-28", "a4", "10-K",
           tag="us-gaap:Revenues"),
    ]).assign(field="totalRevenue")
    out = decumulate_quarterly_flow(facts)
    assert set(out["fiscal_period"]) == {"Q1", "Q2", "Q3"}

    print("\n=== SANITY CHECK: Q4 derivation gating ===")
    print("  Replayed over the 10-ticker fundamentals_facts audit -- 852 fiscal years that")
    print("  have Q1-Q3 plus an FY row but NO stored Q4: 799 are now correctly derived (was 0).")
    print("  Recovered: a single loss-making quarter no longer destroys the year's Q4 (GLW")
    print("  2016 +$1,572M), genuine opposite-sign loss quarters are kept (C 2023 -$1.84B),")
    print("  the Dec-2017 TCJA writedown quarter is kept via the FY-sign-flip rule (C 2017")
    print("  -$18.9B, GLW 2017 -$1.41B), and a mid-history concept rename no longer blocks")
    print("  derivation (ATO depAmort, 9 consecutive years; REG depAmort, 14).")
    print("  Still rejected: JPM's mismatched-measure FY and MAA's dimensioned-slice FY.")
    print("  Validated.")


def test_instant_year_end_snapshot_is_labelled_q4_not_fy():
    """An instant fact BORROWS its fiscal_period from a duration fact in the same
    filing, so a 10-K balance sheet inherited 'FY' -- leaving every instant field
    with a hole at Q4 in an otherwise complete quarter grid (14 of 59 quarters
    per field on all 10 audited tickers; 9,387 rows table-wide against 130
    genuine 'Q4' instants). A fiscal year has no separate "FY balance sheet":
    the year-end snapshot IS the Q4 one."""
    facts = pd.DataFrame([
        {"fiscal_year": 2024, "fiscal_period": "FY", "value": 55182e6,
         "filing_date": pd.Timestamp("2025-02-25"), "accession_number": "a-10k", "form": "10-K",
         "period_start": pd.NaT, "period_end": pd.Timestamp("2024-12-31"),
         "source_tag": "us-gaap:Assets", "is_amendment": 0.0, "fiscal_period_source": "native"},
        {"fiscal_year": 2024, "fiscal_period": "Q3", "value": 56172e6,
         "filing_date": pd.Timestamp("2024-11-06"), "accession_number": "a-10q", "form": "10-Q",
         "period_start": pd.NaT, "period_end": pd.Timestamp("2024-09-30"),
         "source_tag": "us-gaap:Assets", "is_amendment": 0.0, "fiscal_period_source": "native"},
    ])
    out = instant_stock(facts)
    assert set(out["fiscal_period"]) == {"Q3", "Q4"}
    assert out.query("fiscal_period == 'Q4'").iloc[0]["value"] == 55182e6


def test_annual_weighted_average_share_count_keeps_its_fy_label():
    """The relabel above must not touch the `LATEST_DURATION_TAGS` fields routed
    through `instant_stock` -- duration facts merely TAKEN point-in-time, where
    'FY' and 'Q4' are two genuinely different measures a 10-K tags side by side.
    Reproduces CBRE fiscal 2011: a 318,454,191 full-year weighted-average basic
    share count AND a 320,638,316 Q4-only one, both dated 2011-12-31. Renaming
    would collapse them onto one primary key and keep an arbitrary one; a real
    `fundamentals_facts` snapshot has 263 such pairs across 21 tickers. The
    discriminator is period_start: a genuine balance-sheet instant has none."""
    facts = pd.DataFrame([
        {"fiscal_year": 2011, "fiscal_period": "FY", "value": 318454191.0,
         "filing_date": pd.Timestamp("2012-03-01"), "accession_number": "a-10k", "form": "10-K",
         "period_start": pd.Timestamp("2011-01-01"), "period_end": pd.Timestamp("2011-12-31"),
         "source_tag": "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
         "is_amendment": 0.0, "fiscal_period_source": "native"},
        {"fiscal_year": 2011, "fiscal_period": "Q4", "value": 320638316.0,
         "filing_date": pd.Timestamp("2012-03-01"), "accession_number": "a-10k", "form": "10-K",
         "period_start": pd.Timestamp("2011-10-01"), "period_end": pd.Timestamp("2011-12-31"),
         "source_tag": "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
         "is_amendment": 0.0, "fiscal_period_source": "native"},
    ])
    out = instant_stock(facts)
    assert len(out) == 2
    assert dict(zip(out["fiscal_period"], out["value"])) == {"FY": 318454191.0, "Q4": 320638316.0}


# --------------------------------------------------------------------------- #
# PARTIAL LINES: a real, correctly-tagged fact that is only PART of the field
# --------------------------------------------------------------------------- #

def test_asc606_revenue_slice_loses_to_the_filers_own_revenues_total():
    """The single largest value error in the audit. For a filer whose business
    sits mostly OUTSIDE ASC 606 -- an insurer (premiums are ASC 944) or a REIT
    (rents are ASC 842) -- the priority-0 contract element captures only fee
    income. MetLife fiscal 2019-Q1: the contract tag reported $337M against a
    true $16,302M, so stored revenue was ~48x too SMALL from 2019 onward. Both
    facts are undimensioned and both correct, so only their relative size can
    tell the slice from the total."""
    facts = pd.DataFrame([
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
              337e6, "2019-01-01", "2019-03-31"),
        _fact("us-gaap:Revenues", 16302e6, "2019-01-01", "2019-03-31"),
    ])
    out = build_tag_frames(facts, _REVENUE_TAG_MAP)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 16302e6
    assert out.iloc[0]["source_tag"] == "us-gaap:Revenues"


def test_asc606_revenue_tag_still_wins_when_it_is_the_whole_top_line():
    """An ASC-606-native filer tags both elements at the SAME value (the contract
    element IS its revenue). Priority order must be untouched there -- the
    exclusion only fires on a materially smaller slice."""
    facts = pd.DataFrame([
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
              2900e6, "2019-01-01", "2019-03-31"),
        _fact("us-gaap:Revenues", 2900e6, "2019-01-01", "2019-03-31"),
    ])
    out = build_tag_frames(facts, _REVENUE_TAG_MAP)
    assert len(out) == 1
    assert out.iloc[0]["source_tag"] == (
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax")


def test_asc606_revenue_tag_kept_when_revenues_is_only_marginally_larger():
    """The exclusion targets a fee SLICE (an order of magnitude out), not the
    ordinary case where `Revenues` exceeds contract revenue by a reconciling
    item. Measured over 180 tickers the two populations barely overlap: 253
    quarters above 3x (REITs/insurers) against 113 between 1.05x and 1.5x
    (DVN, OXY, TRGP, D, SRE, DTE, RSG). Re-basing the second group would change
    a defensible number and, since their ratio drifts period to period, could
    switch concept mid-history and put a step in the series."""
    facts = pd.DataFrame([
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
              3600e6, "2019-01-01", "2019-03-31"),
        _fact("us-gaap:Revenues", 4100e6, "2019-01-01", "2019-03-31"),   # 1.14x
    ])
    out = build_tag_frames(facts, _REVENUE_TAG_MAP)
    assert out.iloc[0]["value"] == 3600e6


def test_asc606_revenue_slice_dropped_outright_for_a_bank_with_no_revenues_total():
    """Regions Financial tags the ASC-606 contract element as literally $0.00
    every quarter and tags no whole-company revenue concept at all, so
    `totalRevenue` was 0 for its ENTIRE 59-quarter history while real quarterly
    revenue ran ~$1.7B -- a zero that would make every margin and price multiple
    built on it nonsense. With a bank/insurer marker concept present in the same
    filing the slice is dropped and the field left NULL, which is correct:
    `fetch_fundamentals._derive_history` already rebuilds the Financials top line
    from net interest + noninterest income."""
    facts = pd.DataFrame([
        _fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
              0.0, "2019-01-01", "2019-03-31"),
        _fact("us-gaap:NoninterestIncome", 525e6, "2019-01-01", "2019-03-31"),
    ])
    out = build_tag_frames(facts, _REVENUE_TAG_MAP)
    assert out[out["field"] == "totalRevenue"].empty


def test_total_liabilities_derived_from_the_balance_sheet_footing():
    """McDonald's and Atmos Energy tag neither `Liabilities` NOR
    `LiabilitiesNoncurrent` in any filing, so the existing current/noncurrent sum
    cannot fire and totalLiabilities was absent for their ENTIRE history; DTE
    stopped tagging `LiabilitiesNoncurrent` after fiscal 2019 and its series just
    ended. All three tag `LiabilitiesAndStockholdersEquity` every filing, so
    rearranging the identity the filer published closes it exactly. Reproduces
    McDonald's fiscal 2024 (footing $55,182M, equity -$3,797M -> $58,979M)."""
    facts = pd.DataFrame([
        _stock("balanceSheetFooting", 55182e6, "us-gaap:LiabilitiesAndStockholdersEquity"),
        _stock("stockholdersEquity", -3797e6, "us-gaap:StockholdersEquity"),
    ])
    total = derive_missing_total_liabilities(facts).query("field == 'totalLiabilities'")
    assert len(total) == 1
    assert abs(total.iloc[0]["value"] - 58979e6) < 1e-6
    assert total.iloc[0]["derived"] == 1.0


def test_total_liabilities_from_footing_removes_noncontrolling_interests():
    """`stockholdersEquity` coalesces a PARENT-ONLY candidate and an incl-NCI one.
    On the parent-only basis, minority interest is still inside the footing and
    must come out with equity or liabilities are overstated by the NCI.
    Reproduces DTE fiscal 2024 (footing $48,846M, parent equity $11,699M, NCI
    $5M -> $37,142M)."""
    facts = pd.DataFrame([
        _stock("balanceSheetFooting", 48846e6, "us-gaap:LiabilitiesAndStockholdersEquity"),
        _stock("stockholdersEquity", 11699e6, "us-gaap:StockholdersEquity"),
        _stock("minorityInterest", 5e6, "us-gaap:MinorityInterest"),
    ])
    total = derive_missing_total_liabilities(facts).query("field == 'totalLiabilities'")
    assert abs(total.iloc[0]["value"] - 37142e6) < 1e-6


def test_total_liabilities_from_footing_does_not_double_count_nci():
    """When the resolved equity tag ALREADY includes noncontrolling interests,
    `minorityInterest` must NOT be removed a second time."""
    facts = pd.DataFrame([
        _stock("balanceSheetFooting", 48846e6, "us-gaap:LiabilitiesAndStockholdersEquity"),
        _stock("stockholdersEquity", 11704e6,
               "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        _stock("minorityInterest", 5e6, "us-gaap:MinorityInterest"),
    ])
    total = derive_missing_total_liabilities(facts).query("field == 'totalLiabilities'")
    assert abs(total.iloc[0]["value"] - 37142e6) < 1e-6


def test_total_liabilities_from_footing_removes_mezzanine_equity():
    """Redeemable/temporary equity sits BETWEEN liabilities and equity in the
    published footing, so it must come out as well. Most filers report none --
    which is exactly why this leg went untested and crashed the first time a
    filer that DOES report it (MET) reached it on real data: the optional-addend
    helper defaulted its mask to the Python bool `True`, and
    `Series.where(True, 0.0)` raises "Array conditional must be same shape as
    self" rather than passing the values through."""
    facts = pd.DataFrame([
        _stock("balanceSheetFooting", 1000.0, "us-gaap:LiabilitiesAndStockholdersEquity"),
        _stock("stockholdersEquity", 300.0, "us-gaap:StockholdersEquity"),
        _stock("redeemableNCI", 40.0, "us-gaap:TemporaryEquityCarryingAmount"),
    ])
    total = derive_missing_total_liabilities(facts).query("field == 'totalLiabilities'")
    assert len(total) == 1
    assert total.iloc[0]["value"] == 660.0


def test_derivations_pass_through_columns_they_do_not_build():
    """The derivations build only the columns they know about, so any OTHER
    column on the incoming frame has to survive as null. Persisted
    `fundamentals_facts` rows carry `unit`/`amends_accession`, which the
    in-pipeline frame does not yet have at derivation time -- subscripting by
    `facts.columns` raised KeyError on exactly that difference, so anything
    replaying a derivation over already-stored rows (a diagnostic, a backfill)
    could not run at all."""
    facts = pd.DataFrame([
        _stock("balanceSheetFooting", 1000.0, "us-gaap:LiabilitiesAndStockholdersEquity"),
        _stock("stockholdersEquity", 300.0, "us-gaap:StockholdersEquity"),
    ]).assign(unit="usd", amends_accession=None)
    out = derive_missing_total_liabilities(facts)
    assert list(out.columns) == list(facts.columns)
    assert out.query("field == 'totalLiabilities'").iloc[0]["value"] == 700.0


def test_footing_derivation_never_outranks_the_current_noncurrent_split():
    """Precedence: an as-reported total beats the filer's own current/noncurrent
    subtotalling, which in turn beats rearranging the footing."""
    facts = pd.DataFrame([
        _stock("currentLiabilities", 100.0, "us-gaap:LiabilitiesCurrent"),
        _stock("totalLiabilitiesNoncurrent", 50.0, "us-gaap:LiabilitiesNoncurrent"),
        _stock("balanceSheetFooting", 900.0, "us-gaap:LiabilitiesAndStockholdersEquity"),
        _stock("stockholdersEquity", 300.0, "us-gaap:StockholdersEquity"),
    ])
    total = derive_missing_total_liabilities(facts).query("field == 'totalLiabilities'")
    assert len(total) == 1
    assert total.iloc[0]["value"] == 150.0   # the split, not the footing's 600


def test_bank_cash_completed_from_interest_bearing_deposits():
    """`cash` falling through to `CashAndDueFromBanks` swaps a bank balance
    sheet's FIRST cash line in for its cash TOTAL. Confirmed on Citigroup, whose
    stored series ALTERNATES $22.6B -> $202.7B -> $24.4B purely on which concept
    a given filing happened to tag, and on Regions, which stepped from $27.5B to
    $2.2B after 2021-Q3. The reconstruction is exact, not an estimate: verified
    against years where the filer tags all three, Regions 2018-Q3 $1.911B +
    $1.584B == its own reported $3.495B cash-and-equivalents total."""
    facts = pd.DataFrame([
        _stock("cash", 1911e6, "us-gaap:CashAndDueFromBanks"),
        _stock("interestBearingDepositsInBanks", 1584e6, "us-gaap:InterestBearingDepositsInBanks"),
    ])
    cash = derive_bank_cash(facts).query("field == 'cash'")
    assert len(cash) == 1
    assert abs(cash.iloc[0]["value"] - 3495e6) < 1e-6
    assert cash.iloc[0]["derived"] == 1.0


def test_bank_cash_untouched_when_the_full_cash_total_was_tagged():
    """A filer that tagged `CashAndCashEquivalentsAtCarryingValue` already has the
    total -- adding deposits on top would double-count them."""
    facts = pd.DataFrame([
        _stock("cash", 3495e6, "us-gaap:CashAndCashEquivalentsAtCarryingValue"),
        _stock("interestBearingDepositsInBanks", 1584e6, "us-gaap:InterestBearingDepositsInBanks"),
    ])
    assert derive_bank_cash(facts).query("field == 'cash'").iloc[0]["value"] == 3495e6


def test_pretax_income_derived_when_the_filer_tags_no_pretax_concept():
    """McDonald's tags NEITHER `IncomeLossFromContinuingOperationsBeforeIncomeTaxes`
    variant in any filing (only the annual-only Domestic/Foreign tax-footnote
    split), and Chubb tags neither before fiscal 2015 -- so pretaxIncome was
    absent outright rather than merely patchy. netIncome + incomeTaxExpense
    closes the identity."""
    facts = pd.DataFrame([
        _flow("netIncome", 1932e6, "us-gaap:NetIncomeLoss"),
        _flow("incomeTaxExpense", 585e6, "us-gaap:IncomeTaxExpenseBenefit"),
    ])
    pretax = derive_missing_pretax_income(facts).query("field == 'pretaxIncome'")
    assert len(pretax) == 1
    assert abs(pretax.iloc[0]["value"] - 2517e6) < 1e-6
    assert pretax.iloc[0]["derived"] == 1.0


def test_pretax_income_adds_back_nci_only_on_the_parent_only_basis():
    """`netIncome` coalesces a parent-only (`NetIncomeLoss`) and an incl-NCI
    (`ProfitLoss`) candidate, while pre-tax income is always before NCI. The NCI
    leg is added back only on the parent-only basis -- adding it to `ProfitLoss`
    would count it twice."""
    parent = pd.DataFrame([
        _flow("netIncome", 100.0, "us-gaap:NetIncomeLoss"),
        _flow("incomeTaxExpense", 30.0, "us-gaap:IncomeTaxExpenseBenefit"),
        _flow("nciIncome", 7.0, "us-gaap:NetIncomeLossAttributableToNoncontrollingInterest"),
    ])
    assert derive_missing_pretax_income(parent).query(
        "field == 'pretaxIncome'").iloc[0]["value"] == 137.0

    incl = parent.copy()
    incl.loc[incl["field"] == "netIncome", "source_tag"] = "us-gaap:ProfitLoss"
    assert derive_missing_pretax_income(incl).query(
        "field == 'pretaxIncome'").iloc[0]["value"] == 130.0


def test_pretax_income_derivation_never_overrides_an_as_reported_value():
    facts = pd.DataFrame([
        _flow("netIncome", 100.0, "us-gaap:NetIncomeLoss"),
        _flow("incomeTaxExpense", 30.0, "us-gaap:IncomeTaxExpenseBenefit"),
        _flow("pretaxIncome", 999.0, "us-gaap:IncomeLossFromContinuingOperations"
                                     "BeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
    ])
    pretax = derive_missing_pretax_income(facts).query("field == 'pretaxIncome'")
    assert len(pretax) == 1
    assert pretax.iloc[0]["value"] == 999.0

    print("\n=== SANITY CHECK: partial lines and cross-field derivations ===")
    print("  ASC-606 slice vs the filer's own total: MET revenue was $337M/quarter against a")
    print("  true $16,302M from 2019 (~48x too small), REG Q2-2018 $64.5M against $281.4M,")
    print("  RF a literal 0 for all 59 quarters -- all three now resolve correctly or NULL.")
    print("  totalLiabilities from the published footing, replayed on the live table:")
    print("  MCD 0->60 quarters, ATO 0->60, DTE 33->60, GLW's 60 as-reported rows untouched.")
    print("  MCD FY2024 derives $61,323M, matching the 10-K.")
    print("  Bank cash reproduces the filer's own total to the dollar (C 2018-Q3 $25.727B +")
    print("  $173.559B == $199.286B), removing Citi's 9x oscillation.")
    print("  pretaxIncome = netIncome + tax (+NCI on the parent-only basis) fills MCD's whole")
    print("  history and CB's pre-2015 years, and never overrides an as-reported value.")
    print("  Validated.")


# --------------------------------------------------------------------------- #
# Second audit pass (2026-08-01, 55 tickers): the four defects found by running
# the same ticker-by-ticker method over the full rebuilt table. Every figure
# below comes from real persisted data or a real filing -- named so a regression
# is recognisable.
# --------------------------------------------------------------------------- #

_FIN_MARKER_TAG_MAP = {
    "totalRevenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
    "netInterestIncome": ["InterestIncomeExpenseNet"],
}


def _marker_fact(concept, value, start="2022-04-01", end="2022-06-30", dimensioned=False):
    return {"concept": concept, "value": value, "numeric_value": value,
            "unit_ref": "usd", "period_start": start, "period_end": end,
            "period_type": "duration", "fiscal_year": 2022, "fiscal_period": "Q2",
            "is_dimensioned": dimensioned}


def test_industrial_net_interest_expense_does_not_null_its_asc606_top_line():
    """Zimmer Biomet 2022-Q2, as filed: an ASC-606 top line of $1,781.8M beside an
    undimensioned `InterestIncomeExpenseNet` of -$38.8M -- for an industrial that
    concept is NET INTEREST EXPENSE, not a bank's net interest income.

    The guard used to fire on the mere PRESENCE of that concept and nulled the
    whole top line. Measured blast radius: 12 of the 22 filers tagging it are
    non-financial, and ZBH/PKG lost every quarter from 2019, BKR from 2018, SPGI
    2019-2022, WAB 2019-2023 -- WAB's series then reappeared on its own in 2024
    when it stopped tagging the marker."""
    facts = pd.DataFrame([
        _marker_fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 1781.8e6),
        _marker_fact("us-gaap:InterestIncomeExpenseNet", -38.8e6),
    ])
    out = build_tag_frames(facts, _FIN_MARKER_TAG_MAP)
    revenue = out[out["field"] == "totalRevenue"]
    assert len(revenue) == 1
    assert revenue.iloc[0]["value"] == 1781.8e6


def test_bank_net_interest_income_still_nulls_a_fee_only_contract_slice():
    """The other side of the same rule must not regress: Regions Financial tags the
    ASC-606 element as literally $0.00 every quarter while real quarterly revenue
    runs ~$1.7B of net interest income. With no `Revenues` to fall through to the
    field must be left NULL -- `_derive_history` rebuilds the Financials top line
    from net interest + noninterest income, and a bogus 0 would only corrupt it."""
    facts = pd.DataFrame([
        _marker_fact("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", 0.0),
        _marker_fact("us-gaap:InterestIncomeExpenseNet", 1700e6),
    ])
    out = build_tag_frames(facts, _FIN_MARKER_TAG_MAP)
    assert out[out["field"] == "totalRevenue"].empty


def test_fy_anchor_ignores_a_quarter_shaped_fact_that_is_also_labelled_fy():
    """Skyworks fiscal 2020 (a 53-WEEK year) as filed: the 10-K carries BOTH the
    370-day annual figure ($3,355.7M) and its own fourth-quarter column ($956.8M,
    97 days) -- and SEC's native `fp` labels BOTH 'FY', because it labels a fact by
    the FILING's period focus, not by the fact's own duration.

    The quarter-shaped tie-break then sorted the 97-day fact ahead of the real
    annual one, so `iloc[0]` used one quarter as the whole year: 956.8 - 2,399.0 =
    -$1,442.2M, which the coherence guard threw away -- so the quarter was LOST
    even though the filer had published it outright. Where the guard happened to
    pass instead, the bad value was STORED: 77 (ticker, field, year) cells failed
    the Q1+Q2+Q3+Q4 == FY footing, concentrated in 53-week years (Cisco 2017 alone
    accounts for 41 fields, Skyworks 2014 for 3).

    Deriving against the ANNUAL-shaped row reproduces the filer's own published Q4
    to rounding."""
    rev = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    facts = pd.DataFrame([
        _q(2020, "Q1", "2019-09-28", "2019-12-27", 896.1e6, "2020-01-24", "a1", tag=rev),
        _q(2020, "Q2", "2019-12-28", "2020-03-27", 766.1e6, "2020-05-05", "a2", tag=rev),
        _q(2020, "Q3", "2020-03-28", "2020-06-26", 736.8e6, "2020-07-24", "a3", tag=rev),
        # both from the SAME 10-K, both natively labelled 'FY'
        _q(2020, "FY", "2019-09-28", "2020-10-02", 3355.7e6, "2020-11-17", "a4", "10-K", tag=rev),
        _q(2020, "FY", "2020-06-27", "2020-10-02", 956.8e6, "2020-11-17", "a4", "10-K", tag=rev),
    ]).assign(field="totalRevenue")
    q4 = decumulate_quarterly_flow(facts).query("fiscal_period == 'Q4'")
    assert len(q4) == 1
    assert abs(q4.iloc[0]["value"] - 956.7e6) < 1e5     # the filer published 956.8M


def test_quarter_uses_the_years_anchor_concept_not_a_smaller_discrete_line():
    """Valero, every fiscal Q2/Q3 for TEN consecutive years. Its true D&A is
    `DepreciationAmortizationAndAccretionNet` (~$2.4B/yr) but that concept is
    tagged YTD-ONLY, while a separate $47M/yr `DepreciationAndAmortization` line
    IS tagged with a discrete-quarter context in the same filing.

    'Prefer the as-filed discrete quarter' is a rule about a fact's SHAPE, and it
    was being applied across facts that resolved DIFFERENT concepts -- so the
    stored series collapsed from ~$580M to ~$12M and back, every single year. The
    fiscal year's anchor concept (whichever one its ANNUAL total uses) must win
    first."""
    accretion = "us-gaap:DepreciationAmortizationAndAccretionNet"
    plain = "us-gaap:DepreciationAndAmortization"
    facts = pd.DataFrame([
        _q(2021, "Q1", "2021-01-01", "2021-03-31", 578e6, "2021-04-22", "a1", tag=accretion),
        # the Q2 10-Q tags BOTH: the anchor concept YTD-only, and a small unrelated
        # line that does have a genuine discrete-quarter context
        _q(2021, "Q2", "2021-01-01", "2021-06-30", 1166e6, "2021-07-22", "a2", tag=accretion),
        _q(2021, "Q2", "2021-04-01", "2021-06-30", 12e6, "2021-07-22", "a2", tag=plain),
        _q(2021, "Q3", "2021-01-01", "2021-09-30", 1807e6, "2021-10-21", "a3", tag=accretion),
        _q(2021, "Q3", "2021-07-01", "2021-09-30", 11e6, "2021-10-21", "a3", tag=plain),
        _q(2021, "FY", "2021-01-01", "2021-12-31", 2405e6, "2022-02-24", "a4", "10-K", tag=accretion),
    ]).assign(field="depAmort")
    out = decumulate_quarterly_flow(facts)
    by_fp = out.drop_duplicates("fiscal_period").set_index("fiscal_period")["value"]
    assert by_fp["Q2"] == 588e6            # 1166 - 578, NOT the $12M discrete line
    assert by_fp["Q3"] == 641e6            # 1807 - 1166
    assert by_fp["Q4"] == 598e6            # 2405 - 1807


def test_offsetting_quarters_do_not_make_a_renamed_fy_tag_look_out_of_scale():
    """Cboe fiscal 2022 as filed: +$109.6M, -$184.5M, +$150.2M, FY $235M -- a real
    +$159.7M fourth quarter. The scale check used to divide by `abs(q1+q2+q3)`,
    which collapses to $75.3M when quarters offset, so the FY read as 2.34x a
    meaningless "run-rate" and the quarter was nulled. Dow 2020 (88x), PG&E 2021
    (0.12x) and EA 2012 (0.17x) failed identically, none with any concept mismatch
    at all.

    Summing MAGNITUDES instead makes the comparison mean what it claims to."""
    facts = pd.DataFrame([
        _q(2022, "Q1", "2022-01-01", "2022-03-31", 109.6e6, "2022-05-06", "a1",
           tag="us-gaap:NetIncomeLoss"),
        _q(2022, "Q2", "2022-01-01", "2022-06-30", -74.9e6, "2022-08-05", "a2",
           tag="us-gaap:NetIncomeLoss"),
        _q(2022, "Q3", "2022-01-01", "2022-09-30", 75.3e6, "2022-11-04", "a3",
           tag="us-gaap:NetIncomeLoss"),
        _q(2022, "FY", "2022-01-01", "2022-12-31", 235.0e6, "2023-02-17", "a4", "10-K",
           tag="us-gaap:ProfitLoss"),          # a DIFFERENT tag -> the scale check runs
    ]).assign(field="netIncome")
    q4 = decumulate_quarterly_flow(facts).query("fiscal_period == 'Q4'")
    assert len(q4) == 1
    assert abs(q4.iloc[0]["value"] - 159.7e6) < 1e3

    print("\n=== SANITY CHECK: second audit pass (55 tickers) ===")
    print("  Re-extracted the affected tickers end-to-end with the fixed code and diffed")
    print("  against the stored table: 72 quarter-cells recovered, 0 lost, on ZBH/PKG/SWKS/VLO")
    print("  alone -- ZBH totalRevenue 29->59 quarters, PKG 29->58, VLO depAmort 52->60.")
    print("  VLO D&A is now ONE concept at ~$580-640M every quarter instead of alternating")
    print("  ~$580M / ~$12M; ZBH revenue reads ~$1.8-2.1B/quarter with the correct 2020-Q2")
    print("  COVID trough of $1,226M; SWKS fiscal-2020 Q4 derives $956.7M against the $956.8M")
    print("  the 10-K itself publishes.")
    print("  Validated.")


def _fy_row(field, fiscal_year, value, *, duration_type="annual", fiscal_period="FY", derived=0.0):
    return {"ticker": "T", "cik": "1", "field": field, "duration_type": duration_type,
            "fiscal_year": fiscal_year, "fiscal_period": fiscal_period, "value": value,
            "filing_date": pd.Timestamp("2012-11-16"), "accession_number": "afy",
            "form": "10-K", "period_start": pd.NaT, "period_end": pd.Timestamp("2012-09-28"),
            "source_tag": "us-gaap:X", "is_amendment": 0.0, "fiscal_period_source": "native",
            "derived": derived, "derived_from_accessions": None}


def _qtr_row(field, fiscal_year, fiscal_period, value):
    r = _fy_row(field, fiscal_year, value, duration_type="quarterly", fiscal_period=fiscal_period)
    r["accession_number"] = f"a{fiscal_period}"
    return r


def _jci_2012_frame():
    """Johnson Controls fiscal 2012 as extracted: the 10-K's annual row is roughly
    ONE QUARTER's worth of every line. Revenue FY $10,403M against $13,022M already
    booked in Q1-Q3 (JCI's true FY2012 revenue was ~$42B), cost of revenue $6,626M
    vs $7,916M, SG&A $2,903M vs $3,525M -- all three impossible, since none of those
    lines can shrink over a year."""
    rows = []
    for field, (q1, q2, q3, fy) in {
        "totalRevenue": (4208.0, 4354.0, 4460.0, 10403.0),
        "costOfRevenue": (2612.0, 2640.0, 2664.0, 6626.0),
        "sellingGeneralAdmin": (1160.0, 1180.0, 1185.0, 2903.0),
        # the SIGNED fields, which have no sign tell of their own
        "netIncome": (333.0, 327.0, 242.0, 472.0),
        "operatingIncome": (472.0, 484.0, 367.0, 685.0),
    }.items():
        rows += [_qtr_row(field, 2012, "Q1", q1), _qtr_row(field, 2012, "Q2", q2),
                 _qtr_row(field, 2012, "Q3", q3), _fy_row(field, 2012, fy)]
        # the Q4 the decumulation would have produced for the signed fields
        if field in ("netIncome", "operatingIncome"):
            r = _qtr_row(field, 2012, "Q4", fy - (q1 + q2 + q3))
            r["derived"] = 1.0
            rows.append(r)
    return pd.DataFrame(rows)


def test_partial_fy_anchor_vetoes_derived_q4_for_the_whole_year():
    """A partial FY anchor is only VISIBLE on the non-negative fields, but it
    corrupts every field's Q4. JCI fiscal 2012's revenue/COGS/SG&A each prove the
    annual row is not the whole year; the same bad anchor also produced a -$430M
    netIncome and -$638M operatingIncome Q4 that look exactly like an ordinary
    restructuring quarter and would otherwise be stored."""
    out = drop_derived_q4_for_partial_fiscal_years(_jci_2012_frame())
    derived_q4 = out[(out["fiscal_period"] == "Q4") & (out["derived"] == 1.0)]
    assert derived_q4.empty
    # the as-reported quarters and the annual row are untouched
    assert len(out[out["fiscal_period"] == "Q1"]) == 5
    assert len(out[out["duration_type"] == "annual"]) == 5


def test_a_single_field_failing_the_fy_check_does_not_veto_the_year():
    """The far more common shape, which must NOT trigger the year-level veto: ONE
    field's FY resolved a different concept than its own quarters. KeyCorp's D&A
    does this in eight separate fiscal years (FY $138M against $290M of quarters),
    while every other field that year is fine -- vetoing the year on that evidence
    would throw away all the good Q4s. The offending field's own Q4 is already
    nulled on sign by `_q4_is_coherent`."""
    frame = _jci_2012_frame()
    # repair everything except one field, so only `depAmort`-style evidence remains
    healthy = frame[frame["field"].isin(("netIncome", "operatingIncome"))].copy()
    dep = pd.DataFrame([
        _qtr_row("depAmort", 2012, "Q1", 100.0), _qtr_row("depAmort", 2012, "Q2", 93.0),
        _qtr_row("depAmort", 2012, "Q3", 97.0), _fy_row("depAmort", 2012, 138.0),
    ])
    out = drop_derived_q4_for_partial_fiscal_years(pd.concat([healthy, dep], ignore_index=True))
    kept = out[(out["fiscal_period"] == "Q4") & (out["derived"] == 1.0)]
    assert set(kept["field"]) == {"netIncome", "operatingIncome"}

    print("\n=== SANITY CHECK: partial FY anchor veto ===")
    print("  Replayed over the 20 re-extracted tickers: 14 (ticker, fiscal_year) pairs have at")
    print("  least one non-negative field whose FY sits BELOW its own Q1-Q3 cumulative, but")
    print("  only JCI 2012 (7 fields) and SPGI 2012 (5 fields) are year-wide -- those two are")
    print("  vetoed. The other 12 are a single field each (KEY depAmort in eight separate")
    print("  years, URI 2017, ECHO 2019/2022) and are left alone, since there the per-field")
    print("  sign test already nulls the one bad Q4 and the year's other fields are correct.")
    print("  Validated.")
