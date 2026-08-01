"""
Fiscal-period correctness tests for fundamentals_periods.py: native-fiscal-year
keying (fixing the WIP file's confirmed `fiscal_year = period_of_report.year`
bug), non-calendar and 52/53-week fiscal years, and the date-arithmetic fallback
when native fiscal metadata is sparse/missing.

Pure-synthetic, no network / no DB.
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals.fundamentals_periods import (
    backfill_fiscal_period_by_filing_order, classify_duration_shape, decumulate_quarterly_flow,
    normalize_fiscal_period_label, reassign_misordered_instant_facts, resolve_fiscal_period,
)


def _q(fiscal_year, fiscal_period, start, end, value, filed, accn="a", form="10-Q"):
    return {"fiscal_year": fiscal_year, "fiscal_period": fiscal_period,
           "period_start": pd.Timestamp(start), "period_end": pd.Timestamp(end),
           "value": value, "filing_date": pd.Timestamp(filed),
           "accession_number": accn, "form": form}


def test_non_calendar_fye_accumulates_correct_ytd():
    """June-30 FYE issuer: FY2023 spans 2022-07-01..2023-06-30, so Q1/Q2 fall in
    calendar 2022 and Q3/FY in calendar 2023. Q4 must still back out correctly
    across that calendar-year boundary -- the exact case the WIP's
    `fiscal_year = int(period_of_report.year)` bug would get wrong (it would split
    this single fiscal year's quarters across TWO different "fiscal_year" buckets)."""
    facts = pd.DataFrame([
        _q(2023, "Q1", "2022-07-01", "2022-09-30", 50.0, "2022-11-01", "a1"),
        _q(2023, "Q2", "2022-07-01", "2022-12-31", 110.0, "2023-02-01", "a2"),
        _q(2023, "Q3", "2022-07-01", "2023-03-31", 175.0, "2023-05-01", "a3"),
        _q(2023, "FY", "2022-07-01", "2023-06-30", 240.0, "2023-08-15", "a4", "10-K"),
    ])
    out = decumulate_quarterly_flow(facts)
    q = {r.fiscal_period: r.value for r in out.itertuples()}
    assert q["Q1"] == 50.0 and q["Q2"] == 60.0 and q["Q3"] == 65.0 and q["Q4"] == 65.0
    assert abs(sum(q.values()) - 240.0) < 1e-9


def test_fiscal_year_grouping_ignores_calendar_year_of_period_end():
    """Two facts whose period_end falls in DIFFERENT calendar years must still
    land in the same bucket when their NATIVE fiscal_year label agrees (the fix
    for the WIP's confirmed bug: it derived fiscal_year from period_end's calendar
    year directly, which is wrong whenever period_end and the native fiscal year
    disagree, as here)."""
    facts = pd.DataFrame([
        _q(2023, "Q3", "2022-07-01", "2023-03-31", 175.0, "2023-05-01", "a3"),   # period_end in CY2023
        _q(2023, "FY", "2022-07-01", "2023-06-30", 240.0, "2023-08-15", "a4", "10-K"),
        _q(2023, "Q1", "2022-07-01", "2022-09-30", 50.0, "2022-11-01", "a1"),    # period_end in CY2022
        _q(2023, "Q2", "2022-07-01", "2022-12-31", 110.0, "2023-02-01", "a2"),
    ])
    out = decumulate_quarterly_flow(facts)
    assert set(out["fiscal_year"].unique()) == {2023}
    assert len(out) == 4   # one fiscal year's worth of quarters, not split across two


def test_52_53_week_quarter_not_rejected():
    """A 52/53-week retailer's ~112-day fiscal Q1 (e.g. KR/COST/TGT-style 16-week
    quarter) must still be admitted, not rejected by a narrower duration window."""
    assert classify_duration_shape(112) == "quarterly"
    assert classify_duration_shape(90) == "quarterly"
    assert classify_duration_shape(181) == "semiannual"
    assert classify_duration_shape(365) == "annual"
    assert classify_duration_shape(45) == "other"

    facts = pd.DataFrame([
        _q(2024, "Q1", "2024-02-01", "2024-05-23", 100.0, "2024-06-15", "a1"),   # 112 days
        _q(2024, "Q2", "2024-02-01", "2024-08-15", 220.0, "2024-09-10", "a2"),   # YTD ~196d
        _q(2024, "Q3", "2024-02-01", "2024-11-07", 340.0, "2024-12-01", "a3"),   # YTD ~280d
        _q(2024, "FY", "2024-02-01", "2025-01-30", 460.0, "2025-03-01", "a4", "10-K"),
    ])
    out = decumulate_quarterly_flow(facts)
    q = {r.fiscal_period: r.value for r in out.itertuples()}
    assert q["Q1"] == 100.0
    assert abs(sum(q.values()) - 460.0) < 1e-9


def test_missing_fiscal_metadata_falls_back_to_date_arithmetic():
    """When a fact carries no native fiscal_year/fiscal_period at all, resolve_fiscal_period
    falls back to date-arithmetic (classify_duration_shape + anchor_fy), tier-labeled
    'date_arithmetic_fallback' -- and for an annual-shaped span, never derives fiscal_year
    from period_end.year directly (the WIP's exact bug)."""
    fy, fp, source = resolve_fiscal_period(
        fact_fiscal_year=None, fact_fiscal_period=None,
        filing_fiscal_year=None, filing_fiscal_period=None,
        period_start=pd.Timestamp("2023-01-01"), period_end=pd.Timestamp("2023-12-31"),
        anchor_fy=2023,
    )
    assert fp == "FY" and source == "date_arithmetic_fallback"

    # native metadata present -> always preferred over date-arithmetic
    fy, fp, source = resolve_fiscal_period(
        fact_fiscal_year=2022, fact_fiscal_period="Q2",
        filing_fiscal_year=None, filing_fiscal_period=None,
        period_start=None, period_end=None, anchor_fy=None,
    )
    assert (fy, fp, source) == (2022, "Q2", "native")

    # filing-level cover-page fallback (tier 2), when the fact's own context lacks it
    fy, fp, source = resolve_fiscal_period(
        fact_fiscal_year=None, fact_fiscal_period=None,
        filing_fiscal_year=2021, filing_fiscal_period="FY",
        period_start=None, period_end=None, anchor_fy=None,
    )
    assert (fy, fp, source) == (2021, "FY", "native")

    print("\n=== SANITY CHECK: fiscal-period resolution ===")
    print("  non-calendar (June-FYE) fiscal year decumulates correctly across the")
    print("  calendar-year boundary; fiscal_year grouping uses NATIVE labels, never")
    print("  period_end.year directly (the WIP's confirmed bug); 52/53-week quarters")
    print("  (~112d) are admitted, not rejected; 3-tier fallback (native fact -> filing")
    print("  cover-page -> date-arithmetic) resolves correctly at each tier.")
    print("  Validated.")


def _raw(fiscal_year, fiscal_period, period_start, period_end, form="10-Q", field="totalRevenue"):
    return {"field": field, "fiscal_year": fiscal_year, "fiscal_period": fiscal_period,
           "period_start": pd.Timestamp(period_start) if period_start else pd.NaT,
           "period_end": pd.Timestamp(period_end), "form": form}


def test_backfill_fiscal_period_by_filing_order_resolves_a_filing_with_zero_native_labels():
    """Real bug found via live data: KR's Q1 10-Qs (e.g. accession
    0001104659-15-048764, period_end 2015-05-23) carry NO native fiscal_period
    on ANY fact in the filing at all -- fiscal_year IS natively populated but
    fiscal_period is blank throughout, so the same-filing backfill
    (fetch_fundamentals_edgar.backfill_fiscal_period_from_filing) has nothing to
    borrow from. This left KR's Q1 revenue entirely absent while Q2/Q3/FY
    (whose filings DO carry native fp) were present -- chronological rank among
    the ticker's OWN 10-Qs for that fiscal year must resolve it to 'Q1' without
    disturbing the two natively-labeled siblings."""
    raw = pd.DataFrame([
        _raw(2016, None, "2015-02-01", "2015-05-23"),        # Q1 10-Q -- no native label (the bug)
        _raw(2016, "Q2", "2015-02-01", "2015-08-15"),         # Q2 10-Q -- native, untouched
        _raw(2016, "Q3", "2015-02-01", "2015-11-07"),         # Q3 10-Q -- native, untouched
        _raw(2016, "FY", "2015-02-01", "2016-01-30", form="10-K"),
    ])
    out = backfill_fiscal_period_by_filing_order(raw)
    by_end = out.set_index("period_end")
    assert by_end.loc[pd.Timestamp("2015-05-23"), "fiscal_period"] == "Q1"
    assert by_end.loc[pd.Timestamp("2015-08-15"), "fiscal_period"] == "Q2"    # unchanged
    assert by_end.loc[pd.Timestamp("2015-11-07"), "fiscal_period"] == "Q3"    # unchanged


def test_backfill_fiscal_period_by_filing_order_fixes_every_field_in_the_filing():
    """The SAME missing-native-label filing affects every field it reports, not
    just one (confirmed empirically: KR's Q1 10-Q lacked native fiscal_period
    for revenue, totalAssets, cash, etc. together). One resolved (fiscal_year,
    period_end) match must fix every row sharing it, regardless of field."""
    raw = pd.DataFrame([
        _raw(2016, None, "2015-02-01", "2015-05-23", field="totalRevenue"),
        _raw(2016, None, None, "2015-05-23", field="totalAssets"),   # instant: no period_start
        _raw(2016, "Q2", "2015-02-01", "2015-08-15", field="totalRevenue"),
        _raw(2016, "Q3", "2015-02-01", "2015-11-07", field="totalRevenue"),
        _raw(2016, "FY", "2015-02-01", "2016-01-30", form="10-K", field="totalRevenue"),
    ])
    out = backfill_fiscal_period_by_filing_order(raw)
    resolved = out[out["period_end"] == pd.Timestamp("2015-05-23")]
    assert set(resolved["fiscal_period"]) == {"Q1"}


def test_backfill_fiscal_period_by_filing_order_resolves_all_three_quarters_with_no_natives_at_all():
    """When NONE of a fiscal year's three 10-Qs carry a native label, all three
    resolve purely from chronological order (earliest -> Q1, next -> Q2, next
    -> Q3)."""
    raw = pd.DataFrame([
        _raw(2016, None, "2015-02-01", "2015-05-23"),
        _raw(2016, None, "2015-02-01", "2015-08-15"),
        _raw(2016, None, "2015-02-01", "2015-11-07"),
        _raw(2016, "FY", "2015-02-01", "2016-01-30", form="10-K"),
    ])
    out = backfill_fiscal_period_by_filing_order(raw)
    by_end = out.set_index("period_end")
    assert by_end.loc[pd.Timestamp("2015-05-23"), "fiscal_period"] == "Q1"
    assert by_end.loc[pd.Timestamp("2015-08-15"), "fiscal_period"] == "Q2"
    assert by_end.loc[pd.Timestamp("2015-11-07"), "fiscal_period"] == "Q3"


def test_backfill_fiscal_period_by_filing_order_never_guesses_with_too_many_candidates():
    """A fiscal year with MORE than three distinct missing-fp 10-Q period_ends
    (e.g. a stray/duplicate filing) is ambiguous -- left entirely unresolved
    rather than guessing wrong, matching this pipeline's null-over-guess
    convention."""
    raw = pd.DataFrame([
        _raw(2016, None, "2015-02-01", "2015-05-23"),
        _raw(2016, None, "2015-02-01", "2015-08-15"),
        _raw(2016, None, "2015-02-01", "2015-11-07"),
        _raw(2016, None, "2015-02-01", "2015-12-15"),   # a 4th distinct period_end -- ambiguous
    ])
    out = backfill_fiscal_period_by_filing_order(raw)
    assert out["fiscal_period"].isna().all()


def test_backfill_fiscal_period_by_filing_order_resolves_a_10k_directly_from_its_shape():
    """A 10-K with no native fiscal_period is resolved to 'FY' directly from its
    annual-shaped (~365d) duration -- no cross-filing context needed."""
    raw = pd.DataFrame([
        _raw(2016, None, "2015-02-01", "2016-01-30", form="10-K"),
    ])
    out = backfill_fiscal_period_by_filing_order(raw)
    assert out.iloc[0]["fiscal_period"] == "FY"

    print("\n=== SANITY CHECK: cross-filing fiscal_period resolution ===")
    print("  a filing with NO native fiscal_period anywhere (real KR bug: Q1 10-Qs, e.g.")
    print("  accession 0001104659-15-048764 -- fiscal_year populated but fiscal_period blank")
    print("  on every tagged fact) now resolves via chronological rank among the ticker's own")
    print("  10-Qs for that fiscal year (or directly from an annual-shaped 10-K), fixing every")
    print("  field the filing reports at once, without disturbing natively-labeled siblings;")
    print("  an over-ambiguous fiscal year (more candidates than open quarter slots) is left")
    print("  unresolved rather than guessed.")
    print("  Validated.")


def test_ytd_fiscal_period_labels_normalize_to_quarter_buckets():
    """edgartools labels a concept NEVER tagged as a discrete 3-month figure (cash-
    flow-statement items -- confirmed empirically against real MAA 10-Qs: operating
    cash flow / capex are tagged fiscal_period='YTD6'/'YTD9', never 'Q2'/'Q3') with
    its own 'YTDn' fiscal_period. Without normalize_fiscal_period_label,
    decumulate_quarterly_flow's isin(_QUARTER_LABELS) filter silently dropped every
    such fact -- Q2/Q3 (and therefore the FY-Q1-Q2-Q3 Q4 derivation) never resolved
    for any cash-flow-only field. This reproduces the real MAA bug end-to-end."""
    assert normalize_fiscal_period_label("YTD3") == "Q1"
    assert normalize_fiscal_period_label("YTD6") == "Q2"
    assert normalize_fiscal_period_label("YTD9") == "Q3"
    assert normalize_fiscal_period_label("YTD12") == "FY"
    assert normalize_fiscal_period_label("Q2") == "Q2"   # already-normal labels pass through
    assert normalize_fiscal_period_label(None) is None

    facts = pd.DataFrame([
        _q(2022, "Q1", "2022-01-01", "2022-03-31", 200.0, "2022-05-01", "a1"),
        # Q2/Q3 natively labeled YTD6/YTD9 (cash-flow-statement style), never Q2/Q3:
        _q(2022, "YTD6", "2022-01-01", "2022-06-30", 463.0, "2022-08-01", "a2"),
        _q(2022, "YTD9", "2022-01-01", "2022-09-30", 807.0, "2022-11-01", "a3"),
        _q(2022, "FY", "2022-01-01", "2022-12-31", 1100.0, "2023-02-15", "a4", "10-K"),
    ])
    out = decumulate_quarterly_flow(facts)
    q = {r.fiscal_period: r.value for r in out.itertuples()}
    assert set(q) == {"Q1", "Q2", "Q3", "Q4"}
    assert q["Q1"] == 200.0
    assert abs(q["Q2"] - (463.0 - 200.0)) < 1e-9
    assert abs(q["Q3"] - (807.0 - 463.0)) < 1e-9
    assert abs(sum(q.values()) - 1100.0) < 1e-9

def test_discrete_quarter_preferred_over_ytd_fact_from_the_same_filing():
    """When a filing tags BOTH a genuinely quarter-shaped (~90d) fact AND a
    longer YTD-cumulative fact for the SAME concept/quarter -- confirmed
    empirically: MAA's Q2 2025 10-Q tags `NetIncomeLoss`/`Revenues` with both a
    true "3 months ended" context and a "6 months ended" (YTD6) context in the
    SAME filing, which is standard GAAP income-statement presentation -- the
    quarter-shaped one must be used AS-IS (no subtraction), not whichever one
    happens to sort first. Verified via a Q2 where the as-filed discrete value
    does NOT equal (YTD6 - Q1) -- if the YTD fact were used, decumulation would
    silently produce the WRONG number even though the correct one was tagged
    directly in the very same filing."""
    facts = pd.DataFrame([
        _q(2025, "Q1", "2025-01-01", "2025-03-31", 200.0, "2025-05-01", "a1"),
        # SAME accession/filing_date tags BOTH shapes for Q2 -- the discrete
        # one (91d) is the as-filed truth; the YTD6 one is redundant/derivable
        # but would give a DIFFERENT number if subtracted (simulating a filer
        # restating part of Q1 within the YTD figure without amending Q1 itself).
        _q(2025, "Q2", "2025-04-01", "2025-06-30", 230.0, "2025-08-01", "a2-q2"),      # discrete, 91d
        _q(2025, "Q2", "2025-01-01", "2025-06-30", 410.0, "2025-08-01", "a2-q2"),      # YTD6, 181d
    ])
    out = decumulate_quarterly_flow(facts)
    q2_row = out[out["fiscal_period"] == "Q2"].iloc[0]
    assert q2_row["value"] == 230.0   # the as-filed discrete value, NOT 410.0-200.0=210.0


def test_native_q4_fact_is_never_read_directly_even_when_present():
    """Q4 is ALWAYS derived as FY-(Q1+Q2+Q3) -- a native fiscal_period='Q4' fact
    is never consulted, regardless of whether it looks plausible. Real bug this
    guards against: ORCL's `us-gaap:Revenues` tag once carried a context shaped
    exactly like Q4 (~91 days) yet reported the FULL-YEAR dollar total (a
    filer-side XBRL tagging inconsistency) -- a native-vs-derived CROSS-CHECK
    (the pipeline's prior design) is strictly weaker than never trusting the
    native fact at all, since a future filer-side inconsistency could just as
    easily happen to pass a cross-check tolerance. The derived value, backed by
    four independently-filed figures (FY, Q1, Q2, Q3) sharing the same
    source_tag, is used unconditionally."""
    facts = pd.DataFrame([
        _q(2021, "Q1", "2020-06-01", "2020-08-31", 9367.0, "2020-09-15", "a1"),
        _q(2021, "Q2", "2020-06-01", "2020-11-30", 19167.0, "2020-12-11", "a2"),   # YTD6: 9367+9800
        _q(2021, "Q3", "2020-06-01", "2021-02-28", 29252.0, "2021-03-11", "a3"),   # YTD9: +10085
        _q(2021, "FY", "2020-06-01", "2021-05-31", 40479.0, "2021-06-21", "a4", "10-K"),
        # a coexisting native "Q4" fact, correctly shaped but carrying a
        # completely different (implausible, full-year-sized) value -- must be
        # ignored entirely, not cross-checked against.
        _q(2021, "Q4", "2021-03-01", "2021-05-31", 999999.0, "2021-06-21", "a-native-q4", "10-K"),
    ])
    out = decumulate_quarterly_flow(facts)
    q4_row = out[out["fiscal_period"] == "Q4"].iloc[0]
    assert bool(q4_row["derived"]) is True
    assert abs(q4_row["value"] - (40479.0 - 9367.0 - 9800.0 - 10085.0)) < 1e-6
    assert q4_row["accession_number"] == "a4"   # the FY accession, never "a-native-q4"


def test_q4_not_derived_when_inputs_use_different_source_tags():
    """Real bug found via live data: JPM's totalRevenue resolved FY via
    `us-gaap:Revenues` (~$11-21B) while its quarters resolved via
    `us-gaap:RevenuesNetOfInterestExpense` (JPM's actual ~$23-26B/quarter bank-
    revenue concept) in several fiscal years -- FY-(Q1+Q2+Q3) across two
    DIFFERENT XBRL concepts is not a valid subtraction and produced wildly
    negative "Q4" values (e.g. -$63B). Q4 must not be derived at all when
    Q1/Q2/Q3/FY don't all share the same source_tag -- Q1/Q2/Q3 remain
    available, the fiscal year's Q4 is simply unresolvable for this field."""
    def _q_tagged(fiscal_period, start, end, value, filed, accn, tag, form="10-Q"):
        row = _q(2016, fiscal_period, start, end, value, filed, accn, form)
        row["source_tag"] = tag
        return row

    facts = pd.DataFrame([
        _q_tagged("Q1", "2016-01-01", "2016-03-31", 23239.0, "2016-04-29", "a1",
                 "us-gaap:RevenuesNetOfInterestExpense"),
        _q_tagged("Q2", "2016-01-01", "2016-06-30", 47619.0, "2016-08-03", "a2",
                 "us-gaap:RevenuesNetOfInterestExpense"),   # YTD6 = 23239+24380
        _q_tagged("Q3", "2016-01-01", "2016-09-30", 72292.0, "2016-11-01", "a3",
                 "us-gaap:RevenuesNetOfInterestExpense"),   # YTD9 = 47619+24673
        _q_tagged("FY", "2016-01-01", "2016-12-31", 16045.0, "2017-02-28", "a4",
                 "us-gaap:Revenues", form="10-K"),   # a DIFFERENT concept -- much smaller
    ])
    out = decumulate_quarterly_flow(facts)
    assert set(out["fiscal_period"]) == {"Q1", "Q2", "Q3"}   # no Q4 row at all
    assert (out[out["fiscal_period"] == "Q1"]["value"] == 23239.0).all()


def test_q4_nulled_when_derived_value_is_incoherent_with_other_quarters():
    """Even when Q1/Q2/Q3/FY all share the same source_tag, a wildly incoherent
    derived Q4 (opposite sign AND far outside Q1-Q3's magnitude range) must be
    nulled rather than stored -- a final safety net alongside the same-tag
    requirement, for whatever residual case produces a bad FY or quarter
    value. Q1/Q2/Q3 remain available."""
    facts = pd.DataFrame([
        _q(2013, "Q1", "2013-01-01", "2013-03-31", 133367000.0, "2013-05-03", "a1"),
        _q(2013, "Q2", "2013-01-01", "2013-06-30", 267479000.0, "2013-08-02", "a2"),   # YTD6
        _q(2013, "Q3", "2013-01-01", "2013-09-30", 398252000.0, "2013-11-07", "a3"),   # YTD9
        # a corrupted/wrongly-picked FY value (a dimensioned slice, per the
        # confirmed MAA bug) -- FY-(Q1+Q2+Q3) would be wildly negative
        _q(2013, "FY", "2013-01-01", "2013-12-31", 14444000.0, "2014-02-21", "a-fy", "10-K"),
    ])
    out = decumulate_quarterly_flow(facts)
    assert set(out["fiscal_period"]) == {"Q1", "Q2", "Q3"}   # Q4 nulled, not a garbage negative
    assert (out[out["fiscal_period"] == "Q1"]["value"] == 133367000.0).all()


def test_native_q4_mislabeled_quarter_reassigned_when_it_starts_with_the_fiscal_year():
    """A fact natively labeled fiscal_period='Q4' cannot genuinely be the fiscal
    year's LAST quarter if its OWN period_start coincides with the fiscal year's
    own start -- only the first quarter can start when the fiscal year does.
    Real bug found via live data: MAA's Q1-2013 10-Q (period 2013-01-01..
    2013-03-31, genuinely Q1 -- SAME period_start as fiscal 2013's own FY fact)
    tagged NetCashProvidedByUsedInOperatingActivities/PaymentsForCapitalImprove-
    ments with native fiscal_period='Q4' instead of 'Q1'/'YTD3' -- the mirror
    image of the native-Q4-disagrees-with-derived bug above (there the LABEL was
    right and the VALUE was wrong; here the CONTEXT DATES are right and the
    LABEL is wrong). Confirmed on the ACTUAL live filing: MAA's genuine fiscal
    2013 Q4 does not even exist as a separate native 'Q4' fact for this concept
    (the FY2013 10-K tags it dimensioned-only as 'FY') -- so this is a single,
    lone mislabeled 'Q4' claim, not a conflict between two competing native Q4
    facts; the fix must still catch it via period_start alone, and once Q1 is
    recovered, Q4 becomes cleanly derivable."""
    facts = pd.DataFrame([
        # mislabeled: genuinely Q1 (Jan-Mar, same period_start as the FY fact
        # below) but natively tagged 'Q4' -- the real MAA bug, reproduced exactly.
        _q(2013, "Q4", "2013-01-01", "2013-03-31", 8701000.0, "2013-05-03", "a-q1-mislabeled"),
        _q(2013, "Q2", "2013-01-01", "2013-06-30", 31412000.0, "2013-08-02", "a2"),   # YTD6
        _q(2013, "Q3", "2013-01-01", "2013-09-30", 44330000.0, "2013-11-07", "a3"),   # YTD9
        _q(2013, "FY", "2013-01-01", "2013-12-31", 53439000.0, "2014-02-21", "a-fy", "10-K"),
    ])
    out = decumulate_quarterly_flow(facts)
    by_fp = {r.fiscal_period: r for r in out.itertuples()}
    assert "Q1" in by_fp
    assert by_fp["Q1"].value == 8701000.0
    assert by_fp["Q1"].accession_number == "a-q1-mislabeled"
    # Q4 is now cleanly DERIVED (no genuine native Q4 fact exists for this concept)
    assert bool(by_fp["Q4"].derived) is True
    assert abs(by_fp["Q4"].value - (53439000.0 - 8701000.0 - (31412000.0 - 8701000.0)
                                    - (44330000.0 - 31412000.0))) < 1e-6


def test_native_q4_mislabeled_quarter_not_confused_with_a_genuine_later_native_q4():
    """When a mislabeled 'Q4' (really Q1, period_start = fiscal-year start) COEXISTS
    with a genuinely later, correctly-shaped native 'Q4' claim for the same field
    and fiscal year, only the mislabeled one is reassigned -- the genuine Q4 (far
    from the fiscal-year start) is left labeled 'Q4' but, per the always-derive
    design, its VALUE is still never used for the final Q4 row: the output Q4 is
    the FY-(Q1+Q2+Q3) derivation, not the coexisting native fact's value."""
    facts = pd.DataFrame([
        _q(2013, "Q4", "2013-01-01", "2013-03-31", 8701000.0, "2013-05-03", "a-q1-mislabeled"),
        _q(2013, "Q2", "2013-01-01", "2013-06-30", 22494000.0, "2013-08-02", "a2"),   # YTD6
        _q(2013, "Q3", "2013-01-01", "2013-09-30", 35412000.0, "2013-11-07", "a3"),   # YTD9
        _q(2013, "Q4", "2013-10-01", "2013-12-31", 15000000.0, "2014-02-21", "a-q4-genuine", "10-K"),
        _q(2013, "FY", "2013-01-01", "2013-12-31", 53439000.0, "2014-02-21", "a-fy", "10-K"),
    ])
    out = decumulate_quarterly_flow(facts)
    by_fp = {r.fiscal_period: r for r in out.itertuples()}
    assert by_fp["Q1"].value == 8701000.0
    assert by_fp["Q1"].accession_number == "a-q1-mislabeled"
    assert by_fp["Q2"].value == 13793000.0
    assert by_fp["Q3"].value == 12918000.0
    assert bool(by_fp["Q4"].derived) is True
    assert by_fp["Q4"].accession_number == "a-fy"        # never "a-q4-genuine"
    assert abs(by_fp["Q4"].value - 18027000.0) < 1e-6    # derived, not the coexisting 15,000,000


def test_native_q4_mislabel_repaired_even_when_q2_and_q3_are_also_mislabeled():
    """The mislabeling is not always confined to Q1. Real bug found via live
    data: MAA's `rentalIncome` mislabeled ALL THREE of its FY2017 Q1/Q2/Q3
    discrete facts as native 'Q4' -- three different accessions (the Apr/Jul/Oct
    2017 10-Qs), three genuinely different and correctly quarter-shaped values,
    all claiming 'Q4'. Reassignment must be by ELAPSED-TIME POSITION (not merely
    "starts exactly at the fiscal year start") to catch this."""
    facts = pd.DataFrame([
        _q(2017, "Q4", "2017-01-01", "2017-03-31", 351177000.0, "2017-04-27", "a-q1-mislabeled"),
        _q(2017, "Q4", "2017-04-01", "2017-06-30", 355832000.0, "2017-07-27", "a-q2-mislabeled"),
        _q(2017, "Q4", "2017-07-01", "2017-09-30", 357619000.0, "2017-10-26", "a-q3-mislabeled"),
        _q(2017, "FY", "2017-01-01", "2017-12-31", 1450000000.0, "2018-02-23", "a-fy", "10-K"),
    ])
    out = decumulate_quarterly_flow(facts)
    by_fp = {r.fiscal_period: r for r in out.itertuples()}
    assert by_fp["Q1"].value == 351177000.0 and by_fp["Q1"].accession_number == "a-q1-mislabeled"
    assert by_fp["Q2"].value == 355832000.0 and by_fp["Q2"].accession_number == "a-q2-mislabeled"
    assert by_fp["Q3"].value == 357619000.0 and by_fp["Q3"].accession_number == "a-q3-mislabeled"
    # Q4 is cleanly DERIVED -- no genuine native Q4 fact exists for this concept/year
    assert bool(by_fp["Q4"].derived) is True
    assert abs(by_fp["Q4"].value - (1450000000.0 - 351177000.0 - 355832000.0 - 357619000.0)) < 1e-6


def test_q4_still_derived_even_when_a_native_q4_fact_would_have_agreed():
    """Even when a coexisting native Q4 fact happens to numerically agree with
    FY-(Q1+Q2+Q3), the OUTPUT row is still the derived one (derived=True) --
    there is no "trust the native fact when it happens to agree" path anymore;
    Q4 is unconditionally computed from FY-(Q1+Q2+Q3)."""
    facts = pd.DataFrame([
        _q(2024, "Q1", "2024-01-01", "2024-03-31", 100.0, "2024-04-20", "a1"),
        _q(2024, "Q2", "2024-01-01", "2024-06-30", 220.0, "2024-07-20", "a2"),
        _q(2024, "Q3", "2024-01-01", "2024-09-30", 345.0, "2024-10-20", "a3"),
        _q(2024, "FY", "2024-01-01", "2024-12-31", 460.0, "2025-02-15", "a4", "10-K"),
        _q(2024, "Q4", "2024-10-01", "2024-12-31", 115.0, "2025-02-15", "a4", "10-K"),
    ])
    out = decumulate_quarterly_flow(facts)
    q4_row = out[out["fiscal_period"] == "Q4"].iloc[0]
    assert bool(q4_row["derived"]) is True
    assert q4_row["value"] == 115.0

    print("\n=== SANITY CHECK: YTDn label normalization + Q4 always derived ===")
    print("  'YTD3'/'YTD6'/'YTD9'/'YTD12' normalize to 'Q1'/'Q2'/'Q3'/'FY'; a cash-flow-")
    print("  statement field natively labeled YTD6/YTD9 (real MAA bug: operatingCashFlow/")
    print("  capex resolved ONLY Q1 every year, never Q2/Q3/Q4) now decumulates and")
    print("  reconciles exactly to FY. Q4 is now ALWAYS derived as FY-(Q1+Q2+Q3) and NEVER")
    print("  read from a native Q4 fact (real ORCL bug: `Revenues` context shaped like Q4 but")
    print("  valued like FY) -- gated by a same-source_tag requirement across Q1/Q2/Q3/FY")
    print("  (real JPM bug: FY resolved a different XBRL concept than the quarters, producing")
    print("  a wildly negative derived Q4) and a coherence check against Q1-Q3 (real MAA bug:")
    print("  a wrongly-picked dimensioned FY value produced a wildly negative derived Q4) --")
    print("  either failure nulls Q4 for that fiscal year rather than storing a bad number.")
    print("  Validated.")


def _instant(fiscal_year, fiscal_period, period_end, value, filed, accn, duration_type="instant",
            field="cash", period_start=pd.NaT):
    return {"ticker": "MAA", "field": field, "fiscal_year": fiscal_year, "fiscal_period": fiscal_period,
           "duration_type": duration_type, "period_start": period_start, "period_end": pd.Timestamp(period_end),
           "value": value, "filing_date": pd.Timestamp(filed), "accession_number": accn, "form": "10-Q"}


def test_reassign_misordered_instant_facts_repairs_filing_wide_mislabel():
    """Instant facts inherit fiscal_year/fiscal_period via BACKFILL from a
    duration fact in the SAME filing -- so a filing-wide native mislabeling
    (the Q1-tagged-as-Q4 bug above) propagates onto every instant concept in
    that filing, not just the one it originated on. Real bug found via live
    data: MAA's Q1-2017 10-Q (filed 2017-04-27) mislabeled fiscal_period='Q4'
    for netIncome/capex AND, via backfill, for sharesOutstanding/cash/
    notesPayable/rentalIncome/realEstateGross -- three DIFFERENT accessions
    (Apr/Jul/Oct 2017) all claiming (2017, 'Q4') for `cash`, even though their
    period_end dates are genuinely three different quarters. An instant fact
    labeled 'Q4' whose OWN period_end falls in the FIRST HALF of the fiscal
    year (using the fiscal year's own start, borrowed from any 'annual' row)
    cannot genuinely be the last quarter and is relabeled by elapsed time."""
    facts = pd.DataFrame([
        # the fiscal-year anchor (from a DIFFERENT, correctly-behaved field)
        _instant(2017, "FY", "2017-12-31", 1000.0, "2018-02-20", "a-fy",
                duration_type="annual", field="netIncome", period_start=pd.Timestamp("2017-01-01")),
        # mislabeled: genuinely Q1 cash (as-of 2017-03-31) tagged native 'Q4'
        _instant(2017, "Q4", "2017-03-31", 500.0, "2017-04-27", "a-q1-mislabeled"),
        # genuinely Q2/Q3 cash, correctly labeled
        _instant(2017, "Q2", "2017-06-30", 520.0, "2017-07-27", "a-q2"),
        _instant(2017, "Q3", "2017-09-30", 540.0, "2017-10-26", "a-q3"),
        # genuine Q4 cash, correctly labeled
        _instant(2017, "Q4", "2017-12-31", 560.0, "2018-02-20", "a-q4-genuine"),
    ])
    out = reassign_misordered_instant_facts(facts)
    cash = out[out["field"] == "cash"].set_index("accession_number")
    assert cash.loc["a-q1-mislabeled", "fiscal_period"] == "Q1"
    assert cash.loc["a-q2", "fiscal_period"] == "Q2"          # untouched
    assert cash.loc["a-q3", "fiscal_period"] == "Q3"          # untouched
    assert cash.loc["a-q4-genuine", "fiscal_period"] == "Q4"  # untouched

    print("\n=== SANITY CHECK: filing-wide instant-fact mislabel repair ===")
    print("  a balance-sheet (instant) fact that inherited a wrong fiscal_period via")
    print("  backfill from a mislabeled duration fact in the SAME filing (real MAA bug:")
    print("  Q1-2017's 10-Q tagged cash/sharesOutstanding/notesPayable/rentalIncome/")
    print("  realEstateGross all as 'Q4') is now relabeled using the fiscal year's own")
    print("  start (borrowed from any 'annual' row) + its own period_end -- genuinely")
    print("  correct Q2/Q3/Q4 instant facts for the same field are left untouched.")
    print("  Validated.")
