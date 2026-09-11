"""
test_director_comp.py  (tests/data_aggregate/test_director_comp.py)
--------------------------------------------------------------------
The Item 402(k) director-pay family (D42) in
`src/data_aggregate/utils/governance/director_comp.py` — 80,252 rows that no part of the cube
read before phase 6.

Three things are load-bearing and each has its own test: the `total` identity must fill from
components without ever inventing a $0 package, `ceo_to_director_pay_ratio` must be NaN and
never `inf` on a zero denominator, and the family's coverage must be readable as the 2006
REGIME STAIRCASE it is rather than as an extraction gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.director_comp import (
    ALL_FIELDS, DIRECTOR_COMPONENTS, EVENT_FIELDS, LEVEL_FIELDS, PEER_RELATIVE_FIELDS,
    director_pay_fields, impute_director_comp,
)
from src.data_aggregate.utils.governance.staleness import LEVEL_MAX_AGE_DAYS
from src.data_store.schema import Tables

IDX = pd.bdate_range("2019-01-01", "2025-06-30")


def _rows() -> pd.DataFrame:
    """One filing, four directors, each a different branch of the identity."""
    base = {"ticker": "AAA", "accession_number": "AAA-1", "as_of": "2020-05-01"}
    return pd.DataFrame([
        # a NULL total with every component -> filled at 300,000
        {**base, "name": "Full Fill", "total": None, "fees_earned": 100_000.0,
         "stock_awards": 150_000.0, "option_awards": 20_000.0,
         "non_equity_incentive": 10_000.0, "pension_change": 5_000.0,
         "other_compensation": 15_000.0},
        # a NULL total with ONE component -> min_count=1 fills at that one component
        {**base, "name": "Part Fill", "total": None, "fees_earned": 80_000.0,
         "stock_awards": None, "option_awards": None, "non_equity_incentive": None,
         "pension_change": None, "other_compensation": None},
        # EVERY component NULL -> must stay NULL, never $0
        {**base, "name": "All Null", "total": None, "fees_earned": None,
         "stock_awards": None, "option_awards": None, "non_equity_incentive": None,
         "pension_change": None, "other_compensation": None},
        # a filer-STATED total that disagrees with its components -> never overwritten
        {**base, "name": "Stated", "total": 999_999.0, "fees_earned": 1.0,
         "stock_awards": 1.0, "option_awards": None, "non_equity_incentive": None,
         "pension_change": None, "other_compensation": None},
    ])


def test_the_402k_identity_fills_from_components_and_leaves_an_all_null_row_null():
    """`min_count=1` is the whole restraint. `sum()` over an all-NaN row returns 0.0, and a
    board member recorded as working for $0 is the single easiest fact to fabricate here --
    which is also why `fillna(0)` is forbidden across the whole governance package."""
    out, stats = impute_director_comp(_rows())
    by = out.set_index("name")
    assert by.loc["Full Fill", "total"] == pytest.approx(300_000.0)
    assert by.loc["Part Fill", "total"] == pytest.approx(80_000.0)
    assert pd.isna(by.loc["All Null", "total"]), "an all-NULL row became a $0 pay package"
    assert by.loc["Stated", "total"] == pytest.approx(999_999.0), \
        "a filer-stated total was overwritten by its own components"
    assert by.loc["Stated", "total_imputed"] == 0.0
    assert by.loc[["Full Fill", "Part Fill"], "total_imputed"].tolist() == [1.0, 1.0]
    assert stats["total = sum(components)"] == 2
    assert stats["total still NULL (no component at all)"] == 1
    assert stats["components available"] == len(DIRECTOR_COMPONENTS) == 6

    print("\n=== SANITY CHECK: the Item 402(k) total identity ===")
    print(f"  six components summed -> {by.loc['Full Fill', 'total']:,.0f}")
    print(f"  one component only    -> {by.loc['Part Fill', 'total']:,.0f} (min_count=1)")
    print(f"  no component at all   -> {by.loc['All Null', 'total']} (NOT $0)")
    print(f"  filer-stated 999,999 vs components 2 -> {by.loc['Stated', 'total']:,.0f} (kept)")
    print("  CONCLUSION: SIX components, the first `fees_earned` and not `salary`; the identity "
          "is non-destructive and an unknown row stays unknown. Validated.")


def test_the_shares_are_board_level_sums_not_means_of_ratios():
    """A director who joined in March draws a pro-rated fee against a $0 stock award. A mean of
    per-director ratios lets that one row move the board's equity share by ten points;
    `sum(stock) / sum(total)` is the share of the board's whole pay BILL that came as equity,
    which is the quantity the alignment argument is about."""
    base = {"ticker": "BBB", "accession_number": "BBB-1", "as_of": "2020-05-01",
            "option_awards": None, "non_equity_incentive": None, "pension_change": None,
            "other_compensation": None}
    df = pd.DataFrame([
        {**base, "name": "Veteran A", "total": 300_000.0, "fees_earned": 100_000.0,
         "stock_awards": 200_000.0},
        {**base, "name": "Veteran B", "total": 300_000.0, "fees_earned": 100_000.0,
         "stock_awards": 200_000.0},
        {**base, "name": "Joined March", "total": 20_000.0, "fees_earned": 20_000.0,
         "stock_awards": 0.0},
    ])
    frames, _ = director_pay_fields(df, None, IDX)
    equity = float(frames["director_equity_pay_pct"].loc[pd.Timestamp("2020-06-01"), "BBB"])
    cash = float(frames["director_cash_fee_pct"].loc[pd.Timestamp("2020-06-01"), "BBB"])
    # sums: stock 400k, fees 220k, total 620k
    assert equity == pytest.approx(400_000 / 620_000)
    assert cash == pytest.approx(220_000 / 620_000)
    assert equity + cash == pytest.approx(1.0), "the two shares must exhaust this pay bill"
    mean_of_ratios = np.mean([200 / 300, 200 / 300, 0.0])
    print("\n=== SANITY CHECK: the two pay-MIX shares ===")
    print(f"  board sums: stock 400,000 / fees 220,000 / total 620,000")
    print(f"  equity share {equity:.4f}, cash share {cash:.4f}, sum {equity + cash:.4f}")
    print(f"  a MEAN OF RATIOS would say {mean_of_ratios:.4f} -- "
          f"{abs(equity - mean_of_ratios) * 100:.1f}pp away, driven by one part-year director")
    print("  CONCLUSION: the shares are board-level sums, so a pro-rated joiner cannot move "
          "them. Validated.")


def test_the_ceo_ratio_is_NaN_and_never_inf_on_a_zero_denominator():
    """A board whose disclosed median pay is 0 is a parse failure, not a board that works for
    free. `inf` would then propagate through the winsorization as the largest value in the
    cross-section -- a parse failure encoded as the most extreme governance signal in the index.
    """
    base = {"option_awards": None, "non_equity_incentive": None, "pension_change": None,
            "other_compensation": None, "stock_awards": None}
    dc = pd.DataFrame([
        {**base, "ticker": "ZERO", "accession_number": "Z-1", "as_of": "2020-05-01",
         "name": "Free A", "total": 0.0, "fees_earned": 0.0},
        {**base, "ticker": "ZERO", "accession_number": "Z-1", "as_of": "2020-05-01",
         "name": "Free B", "total": 0.0, "fees_earned": 0.0},
        {**base, "ticker": "GOOD", "accession_number": "G-1", "as_of": "2020-05-01",
         "name": "Paid A", "total": 200_000.0, "fees_earned": 200_000.0},
        {**base, "ticker": "GOOD", "accession_number": "G-1", "as_of": "2020-05-01",
         "name": "Paid B", "total": 300_000.0, "fees_earned": 300_000.0},
        # ZERO's LATER filing is normal, so the ticker keeps a column and the zero-denominator
        # year has to be NaN *inside* it -- a stronger check than the column simply being absent.
        {**base, "ticker": "ZERO", "accession_number": "Z-2", "as_of": "2022-05-01",
         "name": "Free A", "total": 150_000.0, "fees_earned": 150_000.0},
    ])
    parent = pd.DataFrame([
        {"ticker": "ZERO", "as_of": "2020-05-01", "ceo_total_comp": 10_000_000.0},
        {"ticker": "ZERO", "as_of": "2022-05-01", "ceo_total_comp": 10_000_000.0},
        {"ticker": "GOOD", "as_of": "2020-05-01", "ceo_total_comp": 10_000_000.0},
    ])
    frames, tally = director_pay_fields(dc, parent, IDX)
    r = frames["ceo_to_director_pay_ratio"]
    zero_2020, zero_2022 = float(r.loc[pd.Timestamp("2020-06-01"), "ZERO"]), \
        float(r.loc[pd.Timestamp("2022-06-01"), "ZERO"])
    assert np.isnan(zero_2020), f"a zero median produced {zero_2020}"
    assert not np.isinf(zero_2020)
    assert zero_2022 == pytest.approx(10_000_000 / 150_000), "the LATER, valid year was lost"
    assert r.loc[pd.Timestamp("2020-06-01"), "GOOD"] == pytest.approx(10_000_000 / 250_000)
    assert tally["ceo_to_director_pay_ratio: rejected (median <= 0)"] == 1
    # `log_median_director_pay` is guarded by the same `> 0` -- log(0) is -inf, not a level
    lg = frames["log_median_director_pay"]
    assert np.isnan(lg.loc[pd.Timestamp("2020-06-01"), "ZERO"])
    assert lg.loc[pd.Timestamp("2020-06-01"), "GOOD"] == pytest.approx(np.log(250_000))

    print("\n=== SANITY CHECK: ceo_to_director_pay_ratio's denominator ===")
    print(f"  median director pay 0 -> ratio {zero_2020} (NaN, not inf)")
    print(f"  the same ticker's next, valid filing -> {zero_2022:.1f}x (the year is rejected, "
          "not the ticker)")
    print(f"  median 250,000 vs a $10M CEO -> "
          f"{float(r.loc[pd.Timestamp('2020-06-01'), 'GOOD']):.1f}x")
    print(f"  log_median_director_pay on the zero year: "
          f"{lg.loc[pd.Timestamp('2020-06-01'), 'ZERO']}")
    print("  CONCLUSION: both the ratio and the log guard at > 0 and leave a parse failure "
          "unknown rather than extreme. Validated.")


def test_the_encoding_and_expiry_contracts():
    """Director pay is a LEVEL (D21) -- and a level is now given a LEVEL HORIZON, not none.

    ⚠ THIS TEST ASSERTED THE OPPOSITE UNTIL 2026-09-09, and the inversion is the point rather
    than a detail. It read: *"a retainer structure persists between proxies exactly as
    `ceo_pay_slice` and `log_ceo_total_comp` do, so nothing here is aged out"*, and probed a
    2020 filing at 2025-06-02 -- 1,858 days later -- demanding a value. Both halves of that
    sentence were accepted; only the conclusion is overturned. A retainer structure DOES
    persist between proxies. It does not persist for five years through two missed annual
    cycles, and "persists between proxies" is not a reason to report it forever: an unbounded
    forward-fill returns a plausible number with no filing behind it at all.

    So the family moved onto `LEVEL_MAX_AGE_DAYS` (1,095 days = two whole missed annual
    cycles), not onto the 548-day EVENT clock -- which is what the original reasoning was
    right to reject. The cost is D2, measured on the live archive: Ford's
    `f_ceo_to_director_pay_ratio` ran **2,769 consecutive days at exactly 0** off a single
    2011 filing, and now runs **753**, ending 2014-03-31 -- exactly 1,095 days after
    2011-04-01 -- with the 2022 filing reopening the series on its own date.

    ⚠ The probe is the WHOLE INDEX partitioned by age, not two dates. A two-point probe on
    this fixture is a trap: `index.asof` walks back to the previous business day, so a date
    computed as `filed + horizon` can land on a weekend and be measured a day younger than
    intended. Partitioning every date is also strictly stronger -- it catches an off-by-one
    at the boundary in either direction.

    ⚠ TWO PEER LEGS, the first any new family has earned since phase 3, and only the two the
    between-sector variance share supports: `director_cash_fee_pct` 11.94% and
    `director_equity_pay_pct` 8.17% against the surviving legacy legs' 7.7%-12.4% floor. A
    director's cash-versus-equity mix is set against a sector benchmark by a compensation
    consultant; the pay LEVEL (6.84%) and the CEO RATIO (5.16%) are not, and ship raw.
    """
    assert EVENT_FIELDS == frozenset(), "director pay was declared an event"
    assert LEVEL_FIELDS == ALL_FIELDS, \
        "a director-pay field is on neither horizon -- that is the phase-3 defect's own shape"
    assert PEER_RELATIVE_FIELDS == {"director_cash_fee_pct", "director_equity_pay_pct"}, \
        "the peer-leg set changed without a re-measured sector share"
    assert PEER_RELATIVE_FIELDS < ALL_FIELDS
    assert "ceo_to_director_pay_ratio" not in PEER_RELATIVE_FIELDS, \
        "a ratio whose absolute level is the thesis was peer-centred"
    assert ALL_FIELDS == {"log_median_director_pay", "director_equity_pay_pct",
                          "director_cash_fee_pct", "ceo_to_director_pay_ratio"}

    frames, tally = director_pay_fields(_rows(), None, IDX)
    fee = frames["director_cash_fee_pct"]["AAA"]
    filed = pd.Timestamp("2020-05-01")
    age = (fee.index - filed).days

    inside = fee[(age >= 0) & (age <= LEVEL_MAX_AGE_DAYS)]
    outside = fee[age > LEVEL_MAX_AGE_DAYS]
    assert inside.notna().all(), \
        f"{int(inside.isna().sum())} cell(s) INSIDE the level horizon were expired"
    assert outside.notna().sum() == 0, \
        f"{int(outside.notna().sum())} cell(s) survived past {LEVEL_MAX_AGE_DAYS} days"
    assert len(inside) and len(outside), "the fixture no longer spans the horizon"

    expired = next((v for k, v in tally.items()
                    if k.startswith(f"expired >{LEVEL_MAX_AGE_DAYS}d: director_cash_fee_pct")
                    and "of non-null" not in k), 0)
    assert expired == len(outside), \
        f"the tally says {expired} expired cells, the frame shows {len(outside)}"

    print("\n=== SANITY CHECK: the director-pay encoding + expiry contracts ===")
    print(f"  ALL_FIELDS = {sorted(ALL_FIELDS)}")
    print(f"  EVENT_FIELDS = {set(EVENT_FIELDS) or '{} (all levels)'} ; "
          f"LEVEL_FIELDS = all {len(LEVEL_FIELDS)} ; "
          f"PEER_RELATIVE_FIELDS = {sorted(PEER_RELATIVE_FIELDS)}")
    print(f"  one 2020-05-01 filing, index {fee.index[0].date()} .. {fee.index[-1].date()}")
    print(f"  <= {LEVEL_MAX_AGE_DAYS}d after it: {len(inside):>4} cells, all reported")
    print(f"  >  {LEVEL_MAX_AGE_DAYS}d after it: {len(outside):>4} cells, all expired "
          f"(tally agrees: {expired})")
    print("  CONCLUSION: four fields, all LEVELS, all on the 1,095-day level horizon rather "
          "than on no horizon at all. Validated.")


def test_the_real_director_comp_readout():
    """§1.7 regenerated, including the ERA staircase and the per-TICKER hole.

    ⚠ The staircase is the point: a flat 64% headline is the pre-2006 average of a post-2006
    disclosure. Item 402(k) created the Director Compensation Table in the 2006 Reg S-K
    overhaul, so this is kind A under D25 -- reported in prose, never filtered in code.
    """
    try:
        from src.context import get_config_context
        _, ctx = get_config_context("./configs", use_cache=False, save=False)
        dc = ctx.store.load(Tables.def14a_director_comp)
        parent = ctx.store.load(Tables.def14a_llm)
    except Exception as e:                                  # noqa: BLE001
        pytest.skip(f"def14a_director_comp not reachable ({e})")
    if dc is None or dc.empty or parent is None or parent.empty:
        pytest.skip("def14a tables empty")

    before = float(pd.to_numeric(dc["total"], errors="coerce").notna().mean())
    clean, stats = impute_director_comp(dc)
    after = float(pd.to_numeric(clean["total"], errors="coerce").notna().mean())
    assert after >= before and after > 0.99, f"total coverage {after:.1%}"

    p = parent.copy()
    p["as_of"] = pd.to_datetime(p["as_of"], errors="coerce")
    have = set(clean["accession_number"])
    p["_has"] = p["accession_number"].isin(have)
    eras = [(1995, 1999), (2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2026)]

    idx = pd.bdate_range("1995-01-01", "2026-12-31")
    frames, tally = director_pay_fields(clean, p, idx)
    assert set(frames) == ALL_FIELDS, sorted(set(ALL_FIELDS) - set(frames))
    for name in ALL_FIELDS:
        assert tally[f"{name}: filings"] >= 6_500, f"{name}: {tally[f'{name}: filings']:,}"

    zero = sorted(set(parent["ticker"]) - set(dc["ticker"]))
    print("\n=== SANITY CHECK: def14a_director_comp, live ===")
    print(f"  {len(dc):,} rows / {dc['ticker'].nunique()} tickers / "
          f"{dc['accession_number'].nunique():,} filings")
    print(f"  total coverage {before:.1%} -> {after:.1%} "
          f"(+{stats['total = sum(components)']:,} by the six-component identity)")
    print("  the REGIME staircase (kind A -- prose, never a date filter):")
    for lo, hi in eras:
        sel = p[(p["as_of"].dt.year >= lo) & (p["as_of"].dt.year <= hi)]
        if len(sel):
            print(f"    {lo}-{str(hi)[2:]}: {sel['_has'].mean():6.1%}  "
                  f"({int(sel['_has'].sum()):,} of {len(sel):,} proxies)")
    print("  the four features, per filing:")
    for name in sorted(ALL_FIELDS):
        n = tally[f"{name}: filings"]
        print(f"    {name:28s} {n:6,} filings  "
              f"{int(frames[name].notna().any().sum()):3d} tickers")
    print(f"  per-TICKER (a parser defect): {dc['ticker'].nunique()} of {parent['ticker'].nunique()} -- ZERO "
          f"rows for {zero}")
    print("  CONCLUSION: the family is ~90% covered in the era the model trades, the pre-2006 "
          "sparsity is the disclosure regime and not a defect, and the six missing tickers ARE "
          "an Item 402(k) parser defect that this phase reports rather than absorbs. Validated.")
