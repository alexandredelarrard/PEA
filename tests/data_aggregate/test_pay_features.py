"""
Executive-compensation families (`src/data_aggregate/utils/governance/pay_features.py`).

Synthetic known-truth, because every claim in this phase is arithmetic that a real-data
correlation would not pin down:

  * the TURNOVER GUARD nulls growth across a CEO change while the flag reads 1.0, and does NOT
    null it when only the filer's SPELLING of one CEO's name changed;
  * a MISSING proxy year yields NaN rather than a two-year change mislabelled as one year;
  * the exact CEO Pay Slice sums the five largest NEOs OF THE LATEST FISCAL YEAR ONLY;
  * both legs of every gap are LOG growths, so a 20% raise against 20% revenue growth is a
    gap of ZERO -- the test that fails if either leg reverts to a percentage change;
  * the severity products are ONE-SIDED.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.pay_features import (
    ALL_FIELDS,
    EVENT_FIELDS,
    PEER_RELATIVE_FIELDS,
    RAW_FLAG_FIELDS,
    _comp_history,
    _severity,
    _slice_history,
    _top5_history,
    pay_fields,
)

# Ends well past 2023-04-01 + 548d (= 2024-10-01), so the expiry horizon is actually
# CROSSED inside the index -- otherwise the expiry assertion below passes vacuously.
IDX = pd.date_range("2020-01-01", "2025-06-30", freq="B")
PEERS = {t: {p: 1.0 for p in ("AAA", "BBB", "CCC", "DDD") if p != t}
         for t in ("AAA", "BBB", "CCC", "DDD")}


def _def14a() -> pd.DataFrame:
    """Four tickers, each carrying one distinct guard case.

    AAA  one CEO, three spellings of their name -> growth on both pairs
    BBB  a genuine CEO change in 2022           -> growth NaN there, flag 1.0
    CCC  a MISSING 2022 proxy                   -> the 2023 pair spans two years -> NaN
    DDD  an unknown name in 2022                -> growth NaN, flag NaN (not 0.0)
    """
    rows = [
        # (ticker, as_of, comp, name)
        ("AAA", "2021-04-01", 10e6, "Timothy D. Cook"),
        ("AAA", "2022-04-01", 12e6, "Tim Cook"),          # +20%, SAME person
        ("AAA", "2023-04-01", 15e6, "Timothy Cook"),      # +25%, same person again
        ("BBB", "2021-04-01", 8e6, "Alice Smith"),
        ("BBB", "2022-04-01", 20e6, "Bob Jones"),         # turnover -> growth NaN
        ("BBB", "2023-04-01", 22e6, "Bob Jones"),         # +10%, computable
        ("CCC", "2021-04-01", 5e6, "Carol White"),
        ("CCC", "2023-04-01", 9e6, "Carol White"),        # 2022 MISSING -> NaN
        ("DDD", "2021-04-01", 6e6, "Dan Brown"),
        ("DDD", "2022-04-01", 7e6, None),                 # unknown -> NaN both ways
    ]
    return pd.DataFrame([
        {"ticker": t, "accession_number": f"{t}-{d[:4]}", "as_of": pd.Timestamp(d),
         "ceo_total_comp": c, "ceo_name_proxy": n}
        for t, d, c, n in rows])


def _exec_comp() -> pd.DataFrame:
    """AAA's 2023 filing: seven NEOs for FY2022 and the same seven for FY2021 at HALF the pay.

    Item 402(c) requires three fiscal years, so the real table looks exactly like this. If the
    builder summed the whole table the denominator would carry two years against a one-year
    numerator -- the defect the latest-fiscal-year rule exists to prevent.
    """
    rows = []
    # Real surnames, distinct initials: `person_key` is `lastname|firstinitial`, so
    # placeholder names like "Exec 0".."Exec 6" would all key to `exec|e` and collapse into
    # ONE deputy -- a fixture artefact that would silently exercise the 3-NEO floor instead of
    # the top-five sum.
    neos = [("Timothy Cook", 9e6), ("Luca Maestri", 5e6), ("Katherine Adams", 4e6),
            ("Deirdre O'Brien", 3e6), ("Jeff Williams", 2e6), ("Greg Joswiak", 1e6),
            ("Sabih Khan", 0.5e6)]                        # top five sum to 23e6
    for name, pay in neos:
        for fy, mult in ((2022, 1.0), (2021, 0.5)):
            rows.append({"ticker": "AAA", "accession_number": "AAA-2023",
                         "as_of": pd.Timestamp("2023-04-01"), "name": name,
                         "fiscal_year": fy, "total": pay * mult, "reconciles": 1.0})
    # BBB's 2023 filing carries only TWO NEOs -> below the 3-NEO floor -> rejected entirely
    for name in ("Bob Jones", "Nadia Farrell"):
        rows.append({"ticker": "BBB", "accession_number": "BBB-2023",
                     "as_of": pd.Timestamp("2023-04-01"), "name": name,
                     "fiscal_year": 2022, "total": 11e6, "reconciles": 1.0})
    return pd.DataFrame(rows)


def test_turnover_guard_and_the_missing_year():
    tally: dict[str, int] = {}
    h = _comp_history(_def14a(), tally).set_index(["ticker", "as_of"])

    g = h["ceo_comp_growth_1y"]
    # 1. one CEO, three spellings -> BOTH pairs computable, and the values are exact log growths
    assert g.loc[("AAA", pd.Timestamp("2022-04-01"))] == pytest.approx(np.log(1.20))
    assert g.loc[("AAA", pd.Timestamp("2023-04-01"))] == pytest.approx(np.log(1.25))
    # 2. a genuine turnover nulls the growth and raises the flag
    assert np.isnan(g.loc[("BBB", pd.Timestamp("2022-04-01"))])
    assert h["ceo_turnover_flag"].loc[("BBB", pd.Timestamp("2022-04-01"))] == 1.0
    assert g.loc[("BBB", pd.Timestamp("2023-04-01"))] == pytest.approx(np.log(22 / 20))
    # 3. ⚠ THE MISSING-YEAR CASE. CCC's 2023 proxy follows its 2021 one, so `shift(1)` offers a
    #    TWO-year pair. It must be NaN, not a 80% "annual" raise borrowed from Y-2.
    assert np.isnan(g.loc[("CCC", pd.Timestamp("2023-04-01"))])
    assert tally["growth nulled: filings not ~1y apart"] == 1
    # 4. an unknown name is UNKNOWN, never "no change": growth NaN and the flag NaN too
    assert np.isnan(g.loc[("DDD", pd.Timestamp("2022-04-01"))])
    assert np.isnan(h["ceo_turnover_flag"].loc[("DDD", pd.Timestamp("2022-04-01"))])
    # 5. a first observation has no prior filing at all
    assert np.isnan(g.loc[("AAA", pd.Timestamp("2021-04-01"))])
    assert tally["CEO turnovers detected"] == 1

    print("\n=== SANITY CHECK: the CEO turnover guard ===")
    print(f"  3 spellings of one CEO -> growth kept: log(1.20)={np.log(1.2):.4f} and "
          f"log(1.25)={np.log(1.25):.4f}; a real change -> NaN + flag 1.0.")
    print(f"  a MISSING proxy year -> NaN (not a 2y change relabelled 1y): "
          f"{tally['growth nulled: filings not ~1y apart']} such pair rejected.")
    print("  an unknown name -> NaN on BOTH the growth and the flag. Validated.")


def test_exact_pay_slice_latest_fiscal_year_and_the_floors():
    tally: dict[str, int] = {}
    top5 = _top5_history(_exec_comp(), tally)
    assert list(top5["ticker"]) == ["AAA"], "the 2-NEO filing must be rejected, not shipped"
    # the five largest of FY2022 only: 9 + 5 + 4 + 3 + 2 = 23e6. Summing both fiscal years
    # would give 34.5e6, and the CEO's slice would silently shrink by a third.
    assert top5["top5_neo_total_comp"].iloc[0] == pytest.approx(23e6)
    assert top5["n_neos_top5"].iloc[0] == 5
    assert tally["CPS filings rejected (< 3 NEOs)"] == 1

    sl = _slice_history(_def14a(), top5, tally)
    # AAA's CEO earned 15e6 of a 23e6 top-five pool
    assert sl["ceo_pay_slice"].iloc[0] == pytest.approx(15e6 / 23e6)

    # a slice above 1 is an extraction defect: rejected and COUNTED, never clipped to 1.0
    small = top5.copy()
    small["top5_neo_total_comp"] = 1e6
    t2: dict[str, int] = {}
    assert _slice_history(_def14a(), small, t2) is None
    assert t2["CPS slices rejected (outside (0, 1])"] == 1

    print("\n=== SANITY CHECK: the exact CEO Pay Slice ===")
    print(f"  top five of the LATEST fiscal year = $23.0M (both years would be $34.5M); "
          f"slice = {sl['ceo_pay_slice'].iloc[0]:.3f}.")
    print(f"  the 2-NEO filing is rejected by the {tally['CPS filings rejected (< 3 NEOs)']}-"
          f"filing floor; a slice > 1 is rejected and counted, not clipped. Validated.")


def test_gaps_are_log_on_both_legs_and_severity_is_one_sided():
    """⚠ THE BASIS TEST. A 20% raise against 20% revenue growth is PERFECT alignment, so the
    gap must be 0. It reads -0.018 if the revenue leg stays a percentage change -- small here,
    but 31pp on a doubled package, and always in the direction that invents misalignment."""
    d14 = _def14a()
    fund = pd.DataFrame([
        {"ticker": "AAA", "as_of": pd.Timestamp(f"{y}-12-31"), "totalRevenue": rev}
        for y, rev in ((2020, 1000.0), (2021, 1000.0), (2022, 1200.0), (2023, 1440.0))])
    F, tally = pay_fields(d14, None, fund, None, PEERS, IDX)

    assert "pay_revenue_gap" in F
    # AAA: pay +20% over 2022, revenue +20% over the same year -> gap 0 on a log basis
    gap = F["pay_revenue_gap"].loc["2023-01-03", "AAA"]
    assert gap == pytest.approx(0.0, abs=1e-9), f"log/pct basis mismatch: gap={gap}"
    assert F["pay_up_revenue_down"].loc["2023-01-03", "AAA"] == 0.0

    # one-sided severity: pay FALLING while performance falls is not a governance failure
    pay = pd.DataFrame({"AAA": [-0.30, 0.30]}, index=IDX[:2])
    perf = pd.DataFrame({"AAA": [-0.40, -0.40]}, index=IDX[:2])
    sev = _severity(pay, perf)
    assert sev.iloc[0, 0] == 0.0
    assert sev.iloc[1, 0] == pytest.approx(0.30 * 0.40)

    # no close_total -> the return family is skipped by NAME, and the revenue one still builds
    assert "pay_return_gap" not in F
    assert tally["skipped: no close_total -> no return-based misalignment"] == 1

    print("\n=== SANITY CHECK: pay-vs-performance basis and severity ===")
    print(f"  +20% pay vs +20% revenue -> gap {gap:+.9f} (log on BOTH legs; a pct revenue leg "
          f"would read {np.log(1.2) - 0.2:+.4f}).")
    print(f"  severity is one-sided: pay -30% into perf -40% -> {sev.iloc[0, 0]:.2f}, "
          f"pay +30% into perf -40% -> {sev.iloc[1, 0]:.2f}.")
    print("  absent close_total, the return family is skipped and the revenue one survives.")
    print("  CONCLUSION: both legs share a basis and the products are one-sided. Validated.")


def test_the_encoding_and_expiry_contracts():
    """EVERY pay field ships raw, and only the two LEVELS survive the 548-day expiry.

    ⚠ THE EMPTY PEER SET IS THE TEST. Phase 4 was planned with `log_ceo_total_comp` and
    `ceo_pay_slice` peer-panelled, and both failed their own precondition on two independent
    measures -- between-sector variance 2.4% / 4.7% and peer-basket R^2 2.4% / 4.1%, against
    the 7.7% floor the same rule kept in phase 3 and 13.6% for `profitMargins`. A CEO package
    is idiosyncratic -- set by one board's committee against a peer group of its own choosing --
    so dividing it by seven similar companies' dispersion standardizes it against noise. The
    emptiness is asserted so no future edit can re-add a peer leg without re-measuring.
    """
    assert PEER_RELATIVE_FIELDS == frozenset(), "a peer leg needs a measured peer norm"
    assert not (PEER_RELATIVE_FIELDS & RAW_FLAG_FIELDS)
    assert RAW_FLAG_FIELDS <= EVENT_FIELDS, "a flag records an event and must expire with it"
    # the two LEVELS are the only non-events: a package and a share of the top five are
    # standing facts between proxies, where "pay grew 40% last year" stops being true
    assert not ({"log_ceo_total_comp", "ceo_pay_slice"} & EVENT_FIELDS)
    # ⚠ THE CLASSIFICATION MUST BE EXHAUSTIVE, and this is the assertion that caught a real
    # defect: `_alignment_family` names its members from a LABEL, so the plan's
    # `pay_up_stock_down` sat in both sets while the `pay_up_return_down` actually built sat in
    # neither -- unclassified as a flag and skipped by the 548-day expiry.
    assert EVENT_FIELDS <= ALL_FIELDS, f"dead names: {sorted(EVENT_FIELDS - ALL_FIELDS)}"
    assert RAW_FLAG_FIELDS <= ALL_FIELDS, f"dead names: {sorted(RAW_FLAG_FIELDS - ALL_FIELDS)}"
    assert ALL_FIELDS - EVENT_FIELDS == {"log_ceo_total_comp", "ceo_pay_slice"}, (
        "every field except the two standing levels must expire")

    F, _ = pay_fields(_def14a(), _exec_comp(), None, None, PEERS, IDX)
    built = set(F)
    # the builder may emit FEWER fields than ALL_FIELDS (a source can be absent) but never a
    # name outside it -- otherwise a field ships unclassified and un-expired
    assert built <= ALL_FIELDS, f"unclassified field(s): {sorted(built - ALL_FIELDS)}"
    assert "ceo_pay_slice" in built and "log_ceo_total_comp" in built
    # ⚠ the level is NOT expired: a package is a standing fact between proxies. AAA's last
    # proxy (2023-04-01) is 821 days before the end of the index, so the 548-day horizon is
    # genuinely crossed -- an expiring level would read NaN here and the growth must.
    assert pd.notna(F["log_ceo_total_comp"].loc[IDX[-1], "AAA"])
    assert pd.isna(F["ceo_comp_growth_1y"].loc[IDX[-1], "AAA"]), "growth must expire at 548d"

    print("\n=== SANITY CHECK: encoding + 548-day expiry ===")
    print(f"  {len(PEER_RELATIVE_FIELDS)} peer-encoded fields (the plan expected 2; both "
          f"failed at 2.4% / 4.1% peer R^2 against a 7.7% floor), {len(RAW_FLAG_FIELDS)} raw "
          f"flags, {len(EVENT_FIELDS)} event fields.")
    print("  the pay LEVEL survives past 548d (a standing fact); the GROWTH expires.")
    print("  CONCLUSION: every pay field ships raw; only standing facts outlive 548d.")
