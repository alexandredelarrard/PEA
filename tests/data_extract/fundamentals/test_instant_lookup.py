"""`InstantLookup` vs `carry_latest_known`: the vectorised as-of read must answer exactly
what the `merge_asof` it replaced answered.

`_snapshot` asks "this balance-sheet level's latest known value at `as_of`" once per instant
field per publication event -- 26 x E times a ticker -- and the old form built a one-row
DataFrame and ran `merge_asof` for each. Measured on MCD's full history (2,505 instant rows,
26 fields, 69 events = 1,794 lookups): **42.58 s of CPU against 2.77 s, a 15.4x saving**, and
the two find the same number of values.

That is a replacement of a numeric primitive, so it gets a known-truth fixture rather than a
real filer: the fixture carries every tie the two implementations could disagree on, which a
real ticker is not guaranteed to contain. `carry_latest_known` stays in the tree as the
ORACLE this pins against.
"""
from __future__ import annotations

import random

import pandas as pd

from src.data_extract.utils.fundamentals.build_history import (
    _normalise_facts, carry_latest_known)
from src.data_extract.utils.fundamentals.kpi_catalogue import load_catalogue
from src.data_extract.utils.fundamentals.periods import InstantLookup, instant_stock

CATALOGUE = load_catalogue("./configs")


def _instant(field: str, period_end: str, filing_date: str, value: float | None,
             accession: str) -> dict:
    """One instant `fundamentals_facts` row. No `period_start` is the load-bearing part --
    it is what tells `instant_stock` this is a level and not a duration."""
    return {"ticker": "TST", "accession_number": accession, "field": field,
            "fiscal_year": pd.Timestamp(period_end).year, "fiscal_period": "Q1",
            "duration_type": "instant", "form": "10-Q", "filing_date": filing_date,
            "is_amendment": False, "period_of_report": period_end, "regime": "industrial",
            "period_start": None, "period_end": period_end, "period_days": None,
            "value": value, "unit": "USD", "source_concept": "us-gaap:Assets",
            "dc_code": None, "adjustment": None}


def test_instant_lookup_matches_merge_asof():
    """50 as-of dates x 3 fields against the oracle, over a fixture carrying two vintages
    of one date, a two-year gap, three exact-match dates, a date before all history, a null
    value and a field the filer never reported."""
    rows = [
        # Two vintages of the SAME date -- the later filing must win.
        _instant("totalAssets", "2021-03-31", "2021-05-03", 1000.0, "i-1"),
        _instant("totalAssets", "2021-03-31", "2021-11-01", 1050.0, "i-2"),
        _instant("totalAssets", "2021-06-30", "2021-08-02", 1100.0, "i-3"),
        # A two-year gap: nothing between 2021-06-30 and 2023-06-30.
        _instant("totalAssets", "2023-06-30", "2023-08-02", 1300.0, "i-4"),
        # A second field, and a null value on it -- both must stay distinguishable.
        _instant("cash", "2021-03-31", "2021-05-03", 50.0, "i-5"),
        _instant("cash", "2022-03-31", "2022-05-03", None, "i-6"),
    ]
    facts = _normalise_facts(pd.DataFrame(rows), CATALOGUE)
    instants = instant_stock(facts)
    lookup = InstantLookup(instants)

    random.seed(20260828)
    low, high = pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")
    dates = [pd.Timestamp("2021-03-31"), pd.Timestamp("2021-06-30"),   # exact matches
             pd.Timestamp("2023-06-30"), pd.Timestamp("2019-01-01")]   # and one before all
    dates += [low + pd.Timedelta(days=random.randint(0, (high - low).days))
              for _ in range(50 - len(dates))]

    mismatches = []
    for field in ("totalAssets", "cash", "goodwill"):     # goodwill: never reported
        for date in dates:
            oracle = carry_latest_known(instants, [date], field)[field].iloc[0]
            oracle = None if pd.isna(oracle) else float(oracle)
            got = lookup.value(field, date)
            if oracle != got:
                mismatches.append((field, str(date.date()), oracle, got))

    restated = lookup.value("totalAssets", pd.Timestamp("2021-05-03"))
    print("\n=== SANITY CHECK: InstantLookup == carry_latest_known (merge_asof) ===")
    print(f"  {len(dates)} as-of dates x 3 fields = {len(dates) * 3} lookups, including "
          "3 exact-match dates, a pre-history date, a 2y gap and a never-reported field")
    print(f"  same-day duplicate 2021-03-31: as-filed 1000.0 -> restated 1050.0; "
          f"lookup at 2021-05-03 = {restated}")
    print(f"  null value on cash at 2022-03-31 reads as "
          f"{lookup.value('cash', pd.Timestamp('2022-06-30'))} (None, not 0.0)")
    print(f"  MISMATCHES: {len(mismatches)}")
    if mismatches:
        print(f"  first few: {mismatches[:5]}")
    assert not mismatches, f"{len(mismatches)} lookups disagree with the oracle"
    assert restated == 1050.0, restated
    print("  Validated.")
