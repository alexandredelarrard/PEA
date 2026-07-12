"""Tests for the FMP employee-count normalization
(src/data_extract/fetch_employees.py).

Rotation / incremental / key-discovery live in the shared client and are tested
in test_fmp_client.py; here we only check the employee-specific parsing (filing
date -> point-in-time `as_of`, dedup, bad-row dropping).
"""
from __future__ import annotations

import pandas as pd

from src.data_extract.fetch_employees import normalize_employees


def _fmp_records():
    return [
        {"symbol": "AAPL", "periodOfReport": "2023-09-30", "filingDate": "2023-11-03",
         "formType": "10-K", "employeeCount": 161000},
        {"symbol": "AAPL", "periodOfReport": "2022-09-24", "filingDate": "2022-10-28",
         "formType": "10-K", "employeeCount": 164000},
        # duplicate filing date -> keep last; and junk rows that must be dropped
        {"symbol": "AAPL", "periodOfReport": "2023-09-30", "filingDate": "2023-11-03",
         "formType": "10-K/A", "employeeCount": 161500},
        {"symbol": "AAPL", "periodOfReport": None, "filingDate": None,
         "formType": "10-K", "employeeCount": 999},          # no date -> drop
        {"symbol": "AAPL", "periodOfReport": "2021-09-25", "filingDate": "2021-10-29",
         "formType": "10-K", "employeeCount": 0},             # non-positive -> drop
    ]


def test_normalize_employees_pit_and_cleaning():
    df = normalize_employees(_fmp_records(), "AAPL")

    assert len(df) == 2
    assert df["as_of"].is_monotonic_increasing
    assert list(df["employees"]) == [164000, 161500]   # 2022, then dedup-kept 2023
    assert df.iloc[-1]["as_of"] == pd.Timestamp("2023-11-03")   # FILING date (public)
    assert df.iloc[-1]["period"] == pd.Timestamp("2023-09-30")
    assert normalize_employees([], "AAPL").empty

    print("\n=== SANITY CHECK: FMP employee-count normalization ===")
    print(f"  kept {len(df)} clean rows (dropped no-date + zero-count); dup collapsed.")
    print("  as_of = filing date 2023-11-03 (public) vs period 2023-09-30 -> PIT. Validated.")
