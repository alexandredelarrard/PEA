"""Offline known-truth check: a NULL employee row never interrupts the prior year's carry."""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.fundamentals_sharadar.merge_history import attach_employees


def test_null_employee_row_keeps_the_prior_count_carrying():
    frame = pd.DataFrame({"ticker": ["AAA", "AAA"], "as_of": pd.to_datetime(["2024-06-30", "2025-01-31"])})
    employees = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "as_of": pd.to_datetime(["2024-03-01", "2025-01-15"]),
            "employees": [40_000.0, float("nan")],  # the 2025 10-K stated no usable count
        }
    )
    out = attach_employees(frame, employees)
    assert out["employees"].tolist() == [40_000.0, 40_000.0]
    assert attach_employees(frame, employees.iloc[1:])["employees"].isna().all()
    print("\nSANITY: a NULL 10-K row is skipped by the as-of join, so the prior count still carries within 370 days.")
