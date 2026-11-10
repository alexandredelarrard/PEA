"""`load_tape_cohorts` loads an OR scope one branch at a time and returns what the single OR query returns."""

from __future__ import annotations

import pandas as pd

from src.data_extract.utils.institutionals.security_tape import load_tape_cohorts
from src.data_store.schema import Tables
from src.data_store.store import DataStore


def test_an_or_scope_loads_branch_by_branch_with_each_row_once(sqlite_store: DataStore) -> None:
    """A row matched by the period branch and by the key branch comes back once; the union equals the OR load."""
    table = Tables.sec_fails_to_deliver_security
    stored = pd.DataFrame(
        {
            "cusip": ["000111AAA", "000111AAA", "222333BBB"],
            "date": pd.to_datetime(["2015-01-20", "2015-02-03", "2015-01-20"]),
            "period": ["201501b", "201502a", "201501b"],
            "source_symbol": ["ABC", "ABC", "XYZ"],
            "fails_quantity": [100.0, 50.0, 9.0],
        }
    )
    sqlite_store.save(table, stored)
    keys = stored.loc[[0, 1], ["cusip", "date"]]
    where = [{"period": "201501b"}, keys]

    cohort = load_tape_cohorts(sqlite_store, {table: where})[table]
    expected = sqlite_store.load(table, columns=sqlite_store.columns(table), where=where, markers=True)

    assert expected is not None and len(cohort) == len(expected) == 3, cohort
    assert not cohort.duplicated(["cusip", "date"]).any()
    assert sorted(zip(cohort["cusip"], cohort["period"], strict=True)) == sorted(zip(expected["cusip"], expected["period"], strict=True))
    assert load_tape_cohorts(sqlite_store, {Tables.sec_short_volume_security: where})[Tables.sec_short_volume_security].empty, "an absent table"
    print("\nSANITY: an OR scope loaded branch by branch returns the 3 rows of the OR load, the doubly matched row once.")
