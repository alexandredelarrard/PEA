"""Ordered streaming preserves call groups over chunk boundaries."""

from __future__ import annotations

import pandas as pd

from src.data_store.schema import Tables
from src.data_store.store import DataStore


def test_iter_load_orders_before_chunking(sqlite_store: DataStore) -> None:
    rows = pd.DataFrame(
        {"ticker": ["B", "A", "A", "B"], "quarter": ["2024Q1", "2024Q2", "2024Q1", "2024Q2"], "paragraph": [1] * 4, "as_of": ["2024-01-01"] * 4}
    )
    sqlite_store.save(Tables.earnings_call_sections, rows)
    columns = ["ticker", "quarter", "paragraph"]
    chunks = list(sqlite_store.iter_load(Tables.earnings_call_sections, columns=columns, order_by=columns, chunksize=1))
    actual = pd.concat(chunks, ignore_index=True)
    pd.testing.assert_frame_equal(actual, rows[columns].sort_values(columns).reset_index(drop=True), check_dtype=False)
    print("SANITY: SQL ordering precedes one-row chunking; every call key appears once in deterministic order.")
