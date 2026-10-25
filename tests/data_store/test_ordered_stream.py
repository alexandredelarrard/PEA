"""Ordered streaming preserves call groups over chunk boundaries."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from sqlalchemy import event

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


@pytest.mark.parametrize("exit_mode", ["exhaust", "close", "consumer_error", "read_error"])
def test_iter_load_returns_connection_on_every_exit(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, exit_mode: str) -> None:
    rows = pd.DataFrame({"ticker": ["A", "B"], "date": ["2024-01-01"] * 2})
    sqlite_store.save(Tables.prices, rows)
    checked_out = []

    def checkout(*args: Any) -> None:
        checked_out.append(True)

    def checkin(*args: Any) -> None:
        checked_out.pop()

    event.listen(sqlite_store.engine, "checkout", checkout)
    event.listen(sqlite_store.engine, "checkin", checkin)
    if exit_mode == "read_error":

        def fail_read(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("read failed")

        monkeypatch.setattr(pd, "read_sql", fail_read)
    stream = sqlite_store.iter_load(Tables.prices, columns=["ticker", "date"], chunksize=1)
    if exit_mode == "read_error":
        with pytest.raises(RuntimeError, match="read failed"):
            next(stream)
    else:
        next(stream)
        assert checked_out == [True]
        if exit_mode == "exhaust":
            list(stream)
        elif exit_mode == "close":
            stream.close()
        else:
            with pytest.raises(RuntimeError, match="consumer failed"):
                stream.throw(RuntimeError("consumer failed"))
    assert checked_out == []
    print(f"SANITY: stream {exit_mode} returns its held connection to the pool.")
