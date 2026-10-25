"""Schema-only lookups and projected reads must avoid repeated metadata work."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data_store import store as store_module
from src.data_store.schema import Tables
from src.data_store.store import DataStore


def test_columns_uses_lightweight_inspection(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"ticker": ["A"], "date": ["2024-01-01"], "close": [1.0]})
    sqlite_store.save(Tables.prices, frame)

    def refuse_reflection(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Column names do not need a reflected Table")

    monkeypatch.setattr(store_module, "_reflect", refuse_reflection)
    assert sqlite_store.columns(Tables.prices) == list(frame.columns)
    assert sqlite_store.columns(Tables.fundamentals_history) == []
    print("SANITY: ordered names and missing-table behavior use inspection without full reflection.")


def test_bulk_seed_reads_columns_once(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"ticker": ["A"], "date": ["2024-01-01"], "close": [1.0]})
    sqlite_store.save(Tables.prices, frame)
    original = sqlite_store.columns
    calls = []

    def columns(table: Any) -> list[str]:
        calls.append(table)
        return original(table)

    monkeypatch.setattr(sqlite_store, "columns", columns)
    assert sqlite_store.bulk_seed(Tables.prices, frame.assign(ticker="B")) == 1
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(ValueError, match="silently drop"):
        sqlite_store.bulk_seed(Tables.prices, frame.assign(unknown="reject"))
    assert len(calls) == 1
    assert sqlite_store.row_count(Tables.prices) == 2
    print("SANITY: bulk seed checks the schema once and refuses unknown columns before writing.")
