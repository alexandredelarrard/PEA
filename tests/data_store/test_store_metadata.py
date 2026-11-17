"""Schema-only lookups and projected reads must avoid repeated metadata work."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from src.data_store import store as store_module
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.window import Window


def test_columns_uses_lightweight_inspection(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"ticker": ["A"], "date": ["2024-01-01"], "close": [1.0]})
    sqlite_store.save(Tables.prices, frame)

    def refuse_reflection(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Column names do not need a reflected Table")

    monkeypatch.setattr(store_module, "_reflect", refuse_reflection)
    assert sqlite_store.columns(Tables.prices) == list(frame.columns)
    assert sqlite_store.columns(Tables.sharadar_fundamentals_history) == []
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


@pytest.mark.parametrize("stream", [False, True])
def test_projected_read_reuses_reflected_schema(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    table = replace(Tables.prices, read_columns=("ticker", "date", "close"))
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": ["A"], "date": ["2024-01-01"], "close": [1.0]}))
    reader = DataStore(sqlite_store.engine)
    counts = {"exists": 0, "reflect": 0}
    real_exists, real_reflect = store_module.table_exists, store_module._reflect

    def exists(*args: Any, **kwargs: Any) -> bool:
        counts["exists"] += 1
        return real_exists(*args, **kwargs)

    def reflect(*args: Any, **kwargs: Any) -> Any:
        counts["reflect"] += 1
        return real_reflect(*args, **kwargs)

    monkeypatch.setattr(store_module, "table_exists", exists)
    monkeypatch.setattr(store_module, "_reflect", reflect)
    if stream:
        result = pd.concat(reader.iter_load(table, project=True, chunksize=1))
    else:
        result = reader.load(table, project=True)
    assert result is not None and result.ticker.tolist() == ["A"]
    assert counts == {"exists": 1, "reflect": 1}
    reader.load(table, project=True)
    assert counts == {"exists": 1, "reflect": 1}
    print("SANITY: cold projection shares one reflection/existence check; warm reads reuse the verified schema.")


def test_summary_uses_the_same_inclusive_window_as_reads(sqlite_store: DataStore) -> None:
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": ["A", "A", "A", "B"], "date": dates, "close_split": [1.0, 2.0, 3.0, 4.0]}))
    rows = sqlite_store.summarize(Tables.prices, ["ticker"], {"rows": ("count", None)}, window=Window(dates[1], dates[2]))
    assert rows.to_dict("records") == [{"ticker": "A", "rows": 2}]
    print("SANITY: the grouped summary includes both Window endpoints and excludes prefix, suffix and unrelated keys.")
