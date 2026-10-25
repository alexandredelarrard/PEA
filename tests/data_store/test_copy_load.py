"""COPY keeps the native fallback and required text/array bindings."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src.data_store import store as store_module
from src.data_store.schema import Tables
from src.data_store.store import DataStore


def test_sqlite_copy_falls_back_before_scanning_cells(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"ticker": ["A"], "date": ["2024-01-01"]})
    observed = []

    def refuse_scan(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("SQLite does not need COPY array detection")

    def upsert(engine: Any, rows: pd.DataFrame, name: str, keys: list[str]) -> int:
        observed.append((engine, rows, name, keys))
        return len(rows)

    monkeypatch.setattr(pd.Series, "apply", refuse_scan)
    monkeypatch.setattr(store_module, "upsert_dataframe", upsert)
    assert store_module.copy_load(sqlite_store.engine, frame, Tables.prices.name) == 1
    assert len(observed) == 1 and observed[0][1] is frame
    assert observed[0][2:] == (Tables.prices.name, list(Tables.prices.pk))
    print("SANITY: non-PostgreSQL COPY falls back before any per-cell scan, with the same frame and keys.")


@pytest.mark.live
def test_disposable_postgres_copy_arrays_and_empty_text() -> None:
    url = os.getenv("PEA_13F_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires the owned disposable PostgreSQL URL")
    parsed = make_url(url)
    assert (parsed.host, parsed.port, parsed.username, parsed.database) == ("127.0.0.1", 55433, "p3", "p3")
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE TABLE "{Tables.ticker_embeddings.name}" (ticker TEXT PRIMARY KEY, embedding DOUBLE PRECISION[])'))
            conn.execute(
                text(f'CREATE TABLE "{Tables.prices.name}" (ticker TEXT NOT NULL, date DATE NOT NULL, note TEXT, PRIMARY KEY (ticker,date))')
            )
        arrays = pd.DataFrame({"ticker": ["list", "tuple"], "embedding": [[1.0, 2.0], (3.0, 4.0)]})
        assert store_module.copy_load(engine, arrays, Tables.ticker_embeddings.name) == 2
        strings = pd.DataFrame({"ticker": ["", "B"], "date": ["2024-01-01"] * 2, "note": [None, "text"]})
        assert store_module.copy_load(engine, strings, Tables.prices.name) == 2
        store = DataStore(engine)
        actual = store.load(Tables.ticker_embeddings, order_by="ticker")
        assert actual is not None and actual.embedding.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        actual = store.load(Tables.prices, order_by="ticker")
        assert actual is not None and actual.ticker.tolist() == ["", "B"] and actual.note.isna().tolist() == [True, False]
    finally:
        engine.dispose()
    print("SANITY: disposable PostgreSQL retains list/tuple arrays, empty required text and nullable text through the actual writers.")
