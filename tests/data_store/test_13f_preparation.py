"""13F writer preparation must stay cheap without losing schema or transaction checks."""

from __future__ import annotations

import datetime as dt
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url

from src.data_store import store as store_module
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from tests.data_extract.institutionals.test_13f_vintages import packet


@pytest.mark.parametrize("table", [Tables.sec13f_hr, Tables.sec13f_manager_holdings])
def test_verified_filing_reuses_schema_preparation(sqlite_store: DataStore, table: Any) -> None:
    counts = {"metadata": 0}

    def count(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool) -> None:
        if statement.lstrip().startswith("PRAGMA"):
            counts["metadata"] += 1

    event.listen(sqlite_store.engine, "before_cursor_execute", count)
    first, rows = packet("ready-original", "2024-08-14", {"111111111": 100})
    if table == Tables.sec13f_hr:
        rows = rows.assign(ticker="AAA")
    assert sqlite_store.save_13f_filing(first, rows, table=table) == 1
    cold = counts["metadata"]
    second, updated = packet("ready-amendment", "2024-08-20", {"111111111": 90}, kind="restatement")
    if table == Tables.sec13f_hr:
        updated = updated.assign(ticker="AAA")
    assert sqlite_store.save_13f_filing(second, updated, table=table) == 1
    assert sqlite_store.save_13f_filing(second, updated, table=table) == 0
    assert counts["metadata"] == cold
    assert set(sqlite_store.load(table).accession_number) == {"ready-original", "ready-amendment"}
    print("SANITY: verified same-layout filings and immutable replay issue no repeated schema metadata queries; source checks still run.")


@pytest.mark.parametrize("operation", ["drop", "replace", "save", "ensure_columns"])
def test_filing_preparation_refreshes_after_store_changes(sqlite_store: DataStore, operation: str) -> None:
    table = Tables.sec13f_manager_holdings
    original, rows = packet("before-change", "2024-08-14", {"111111111": 100})
    sqlite_store.save_13f_filing(original, rows)
    if operation == "drop":
        sqlite_store.drop(table)
    elif operation == "replace":
        sqlite_store.replace(table, rows.assign(store_note="replacement"))
    elif operation == "save":
        sqlite_store.save(table, rows.assign(store_note="ordinary-save"))
    else:
        sqlite_store.ensure_columns(table, pd.DataFrame({"store_note": ["added"]}))
    revised, changed = packet("after-change", "2024-08-20", {"111111111": 90}, kind="restatement")
    changed = changed.assign(store_note="new-value")
    assert sqlite_store.save_13f_filing(revised, changed) == 1
    assert sqlite_store.load(table, columns=["store_note"], where={"accession_number": "after-change"}).store_note.iloc[0] == "new-value"
    assert sqlite_store.save_13f_filing(revised, changed) == 0
    newer, extra = packet("after-new-column", "2024-08-21", {"111111111": 80}, kind="restatement")
    extra = extra.assign(incoming_note="new-column")
    assert sqlite_store.save_13f_filing(newer, extra) == 1
    assert "incoming_note" in sqlite_store.columns(table)
    print(f"SANITY: {operation} invalidates preparation, incoming columns refresh it, and source/replay integrity survives.")


@pytest.mark.parametrize("table", [Tables.sec13f_hr, Tables.sec13f_manager_holdings])
def test_prepared_filing_builds_no_typed_frame(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, table: Any) -> None:
    ticker = {"ticker": "AAA"} if table == Tables.sec13f_hr else {}
    first, rows = packet("typed-cold", "2024-08-14", {"111111111": 100})
    second, more = packet("typed-warm", "2024-08-20", {"111111111": 90, "222222222": 5}, kind="restatement")
    calls = {"concat": 0}
    real_concat = store_module.pd.concat

    def counting_concat(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls["concat"] += 1
        return real_concat(*args, **kwargs)

    monkeypatch.setattr(store_module.pd, "concat", counting_concat)
    assert sqlite_store.save_13f_filing(first, rows.assign(**ticker), table=table) == 1
    cold = calls["concat"]
    assert sqlite_store.save_13f_filing(second, more.assign(**ticker), table=table) == 2
    warm = calls["concat"] - cold
    stored = sqlite_store.load(table, where={"accession_number": "typed-warm"}, markers=True)
    assert (cold, warm) == (1, 0)
    assert sorted(stored.parse_status) == ["complete", "position", "position"]
    print(
        f"SANITY: {table.name}: the typed DDL frame is built {cold}x on the cold save and {warm}x on a prepared save; 2 positions + 1 marker stored."
    )


def test_records_match_the_to_dict_binding() -> None:
    frame = pd.DataFrame(
        {
            "i": np.array([1, 2, 3], dtype="int64"),
            "f": [1.5, np.nan, 3.0],
            "I": pd.array([1, None, 3], dtype="Int64"),
            "B": pd.array([True, None, False], dtype="boolean"),
            "s": pd.Series(["a", None, "c"], dtype="str"),
            "o": pd.Series([np.int64(4), None, "x"], dtype=object),
            "t": pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            "d": pd.Series([dt.date(2024, 1, 1), None, pd.NaT], dtype=object),
            "ci": pd.Categorical([1, None, 2]),
            "l": pd.Series([[1, 2], None, (3,)], dtype=object),
        }
    )
    expected = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    got = store_module._records(frame)
    pairs = [(a[k], b[k]) for a, b in zip(expected, got, strict=True) for k in frame.columns]
    assert [list(r) for r in got] == [list(r) for r in expected]
    assert all(type(a) is type(b) and (a is None or (isinstance(a, float) and math.isnan(a)) or a == b) for a, b in pairs)
    assert store_module._records(pd.DataFrame(index=range(2))) == []
    print(f"SANITY: _records binds the same Python type and value as to_dict on {len(pairs)} cells over {frame.shape[1]} dtypes; NaN/NaT/NA -> None.")


@pytest.mark.live
def test_disposable_postgres_concurrent_cold_preparation() -> None:
    url = os.environ.get("PEA_13F_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires dedicated disposable PostgreSQL URL")
    parsed = make_url(url)
    assert parsed.host == "127.0.0.1" and parsed.port == 55433 and parsed.database == "p3" and parsed.username == "p3"
    store = DataStore(create_engine(url))
    tables = (Tables.sec13f_hr, Tables.sec13f_manager_holdings)
    for table in tables:
        store.drop(table)
    manifest, rows = packet("concurrent-cold", "2024-08-14", {"111111111": 100})

    def write(table: Any) -> tuple[str, int]:
        positions = rows.assign(ticker="AAA") if table == Tables.sec13f_hr else rows
        return table.name, store.save_13f_filing(manifest, positions, table=table)

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(write, [*tables, *tables]))
    for table in tables:
        assert sorted(value for name, value in outcomes if name == table.name) == [0, 1]
        assert len(store.load(table, where={"parse_status": "complete"}, markers=True)) == 1
        store.drop(table)
    store.engine.dispose()
    print("SANITY: simultaneous first-use writers prepare each cold destination safely and preserve atomic one-save/one-replay results.")
