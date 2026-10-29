"""Fresh selected managed schemas come from canonical SQL without runtime adoption."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import event, inspect

from src.data_store.schema import Tables
from src.data_store.store import DataStore


def test_canonical_source_keys_and_retry_fields() -> None:
    tables = [
        Tables.sec13f_hr,
        Tables.sec13f_manager_holdings,
        Tables.def14a_llm,
        Tables.def14a_executive_comp,
        Tables.def14a_director_comp,
        Tables.def14a_ownership,
        Tables.def14a_directors,
    ]
    with DataStore.temporary(tables) as store:
        for table in tables:
            assert inspect(store.engine).get_pk_constraint(table.name)["constrained_columns"] == list(table.pk)
            if table in tables[2:]:
                assert "retry" in store.columns(table)
            else:
                assert "accession_number" in store.columns(table)
                assert "content_checksum" in store.columns(table)
    print("SANITY: both accession keys and all five retry fields exist on a fresh canonical schema.")


def test_bound_writer_does_not_adopt_missing_fields(sqlite_store: DataStore) -> None:
    table = Tables.def14a_llm
    frame = pd.DataFrame({"ticker": ["A"], "accession_number": ["0001"]})
    sqlite_store.save(table, frame)
    before = sqlite_store.columns(table)
    with pytest.raises(ValueError, match="unknown columns"):
        with sqlite_store.transaction() as store:
            store.save(table, frame.assign(retry=1))
    assert sqlite_store.columns(table) == before
    print("SANITY: a bound managed writer refuses missing fields and never silently repairs an existing schema.")


@pytest.mark.parametrize("method", ["save", "replace"])
def test_ordinary_managed_writes_never_adopt_an_existing_schema(sqlite_store: DataStore, method: str) -> None:
    table = Tables.prices
    df_rows = pd.DataFrame({"ticker": ["AAA"], "date": [pd.Timestamp("2024-01-01")], "close_split": [1.0]})
    sqlite_store.save(table, df_rows)
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(sqlite_store.engine, "before_cursor_execute", record)
    with pytest.raises(ValueError, match="schema lacks"):
        getattr(sqlite_store, method)(table, df_rows.assign(volume=100.0))
    assert "volume" not in sqlite_store.columns(table)
    assert sqlite_store.row_count(table) == 1
    assert not any(statement.lstrip().upper().startswith("ALTER ") for statement in statements)
    print("SANITY: ordinary managed writes reject missing fields before mutation, without adoption ALTERs.")
