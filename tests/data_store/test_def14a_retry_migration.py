"""Retry metadata is installed before partial writes and never resets attempts."""

from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import Connection, create_engine, event

from src.data_extract.utils.common.empty_markers import marker_frame
from src.data_extract.utils.structure.def14a.retry import reservation_frame
from src.data_store.schema import Tables
from src.data_store.store import DataStore


def test_first_retry_write_initializes_existing_information_once(sqlite_store: DataStore) -> None:
    table = Tables.def14a_llm
    sqlite_store.save(table, pd.DataFrame([{"ticker": "AAA", "accession_number": "old", "as_of": "2020-01-01", "ceo_age": 61.0}]))
    sqlite_store.save(table, pd.DataFrame([{"ticker": "AAA", "accession_number": "new", "as_of": "2021-01-01", "retry": 1}]))

    df_saved = sqlite_store.load(table, columns=["accession_number", "ceo_age", "retry"], where={"ticker": "AAA"}, markers=True)
    assert df_saved is not None
    old = df_saved.set_index("accession_number").loc["old"]
    assert old["retry"] == 1
    assert old["ceo_age"] == 61

    sqlite_store.save(table, pd.DataFrame([{"ticker": "AAA", "accession_number": "old", "retry": 2}]))
    sqlite_store.prepare_def14a_retries()
    df_again = sqlite_store.load(table, columns=["retry", "ceo_age"], where={"accession_number": "old"}, markers=True)
    assert df_again is not None
    assert df_again.iloc[0].to_dict() == {"retry": 2, "ceo_age": 61}
    print("SANITY: the actual retry-bearing save initializes old information once; later preparation preserves retry=2 and facts.")


def test_cold_retry_write_creates_complete_typed_tables(sqlite_store: DataStore) -> None:
    sqlite_store.save(Tables.def14a_llm, pd.DataFrame([{"ticker": "AAA", "accession_number": "first", "retry": 1}]))

    expected = {
        Tables.def14a_llm: {"auditor_fees", "ceo_age", "def14a_json", "retry"},
        Tables.def14a_executive_comp: {"salary", "fiscal_year", "retry"},
        Tables.def14a_director_comp: {"fees_earned", "total", "retry"},
        Tables.def14a_ownership: {"holder_type", "percent_of_voting_power", "retry"},
        Tables.def14a_directors: {"age", "gender_basis", "retry"},
    }
    for table, required in expected.items():
        assert required <= set(sqlite_store.columns(table)), table.name
    print("SANITY: a counter-only first write creates the five complete canonical schemas, not a key-only skeleton.")


def test_empty_section_counters_are_hidden_but_resume_visible(sqlite_store: DataStore) -> None:
    tables = (Tables.def14a_executive_comp, Tables.def14a_director_comp, Tables.def14a_ownership, Tables.def14a_directors)
    for table in tables:
        values = {"ticker": "AAA", "accession_number": "empty", "as_of": "2020-01-01", "retry": 1}
        df_marker = marker_frame(table, values, key_fill=dict(table.marker_key_fill))
        assert not df_marker[list(table.pk)].isna().any().any()
        sqlite_store.save(table, df_marker)
        assert sqlite_store.load(table, columns=["ticker"], where={"ticker": "AAA"}, optional=True) is None
        df_state = sqlite_store.load(table, columns=[*table.pk, "retry"], where={"ticker": "AAA"}, markers=True)
        assert df_state is not None and df_state.iloc[0]["retry"] == 1
        assert sqlite_store.distinct(table, "accession_number", where={"ticker": "AAA"}) == ["empty"]
    print("SANITY: all four empty sections keep non-null keys and retry=1; ordinary reads hide them while resume sees them.")


def test_failed_retry_migration_rolls_back_columns_and_initialization(sqlite_store: DataStore) -> None:
    for table in (Tables.def14a_llm, Tables.def14a_executive_comp):
        row = {"ticker": "AAA", "accession_number": "legacy", "as_of": "2020-01-01", "name": "A Person", "fiscal_year": 2019}
        sqlite_store.save(table, pd.DataFrame([{k: v for k, v in row.items() if k in {*table.pk, "as_of"}}]))

    def fail_second_update(conn: Connection, cursor: object, statement: str, parameters: object, context: object, executemany: bool) -> None:
        if statement.startswith('UPDATE "' + Tables.def14a_executive_comp.name + '" SET retry'):
            raise RuntimeError("migration interrupted")

    event.listen(sqlite_store.engine, "before_cursor_execute", fail_second_update)
    try:
        with pytest.raises(RuntimeError, match="migration interrupted"):
            sqlite_store.prepare_def14a_retries()
    finally:
        event.remove(sqlite_store.engine, "before_cursor_execute", fail_second_update)

    for table in (Tables.def14a_llm, Tables.def14a_executive_comp):
        assert "retry" not in sqlite_store.columns(table)
        assert sqlite_store.row_count(table) == 1
    sqlite_store.prepare_def14a_retries()
    print("SANITY: a failed migration rolls back ADD and backfill together; the next preparation completes safely.")


@pytest.mark.live
def test_disposable_postgres_retry_preparation() -> None:
    url = os.environ.get("PEA_13F_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires the existing dedicated disposable PostgreSQL test URL")
    engine = create_engine(url)
    if engine.url.database != "def14a_test" or engine.url.username != "def14a_test" or engine.url.host != "127.0.0.1":
        engine.dispose()
        raise ValueError("DEF14A migration proof requires its disposable def14a_test database")
    tables = (Tables.def14a_llm, Tables.def14a_executive_comp, Tables.def14a_director_comp, Tables.def14a_ownership, Tables.def14a_directors)
    store = DataStore(engine)
    checks = (
        test_first_retry_write_initializes_existing_information_once,
        test_cold_retry_write_creates_complete_typed_tables,
        test_empty_section_counters_are_hidden_but_resume_visible,
        test_failed_retry_migration_rolls_back_columns_and_initialization,
    )
    try:
        for check in checks:
            for table in tables:
                store.drop(table)
            check(DataStore(engine))
    finally:
        for table in tables:
            store.drop(table)
        engine.dispose()
    print("SANITY: PostgreSQL confirms one-time initialization, preserved facts/2, typed cold schemas, hidden markers and transactional rollback.")


def test_counter_patch_keeps_the_maximum_attempt_and_only_owned_fields() -> None:
    stamp = {"ticker": "AAA", "cik": "1", "accession_number": "a", "as_of": "2020-01-01"}
    df_parent = reservation_frame(Tables.def14a_llm, stamp, None, 1)
    assert set(df_parent.columns) == {"ticker", "accession_number", "as_of", "retry"}
    df_saved = pd.DataFrame([{**stamp, "name": "A Person", "retry": 2, "fees_earned": 5000.0}])
    df_patch = reservation_frame(Tables.def14a_director_comp, stamp, df_saved, 1)
    assert set(df_patch.columns) == {*Tables.def14a_director_comp.pk, "as_of", "retry"}
    assert df_patch.iloc[0]["retry"] == 2
    print("SANITY: parent patches cannot introduce child-only cik; repeated reservations retain 2 and omit economic facts.")
