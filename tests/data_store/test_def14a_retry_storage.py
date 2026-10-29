"""Current-schema retry writes preserve facts and hidden resume markers."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import event

from src.data_extract.utils.common.empty_markers import marker_frame
from src.data_extract.utils.structure.def14a.retry import reservation_frame
from src.data_store.schema import Tables
from src.data_store.store import DataStore


def test_retry_partial_write_preserves_facts_without_schema_adoption(filing_store: DataStore) -> None:
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(filing_store.engine, "before_cursor_execute", record)
    table = Tables.def14a_llm
    filing_store.save(table, pd.DataFrame([{"ticker": "AAA", "accession_number": "old", "as_of": "2020-01-01", "ceo_age": 61.0, "retry": 1}]))
    filing_store.save(table, pd.DataFrame([{"ticker": "AAA", "accession_number": "old", "retry": 2}]))
    row = filing_store.load(table, columns=["retry", "ceo_age"], where={"accession_number": "old"}, markers=True)
    assert row is not None
    assert row.iloc[0].to_dict() == {"retry": 2, "ceo_age": 61.0}
    assert not any(statement.lstrip().upper().startswith("ALTER ") for statement in statements)
    print("SANITY: initialized retry storage keeps attempts/facts without schema adoption or backfill.")


def test_empty_section_counters_are_hidden_but_resume_visible(filing_store: DataStore) -> None:
    tables = (Tables.def14a_executive_comp, Tables.def14a_director_comp, Tables.def14a_ownership, Tables.def14a_directors)
    for table in tables:
        values = {"ticker": "AAA", "accession_number": "empty", "as_of": "2020-01-01", "retry": 1}
        df_marker = marker_frame(table, values, key_fill=dict(table.marker_key_fill))
        assert not df_marker[list(table.pk)].isna().any().any()
        filing_store.save(table, df_marker)
        assert filing_store.load(table, columns=["ticker"], where={"ticker": "AAA"}, optional=True) is None
        df_state = filing_store.load(table, columns=[*table.pk, "retry"], where={"ticker": "AAA"}, markers=True)
        assert df_state is not None and df_state.iloc[0]["retry"] == 1
        assert filing_store.distinct(table, "accession_number", where={"ticker": "AAA"}) == ["empty"]
    print("SANITY: all four empty sections keep non-null keys and retry=1; ordinary reads hide them while resume sees them.")


def test_counter_patch_keeps_the_maximum_attempt_and_only_owned_fields() -> None:
    stamp: dict[str, object] = {"ticker": "AAA", "cik": "1", "accession_number": "a", "as_of": "2020-01-01"}
    df_parent = reservation_frame(Tables.def14a_llm, stamp, None, 1)
    assert set(df_parent.columns) == {"ticker", "accession_number", "as_of", "retry"}
    df_saved = pd.DataFrame([{**stamp, "name": "A Person", "retry": 2, "fees_earned": 5000.0}])
    df_patch = reservation_frame(Tables.def14a_director_comp, stamp, df_saved, 1)
    assert set(df_patch.columns) == {*Tables.def14a_director_comp.pk, "as_of", "retry"}
    assert df_patch.iloc[0]["retry"] == 2
    print("SANITY: parent patches cannot introduce child-only cik; repeated reservations retain 2 and omit economic facts.")
