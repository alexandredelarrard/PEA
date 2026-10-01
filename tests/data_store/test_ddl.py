"""Focused contracts for registry-directed SQL type and bootstrap DDL generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_store import ddl
from src.data_store.schema import Tables


def test_available_at_is_a_registry_directed_sql_date():
    dtype = pd.Series(["2026-08-31"], dtype="string").dtype
    for table in (Tables.notes_num, Tables.notes_text, Tables.pension_facts):
        assert ddl.sql_type("available_at", dtype, table) == "DATE"

    print("\n=== SANITY CHECK: registry date type ===")
    print("  available_at is forced to SQL DATE for notes and pension facts. Validated.")


def test_pension_facts_ddl_retains_zip_vintages() -> None:
    table = Tables.pension_facts
    assert table.pk == ("cik", "tag", "ddate", "qtrs", "quarter")
    assert table.freshness_date_col == "available_at"
    generated = ddl.table_ddl(
        table,
        [("cik", "TEXT"), ("tag", "TEXT"), ("ddate", "DATE"), ("qtrs", "BIGINT"), ("quarter", "TEXT"), ("available_at", "DATE")],
    )
    assert '"available_at" DATE' in generated
    assert 'PRIMARY KEY ("cik", "tag", "ddate", "qtrs", "quarter")' in generated
    schema_sql = Path(__file__).resolve().parents[2] / "sql/schema.sql"
    persisted = ddl.existing_blocks(schema_sql.read_text(encoding="utf-8"))[table.name]
    assert '"quarter" TEXT NOT NULL' in persisted
    assert '"available_at" DATE' in persisted
    assert 'PRIMARY KEY ("cik", "tag", "ddate", "qtrs", "quarter")' in persisted
    print("\n=== SANITY CHECK: pension archive-vintage table grain ===")
    print("  quarter is in the primary key and available_at is a DATE freshness clock. Validated.")


def test_generated_schema_overlays_missing_registry_date_columns():
    reflected = {
        table.name: [
            ("adsh", "TEXT"),
            ("tag", "TEXT"),
            ("ddate", "DATE"),
            ("qtrs", "BIGINT"),
            ("filed", "DATE"),
        ]
        for table in (Tables.notes_num, Tables.notes_text)
    }

    generated = ddl.generate_schema_sql(reflected)
    blocks = ddl.existing_blocks(generated)
    for table in (Tables.notes_num, Tables.notes_text):
        assert '"available_at" DATE' in blocks[table.name]

    print("\n=== SANITY CHECK: generated notes DDL ===")
    print("  live reflection may lag, but both generated table blocks still include registry-declared available_at DATE. Validated.")
