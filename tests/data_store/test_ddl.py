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


#: Q2a tables: the security master, the SEC current-tickers snapshot and the per-security FTD rows.
Q2A_TABLES = {
    "security_master": (
        ("security_id", "source", "source_symbol", "valid_from"),
        ('"valid_from" DATE NOT NULL', '"valid_to" DATE', '"conversion_ratio" DOUBLE PRECISION', '"issuer_cik" TEXT', '"scope_changed_at" TIMESTAMP'),
    ),
    "sec_company_tickers": (("cik", "ticker"), ('"cik" TEXT NOT NULL', '"exchange" TEXT', '"fetched_at" TIMESTAMP')),
    "sec_fails_to_deliver_security": (
        ("cusip", "date"),
        ('"date" DATE NOT NULL', '"trade_date" DATE', '"price" DOUBLE PRECISION', '"description" TEXT', '"lineage_role" TEXT'),
    ),
}


def test_q2a_tables_are_registered_and_spliced_into_schema_sql() -> None:
    schema_sql = (Path(__file__).resolve().parents[2] / "sql/schema.sql").read_text(encoding="utf-8")
    blocks = ddl.existing_blocks(schema_sql)
    for name, (pk, columns) in Q2A_TABLES.items():
        table = getattr(Tables, name)
        assert table.name == name and table.pk == pk, (name, table.pk)
        persisted = blocks[name]
        assert f"PRIMARY KEY ({', '.join(chr(34) + c + chr(34) for c in pk)})" in persisted, persisted
        for column in columns:
            assert column in persisted, (name, column)
        generated = ddl.table_ddl(table, [(c, "TEXT") for c in table.read_columns])
        assert generated.split("(", 1)[0] == persisted.split("(", 1)[0]
    assert "entity_lineage" in blocks and "sec_fails_to_deliver" in blocks and "ix_entity_lineage_entity_id" in schema_sql
    print("\n=== SANITY CHECK: Q2a DDL ===")
    print(f"  {', '.join(Q2A_TABLES)} registered with their PKs and hand-spliced into sql/schema.sql; existing blocks and indexes kept")


def test_q2c_short_volume_security_table_is_registered_and_spliced() -> None:
    schema_sql = (Path(__file__).resolve().parents[2] / "sql/schema.sql").read_text(encoding="utf-8")
    blocks = ddl.existing_blocks(schema_sql)
    table = Tables.sec_short_volume_security
    assert table.name == "sec_short_volume_security" and table.pk == ("source_symbol", "date")
    persisted = blocks[table.name]
    assert 'PRIMARY KEY ("source_symbol", "date")' in persisted
    for column in (
        '"date" DATE NOT NULL',
        '"source_symbol" TEXT NOT NULL',
        '"market" TEXT',
        '"short_exempt_volume" DOUBLE PRECISION',
        '"security_class" TEXT',
    ):
        assert column in persisted, column
    assert "sec_short_interest" in blocks and "ix_sec_short_volume_security_ticker" in schema_sql
    print("\n=== SANITY CHECK: Q2c DDL ===")
    print("  sec_short_volume_security registered with PK (source_symbol, date) and hand-spliced; sec_short_interest's block kept")
