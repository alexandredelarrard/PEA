"""Focused contracts for registry-directed SQL type and bootstrap DDL generation."""

from __future__ import annotations

import pandas as pd

from src.data_store import ddl
from src.data_store.schema import Tables


def test_available_at_is_a_registry_directed_sql_date():
    dtype = pd.Series(["2026-08-31"], dtype="string").dtype
    for table in (Tables.notes_num, Tables.notes_text):
        assert ddl.sql_type("available_at", dtype, table) == "DATE"

    print("\n=== SANITY CHECK: registry date type ===")
    print("  available_at is forced to SQL DATE for both financial-notes tables. Validated.")


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
