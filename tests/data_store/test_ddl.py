"""Focused contracts for registry-directed SQL type and bootstrap DDL generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_store import ddl
from src.data_store.schema import MANAGED, Tables


def test_available_at_is_a_registry_directed_sql_date():
    dtype = pd.Series(["2026-08-31"], dtype="string").dtype
    for table in (Tables.notes_num, Tables.notes_text):
        assert ddl.sql_type("available_at", dtype, table) == "DATE"

    print("\n=== SANITY CHECK: registry date type ===")
    print("  available_at is forced to SQL DATE for the notes tables. Validated.")


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
    "universe_cusip": (
        ("security_id", "source", "source_symbol", "valid_from"),
        ('"valid_from" DATE NOT NULL', '"valid_to" DATE', '"conversion_ratio" DOUBLE PRECISION', '"issuer_cik" TEXT', '"scope_changed_at" TIMESTAMP'),
    ),
    "sec_company_tickers": (("cik", "ticker"), ('"cik" TEXT NOT NULL', '"exchange" TEXT', '"fetched_at" TIMESTAMP')),
    "sec_fails_to_deliver_security": (
        ("cusip", "date"),
        ('"date" DATE NOT NULL', '"trade_date" DATE', '"price" DOUBLE PRECISION', '"description" TEXT', '"period" TEXT'),
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
    assert "universe_cik_symbol" in blocks and "sec_fails_to_deliver" not in blocks and "ix_universe_cik_symbol_company_id" in schema_sql
    assert '"ticker"' not in blocks["sec_fails_to_deliver_security"], "the FTD lines are stored as filed: no identity stamp"
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
    ):
        assert column in persisted, column
    assert '"ticker"' not in persisted and '"security_class"' not in persisted, "the FINRA lines are stored as filed: no identity stamp"
    assert "sec_short_interest" not in blocks and "ix_sec_short_volume_security_ticker" not in schema_sql
    print("\n=== SANITY CHECK: Q2c DDL ===")
    print("  sec_short_volume_security registered with PK (source_symbol, date), source columns only; sec_short_interest removed")


def test_edgar_text_tables_are_registered_and_spliced() -> None:
    schema_sql = (Path(__file__).resolve().parents[2] / "sql/schema.sql").read_text(encoding="utf-8")
    blocks = ddl.existing_blocks(schema_sql)
    for table in (Tables.sec_employees_text, Tables.sec_def14a_text):
        df = pd.DataFrame({c: [1] if c == "n_chars" else ["x"] for c in table.retrieval_columns})
        assert blocks[table.name] == ddl.table_ddl_from_frame(table, df), table.name
    assert {"sec_filing_text", "sec_8k"} <= set(blocks)
    print("\n=== SANITY CHECK: EDGAR text table DDL ===")
    print(
        "  sec_employees_text and sec_def14a_text blocks equal the generator's output (DATE stamps, BIGINT n_chars, filing_date index); neighbours kept"
    )


def test_price_owner_windows_is_registered_and_spliced() -> None:
    schema_sql = (Path(__file__).resolve().parents[2] / "sql/schema.sql").read_text(encoding="utf-8")
    blocks = ddl.existing_blocks(schema_sql)
    table = Tables.price_owner_windows
    assert table.pk == ("ticker", "valid_from") and table.date_type_cols == ("valid_from", "valid_to")
    persisted = blocks[table.name]
    assert persisted == ddl.table_ddl(table, [(c, "DATE" if c in table.date_type_cols else "TEXT") for c in table.read_columns])
    assert "universe_tickers" in blocks and "ix_universe_cik_symbol_company_id" in schema_sql
    print("\n=== SANITY CHECK: owner table DDL ===")
    print("  price_owner_windows registered with PK (ticker, valid_from), DATE windows, and its spliced block equals the generator's output")


#: The four fundamentals tables: source-prefixed name -> its PK and the indexes the registry DDL gives it.
FUNDAMENTALS_TABLES = {
    "sec_fundamentals_facts": (("ticker", "accession_number", "field", "duration_type", "period_end"), ("ix_sec_fundamentals_facts_filing_date",)),
    "sec_fundamentals_history": (("ticker", "as_of"), ()),
    "sharadar_fundamentals_facts": (("ticker", "dimension", "date", "reportperiod"), ()),
    "sharadar_fundamentals_history": (("ticker", "as_of"), ()),
}


def test_fundamentals_tables_own_their_ddl_and_stay_out_of_schema_sql() -> None:
    """D18: no `sql/schema.sql` block for the four; the pipeline creates them (`managed=False`), and regenerating the
    file from a database that holds them still emits no block, while every other block is carried over."""
    schema_sql = (Path(__file__).resolve().parents[2] / "sql/schema.sql").read_text(encoding="utf-8")
    blocks = ddl.existing_blocks(schema_sql)
    names = set(FUNDAMENTALS_TABLES)
    for name, (pk, indexes) in FUNDAMENTALS_TABLES.items():
        table = getattr(Tables, name)
        assert table.name == name and table.pk == pk and not table.managed, (name, table.pk, table.managed)
        assert table not in MANAGED and name not in schema_sql, name
        created = ddl.table_ddl_from_frame(table, pd.DataFrame(columns=list(pk)))
        assert f"PRIMARY KEY ({', '.join(chr(34) + c + chr(34) for c in pk)})" in created, created
        assert all(f'"{c}" ' in line and line.endswith("NOT NULL,") for c, line in zip(pk, created.splitlines()[1:], strict=False)), created
        assert [line.split()[5] for line in created.splitlines() if line.startswith("CREATE INDEX")] == list(indexes), created
    reflected = {name: [("ticker", "TEXT")] for name in names}
    regenerated = ddl.existing_blocks(ddl.generate_schema_sql(reflected, previous=schema_sql))
    assert not names & set(regenerated) and set(blocks) <= set(regenerated), names & set(regenerated)
    old = ("fundamentals_facts", "fundamentals_history_sec", "fundamentals_history", "fundamentals_sharadar", "sharadar_fundamentals")
    assert not [n for n in old if hasattr(Tables, n) or n in blocks], [n for n in old if hasattr(Tables, n) or n in blocks]
    print("\n=== SANITY CHECK: fundamentals tables own their DDL ===")
    print(
        f"  {', '.join(FUNDAMENTALS_TABLES)}: managed=False, PKs as declared (NOT NULL), indexes {[i for _, ix in FUNDAMENTALS_TABLES.values() for i in ix]}"
    )
    print(f"  from the registry DDL; none in sql/schema.sql, a regeneration emits none and keeps all {len(blocks)} other blocks. Validated.")
