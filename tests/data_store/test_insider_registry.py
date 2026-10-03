"""Registry and DDL contract of the single `insider_transactions` table."""

from __future__ import annotations

import re
from pathlib import Path

from src.data_store import ddl
from src.data_store.schema import Tables

INSIDER_COLUMNS = (
    "accession_number",
    "security_type",
    "row_sequence",
    "source",
    "ticker",
    "issuer_cik",
    "issuer_name",
    "owner_cik",
    "owner_name",
    "owner_ciks",
    "n_reporting_owners",
    "is_director",
    "is_officer",
    "is_ten_pct_owner",
    "is_other",
    "officer_title",
    "document_type",
    "original_submission_date",
    "transaction_date",
    "filing_date",
    "period_of_report",
    "security_title",
    "transaction_code",
    "acquired_disposed",
    "shares",
    "price_per_share",
    "value_usd",
    "shares_owned_after",
    "direct_indirect",
    "quarter",
    "is_10b5_1",
    "transaction_form_type",
    "equity_swap_involved",
    "deemed_execution_date",
    "nature_of_ownership",
    "transaction_timeliness",
    "exercise_price",
    "exercise_date",
    "expiration_date",
    "underlying_security_title",
    "underlying_shares",
    "underlying_value",
    "footnote_ids",
    "acceptance_datetime",
    "fetched_at",
)
DATE_COLUMNS = (
    "transaction_date",
    "filing_date",
    "period_of_report",
    "deemed_execution_date",
    "exercise_date",
    "expiration_date",
    "original_submission_date",
)
PK = ("accession_number", "security_type", "row_sequence")
NEW_READ_COLUMNS = ("owner_ciks", "n_reporting_owners", "original_submission_date", "document_type", "source", "row_sequence")
OLD_READ_COLUMNS = (
    "accession_number",
    "ticker",
    "owner_cik",
    "owner_name",
    "filing_date",
    "transaction_date",
    "transaction_code",
    "shares",
    "price_per_share",
    "value_usd",
    "shares_owned_after",
    "security_type",
    "security_title",
    "direct_indirect",
    "officer_title",
    "is_director",
    "is_officer",
    "is_ten_pct_owner",
    "is_10b5_1",
)
_COLUMN_RE = re.compile(r'^\s+"(?P<name>[a-z0-9_]+)" (?P<type>[A-Z ]+?)(?: NOT NULL)?,?$', re.M)


def test_insider_transactions_registry_is_the_single_table_contract() -> None:
    table = Tables.insider_transactions
    assert table.pk == PK
    assert table.date_col == "transaction_date"
    assert set(table.date_type_cols) == set(DATE_COLUMNS)
    assert table.freshness == "daily"
    assert table.freshness_date_col == "filing_date"
    assert table.read_columns == OLD_READ_COLUMNS + NEW_READ_COLUMNS
    assert "transaction_sk" not in table.read_columns
    assert set(table.read_columns) <= set(INSIDER_COLUMNS)
    assert Tables.insider_transactions_live.read_columns == OLD_READ_COLUMNS

    print("\n=== SANITY: insider_transactions registry ===")
    print(f"  pk {table.pk}, freshness {table.freshness} on {table.freshness_date_col}")
    print(f"  {len(table.date_type_cols)} DATE columns incl. original_submission_date; {len(table.read_columns)} read columns, no transaction_sk")
    print("  SANITY: the registry declares the single-table key and column contract.")


def test_insider_transactions_schema_sql_matches_the_contract() -> None:
    schema_sql = Path(__file__).resolve().parents[2] / "sql/schema.sql"
    block = ddl.existing_blocks(schema_sql.read_text(encoding="utf-8"))["insider_transactions"]
    columns = {m.group("name"): m.group("type") for m in _COLUMN_RE.finditer(block)}

    assert tuple(columns) == INSIDER_COLUMNS
    assert {c for c, t in columns.items() if t == "DATE"} == set(DATE_COLUMNS)
    assert columns["acceptance_datetime"] == columns["fetched_at"] == "TIMESTAMP"
    assert columns["row_sequence"] == columns["n_reporting_owners"] == "BIGINT"
    assert 'PRIMARY KEY ("accession_number", "security_type", "row_sequence")' in block
    assert all(f'"{c}" {columns[c]} NOT NULL' in block for c in PK)
    assert "transaction_sk" not in block

    print("\n=== SANITY: sql/schema.sql insider_transactions block ===")
    print(f"  {len(columns)} columns in contract order, {len(DATE_COLUMNS)} DATE, PK {PK}")
    print("  SANITY: the persisted DDL matches the registry contract; transaction_sk is gone.")
