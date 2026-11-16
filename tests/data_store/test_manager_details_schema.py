"""Contract of the `sec_13f_manager_details` registry entry and its spliced DDL block."""

from __future__ import annotations

from pathlib import Path

from src.data_store import ddl
from src.data_store.schema import Resume, Tables

ROOT = Path(__file__).resolve().parents[2]

EXTRACT_COLUMNS = {
    "cik": "TEXT NOT NULL",
    "period": "DATE NOT NULL",
    "manager_name": "TEXT",
    "state_or_country": "TEXT",
    "report_type": "TEXT",
    "sec_file_number": "TEXT",
    "accession_number": "TEXT",
    "first_filed_date": "DATE",
    "source": "TEXT",
    "source_period": "DATE",
}
SECTORS = (
    "basic_materials",
    "communication_services",
    "consumer_cyclical",
    "consumer_defensive",
    "energy",
    "financial_services",
    "healthcare",
    "industrials",
    "real_estate",
    "technology",
    "utilities",
    "unmapped",
)
PROFILE_COLUMNS = {
    "category": "TEXT",
    "n_positions": "DOUBLE PRECISION",
    "hhi": "DOUBLE PRECISION",
    "turnover": "DOUBLE PRECISION",
    "new_position_rate": "DOUBLE PRECISION",
    "pct_equity_us": "DOUBLE PRECISION",
    "pct_equity_foreign": "DOUBLE PRECISION",
    "pct_etf": "DOUBLE PRECISION",
    "pct_bond": "DOUBLE PRECISION",
    "pct_options": "DOUBLE PRECISION",
    "pct_other": "DOUBLE PRECISION",
    "pct_cap_small": "DOUBLE PRECISION",
    "pct_cap_mid": "DOUBLE PRECISION",
    "pct_cap_large": "DOUBLE PRECISION",
    **{f"pct_sector_{name}": "DOUBLE PRECISION" for name in SECTORS},
    "active_share": "DOUBLE PRECISION",
    "book_value_usd": "DOUBLE PRECISION",
    "mapped_share": "DOUBLE PRECISION",
    "n_quarters_used": "BIGINT",
}


def test_registry_lists_the_manager_details_table() -> None:
    table = Tables.sec_13f_manager_details
    assert table.name == "sec_13f_manager_details" and table.pk == ("cik", "period")
    assert table.kind == "extract" and table.managed
    assert table.date_col == "period" and table.ticker_col is None
    assert table.date_type_cols == ("period", "first_filed_date", "source_period")
    assert table.freshness == "quarterly" and table.freshness_col == "first_filed_date"
    assert table.resume == Resume("market", "cik", "first_filed_date", 92, source_start="1999-07-01")
    print("\n=== SANITY CHECK: registry entry ===")
    print("  sec_13f_manager_details: pk (cik, period), extract, quarterly freshness on first_filed_date, market resume. Validated.")


def test_every_documented_column_is_in_the_spliced_ddl_and_the_schema_comment() -> None:
    schema_sql = (ROOT / "sql/schema.sql").read_text(encoding="utf-8")
    block = ddl.existing_blocks(schema_sql)["sec_13f_manager_details"]
    comment_source = (ROOT / "src/data_store/schema.py").read_text(encoding="utf-8")
    columns = {**EXTRACT_COLUMNS, **PROFILE_COLUMNS}
    for name, sql in columns.items():
        assert f'"{name}" {sql}' in block, name
        documented = f"`{name}`" in comment_source or (name.startswith("pct_sector_") and "`pct_sector_<sector>`" in comment_source)
        assert documented, name
    assert 'PRIMARY KEY ("cik", "period")' in block
    declared = [line.split('"')[1] for line in block.splitlines() if line.startswith('    "')]
    assert declared == list(columns), "the block holds exactly the documented columns, extract first"
    assert "ix_sec_13f_manager_details" not in block, "period is in the PK and there is no ticker: no index"
    print("\n=== SANITY CHECK: spliced DDL ===")
    print(f"  {len(EXTRACT_COLUMNS)} extract + {len(PROFILE_COLUMNS)} profile columns all in the spliced block with the right types. Validated.")
