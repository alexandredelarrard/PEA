"""The per-table resume contract and empty-filing marker declared in `schema.py`."""

from __future__ import annotations

import re
from pathlib import Path

from src.data_store import ddl
from src.data_store.schema import (
    KIND_EXTRACT,
    RESUME_ARCHIVE,
    RESUME_DOCUMENTS,
    RESUME_MARKET,
    RESUME_SERIES,
    RESUME_SNAPSHOT,
    Tables,
    by_kind,
    marker_tables,
    resume_tables,
)

_SCHEMA_SQL = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"

# Extract tables no fetcher resumes on its own: snapshots, rebuilt derivations, rows written in
# the same unit as a parent table, and caches derived from another table.
SIDE_TABLES = frozenset(
    {
        Tables.fundamentals_history.name,
        Tables.fundamentals_history_sec.name,
        Tables.fundamentals_reason_codes.name,
        Tables.cusip_ticker_map.name,
        Tables.insider_footnotes.name,
        Tables.insider_transactions_quarantine.name,
        Tables.insider_transactions_live_coverage.name,
        Tables.sec_13d_transactions.name,
        # children flattened out of each `def14a_llm` answer
        Tables.def14a_directors.name,
        Tables.def14a_executive_comp.name,
        Tables.def14a_director_comp.name,
        Tables.def14a_ownership.name,
        # derived caches of stored transcripts, notes and descriptions
        Tables.earnings_call_sentiment.name,
        Tables.earning_calls_embedding.name,
        Tables.notes_embedding.name,
        Tables.ticker_descriptions.name,
        Tables.ticker_embeddings.name,
    }
)
_MODES = {RESUME_SERIES, RESUME_DOCUMENTS, RESUME_ARCHIVE, RESUME_MARKET, RESUME_SNAPSHOT}
_NUMERIC_SQL = {"BIGINT", "DOUBLE PRECISION"}


def _column_types() -> dict[str, dict[str, str]]:
    """{table: {column: SQL type}} parsed from the hand-maintained `sql/schema.sql`."""
    blocks = ddl.existing_blocks(_SCHEMA_SQL.read_text(encoding="utf-8"))
    pattern = re.compile(r'^\s+"([^"]+)" ([A-Z ]+?)(?: NOT NULL)?,?$', re.MULTILINE)
    return {name: dict(pattern.findall(block)) for name, block in blocks.items()}


def test_every_extract_table_resumes_or_is_a_side_table() -> None:
    extract = {t.name for t in by_kind(KIND_EXTRACT)}
    with_contract = {t.name for t in resume_tables()}
    uncovered = extract - with_contract - SIDE_TABLES
    stale_side = SIDE_TABLES & with_contract
    assert not uncovered, f"extract tables with neither a Resume contract nor a side-table entry: {sorted(uncovered)}"
    assert not stale_side, f"side tables that now declare a contract: {sorted(stale_side)}"
    assert all(t.resume is not None and t.resume.mode in _MODES for t in resume_tables())
    assert all(t.resume is not None and t.resume.overlap_days >= 0 for t in resume_tables())

    print("\n=== SANITY CHECK: resume contract coverage ===")
    print(f"  {len(extract)} extract tables = {len(with_contract)} with a contract + {len(extract - with_contract)} side tables. Validated.")


def test_every_resumed_table_has_a_cadence() -> None:
    missing = sorted(t.name for t in resume_tables() if t.freshness is None)
    assert not missing, f"resumed tables with no freshness cadence: {missing}"
    for table in (
        Tables.dividends,
        Tables.prices_splits,
        Tables.sharadar_actions,
        Tables.sharadar_sp500,
        Tables.sec_13d,
        Tables.sec_13g,
        Tables.sec_8k,
        Tables.sec_8k_votes,
    ):
        assert table.freshness == "daily", table.name
    assert Tables.filing_risk_text.freshness == "quarterly"
    assert Tables.def14a_edgar.freshness == "yearly"

    print("\n=== SANITY CHECK: cadences ===")
    print(f"  all {len(resume_tables())} resumed tables carry a freshness cadence (10 newly declared). Validated.")


def test_contract_and_marker_columns_exist_with_compatible_types() -> None:
    types = _column_types()
    for table in resume_tables():
        resume = table.resume
        assert resume is not None
        cols = types[table.name]
        for col in (resume.key, resume.frontier_col, resume.period_col):
            assert col is None or col in cols, f"{table.name}: contract column {col!r} is not in sql/schema.sql"
    for table in marker_tables():
        assert table.empty_marker is not None
        col, sentinel = table.empty_marker
        sql_type = types[table.name].get(col)
        assert sql_type is not None, f"{table.name}: marker column {col!r} is not in sql/schema.sql"
        if isinstance(sentinel, str):
            assert sql_type == "TEXT", f"{table.name}.{col} is {sql_type}, sentinel {sentinel!r} needs TEXT"
        else:
            assert isinstance(sentinel, int | float) and sql_type in _NUMERIC_SQL, f"{table.name}.{col} is {sql_type}, sentinel {sentinel!r}"
    assert "added_on" in types[Tables.sp500_tickers.name] and types[Tables.sp500_tickers.name]["added_on"] == "DATE"

    print("\n=== SANITY CHECK: contract columns in the DDL ===")
    for table in marker_tables():
        assert table.empty_marker is not None
        print(f"  {table.name}.{table.empty_marker[0]} {types[table.name][table.empty_marker[0]]} <- {table.empty_marker[1]!r}")
    print("  every key/frontier/period/marker column exists, sentinels fit their SQL type, sp500_tickers.added_on is DATE. Validated.")


def test_sequence_sentinels_cannot_collide_with_real_rows() -> None:
    markers = {t.name: t.empty_marker for t in marker_tables()}
    assert markers[Tables.sec_13d.name] == ("rp_seq", -1)
    assert markers[Tables.sec_13g.name] == ("rp_seq", -1)
    for name, marker in markers.items():
        assert marker is not None
        col, sentinel = marker
        if col in {"rp_seq", "trade_seq"}:
            assert isinstance(sentinel, int) and sentinel < 0, f"{name}.{col} counts from 0, so its sentinel must be negative"
        if col == "proposal_seq":
            assert sentinel == 0, f"{name}.proposal_seq counts from 1, so its sentinel is 0"

    print("\n=== SANITY CHECK: sequence sentinels ===")
    print("  rp_seq -> -1 (sequence starts at 0), proposal_seq -> 0 (starts at 1): no marker can overwrite a real row. Validated.")
