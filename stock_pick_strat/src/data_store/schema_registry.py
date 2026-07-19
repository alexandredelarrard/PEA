"""
schema_registry.py  (src/data_store/schema_registry.py)
--------------------------------------------------------
Single source of truth mapping each logical DB table to:
  * its source flat-file under data/ (parquet or csv) — used by the migrator,
  * its PRIMARY KEY columns — used both to generate DDL and to upsert/merge,
  * the incremental time column (`date_col`) — used to fetch only missing dates,
  * optional column-type overrides and a vector-collapse column.

Both the schema-SQL generator (schema_sql.py) and the parquet->DB migrator
(scripts/migrate_parquet_to_db.py) import from here so the DDL, the merge keys
and the incremental logic never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSpec:
    name: str                         # DB table name
    source: str                       # path relative to the data/ dir
    pk: tuple[str, ...]               # PRIMARY KEY columns (merge key)
    kind: str                         # 'reference' | 'extract' | 'aggregate'
    date_col: str | None = None       # incremental time column (per-ticker max)
    ticker_col: str | None = "ticker" # entity column, None if table has no ticker
    date_type_cols: tuple[str, ...] = ()   # cols to force to SQL DATE (string dates)
    # collapse many numeric columns (prefix e0..eN) into one float8[] array column
    vector_col: str | None = None
    vector_prefix: str | None = None


# --------------------------------------------------------------------------- #
# Reference / dimension table: the ticker universe. Carries ticker, name, cik and
# GICS sector / industry_group / sub_industry — the single source of truth for
# ticker->CIK resolution too (see sec_utils.load_cik_mapping); the old separate
# `cik_mapping` table was dropped as a redundant duplicate.
# --------------------------------------------------------------------------- #
REFERENCE_TABLES: list[TableSpec] = [
    TableSpec("sp500_tickers", "sp500_tickers.csv", ("ticker",), "reference",
              date_col=None),
]

# --------------------------------------------------------------------------- #
# Extracted (raw) tables                                                       #
# --------------------------------------------------------------------------- #
EXTRACT_TABLES: list[TableSpec] = [
    TableSpec("prices", "prices.parquet", ("ticker", "date"), "extract",
              date_col="date"),
    TableSpec("dividends", "dividends.parquet", ("ticker", "date"), "extract",
              date_col="date"),
    TableSpec("short_interest", "short_interest.parquet", ("ticker", "date"), "extract",
              date_col="date"),
    TableSpec("fundamentals_history", "fundamentals_history.parquet",
              ("ticker", "as_of"), "extract", date_col="as_of",
              date_type_cols=("as_of", "fiscal_end")),
    TableSpec("earnings_surprises", "earnings_surprises.parquet",
              ("ticker", "earnings_date"), "extract", date_col="earnings_date"),
    TableSpec("macro", "macro.parquet", ("date",), "extract",
              date_col="date", ticker_col=None),
    TableSpec("employees_history", "employees_history.parquet",
              ("ticker", "as_of"), "extract", date_col="as_of"),
    TableSpec("sec_filings_index", "sec_filings_index.parquet",
              ("ticker", "accession_number"), "extract", date_col="filing_date"),
    TableSpec("google_trends", "google_trends.parquet", ("ticker", "date"), "extract",
              date_col="date"),
    TableSpec("wiki_pageviews", "wiki_pageviews.parquet", ("ticker", "date"), "extract",
              date_col="date"),
    TableSpec("ticker_embeddings", "ticker_embeddings.parquet", ("ticker",), "extract",
              date_col=None, vector_col="embedding", vector_prefix="e"),
    # tables with no seed file yet (created on first fetcher write via ensure_table)
    TableSpec("cusip_ticker_map", "cusip_ticker_map.parquet", ("cusip",), "extract",
              date_col=None, ticker_col="ticker"),
    TableSpec("institutional_holdings", "institutional_holdings.parquet",
              ("cik", "period", "ticker", "cusip"), "extract", date_col="period"),
    TableSpec("def14a_llm", "def14a_llm.parquet", ("ticker", "accession_number"),
              "extract", date_col="as_of"),
    TableSpec("ticker_descriptions", "ticker_descriptions.parquet", ("ticker",),
              "extract", date_col=None),
    # SEC Insider Transactions Data Sets (Forms 3/4/5): one row per reported
    # transaction (non-derivative + derivative), keyed by accession + table + SK.
    TableSpec("insider_transactions", "insider_transactions.parquet",
              ("accession_number", "security_type", "transaction_sk"), "extract",
              date_col="transaction_date",
              date_type_cols=("transaction_date", "filing_date", "period_of_report")),
    # SEC Financial Statement Data Sets (num/sub): curated pension facts per
    # company/tag/period-end (`ddate`) / duration (`qtrs`).
    TableSpec("pension_facts", "pension_facts.parquet",
              ("cik", "tag", "ddate", "qtrs"), "extract", date_col="ddate",
              date_type_cols=("ddate", "filed")),
]

# --------------------------------------------------------------------------- #
# Aggregated (output) tables                                                   #
# --------------------------------------------------------------------------- #
AGGREGATE_TABLES: list[TableSpec] = [
    TableSpec("cube", "output/cube.parquet",
              ("ticker", "date", "target_horizon"), "aggregate", date_col="date"),
    TableSpec("cube_signal", "output/cube_signal.parquet",
              ("ticker", "date"), "aggregate", date_col="date"),
    TableSpec("predictions", "output/predictions.parquet",
              ("ticker", "date"), "aggregate", date_col="date"),
]

ALL_TABLES: list[TableSpec] = REFERENCE_TABLES + EXTRACT_TABLES + AGGREGATE_TABLES
BY_NAME: dict[str, TableSpec] = {t.name: t for t in ALL_TABLES}
