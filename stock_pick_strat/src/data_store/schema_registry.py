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
    # SEC Fails-to-Deliver: settlement fails per ticker x date (same grain as
    # short_interest but a separate table -> its semi-monthly, ~2-month-lagged files
    # don't pollute short_interest's global-max-date incremental; combined at the
    # feature layer). `period` = source semi-monthly file tag (for incremental skip).
    TableSpec("fails_to_deliver", "fails_to_deliver.parquet", ("ticker", "date"), "extract",
              date_col="date", date_type_cols=("date",)),
    TableSpec("fundamentals_history", "fundamentals_history.parquet",
              ("ticker", "as_of"), "extract", date_col="as_of",
              date_type_cols=("as_of", "fiscal_end")),
    TableSpec("earnings_surprises", "earnings_surprises.parquet",
              ("ticker", "earnings_date"), "extract", date_col="earnings_date"),
    TableSpec("macro", "macro.parquet", ("date",), "extract",
              date_col="date", ticker_col=None),
    # Long-history multi-asset ALLOCATION series (FRED, since ~1995): one row per
    # date, no ticker. Total-return / level legs (equity_tr [Wilshire 5000], gold,
    # bond_10y_tr [reconstructed from the 10Y yield], cash_rate, fx_usdeur) for the
    # risk-parity + trend allocation sleeve. See fetch_macro_assets.py.
    TableSpec("macro_asset_prices", "macro_asset_prices.parquet", ("date",), "extract",
              date_col="date", ticker_col=None, date_type_cols=("date",)),
    TableSpec("employees_history", "employees_history.parquet",
              ("ticker", "as_of"), "extract", date_col="as_of"),
    TableSpec("google_trends", "google_trends.parquet", ("ticker", "date"), "extract",
              date_col="date"),
    TableSpec("wiki_pageviews", "wiki_pageviews.parquet", ("ticker", "date"), "extract",
              date_col="date"),
    TableSpec("ticker_embeddings", "ticker_embeddings.parquet", ("ticker",), "extract",
              date_col=None, vector_col="embedding", vector_prefix="e"),
    # OpenAI earnings-call embeddings: one row PER SPEAKER TURN (question / answer / prepared),
    # each with its own float8[] `embedding` + raw `text` + `person` + `tag` + `exchange_idx`
    # (links a question to its answer turns) + model/run stamp + `as_of` (call date). The
    # Q&A-coherence + QoQ-distance cube features are DERIVED from these turns at build time (cos
    # of question vs its answers, pooled per section for drift). See
    # src/data_aggregate/utils/earnings_call_embeddings.py.
    TableSpec("earning_calls_embedding", "earning_calls_embedding.parquet",
              ("ticker", "quarter", "seq"), "extract", date_col=None, ticker_col="ticker",
              vector_col="embedding", vector_prefix="e"),
    # OpenAI embeddings of SEC footnote NARRATIVE (`notes_text`): one mean-pooled vector PER
    # (ticker, filing `adsh`, TextBlock `tag`), with its `theme`, `filed`/`ddate` for point-in-time
    # ordering, `txtlen` and chunk count. The narrative-drift / risk-anchor / disclosure-length
    # cube features are DERIVED from these per filing at build time. See
    # src/data_aggregate/utils/notes_features.py.
    TableSpec("notes_embedding", "notes_embedding.parquet",
              ("ticker", "adsh", "tag"), "extract", date_col=None, ticker_col="ticker",
              date_type_cols=("as_of", "ddate"), vector_col="embedding", vector_prefix="e"),
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
    # SEC Financial Statement AND NOTES Data Sets — footnote NUMERIC facts
    # (consolidated / undimensioned, curated tag set: PBO, plan assets, funded
    # status, service cost, employer contributions, discount rate). Grain = one
    # fact per filing (`adsh`) / tag / period-end (`ddate`) / duration (`qtrs`).
    # `period` = source zip tag (quarterly "YYYYqQ" or monthly "YYYY_MM") for
    # incremental skip.
    TableSpec("notes_num", "notes_num.parquet",
              ("adsh", "tag", "ddate", "qtrs"), "extract", date_col="ddate",
              date_type_cols=("ddate", "filed")),
    # SEC notes NARRATIVE TEXT blocks (high-signal notes only), stored raw for
    # later embedding / sentiment. Same grain as notes_num; `value` is the text.
    TableSpec("notes_text", "notes_text.parquet",
              ("adsh", "tag", "ddate", "qtrs"), "extract", date_col="ddate",
              date_type_cols=("ddate", "filed")),
    # FREE earnings-call transcripts (Motley Fool), split into high-signal sections
    # (prepared_remarks / qa / participants) for later sentiment / embedding. One row
    # per ticker / fiscal quarter / section; `as_of` = call date, `text` = the prose.
    TableSpec("earnings_call_sections", "earnings_call_sections.parquet",
              ("ticker", "quarter", "tag"), "extract", date_col="as_of",
              date_type_cols=("as_of",)),
    # Per-call sentiment / text-metrics cache (FinBERT-tone + LM lexicon), one row per
    # ticker / fiscal quarter / section. Holds the EXPENSIVE, call-intrinsic scores
    # (tone probs, word count, uncertainty ratio) so the GPU pass runs once; the
    # cross-call KPIs are derived cheaply at cube-build time. Same grain as sections.
    TableSpec("earnings_call_sentiment", "earnings_call_sentiment.parquet",
              ("ticker", "quarter", "tag"), "extract", date_col="as_of",
              date_type_cols=("as_of",)),
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
    # Multi-asset trend sleeve daily NET returns (one row per date, no ticker) — a directional
    # cross-asset time-series-momentum book blended with the equity alpha + SPY in the backtest.
    TableSpec("trend_asset_returns", "output/trend_asset_returns.parquet",
              ("date",), "aggregate", date_col="date", ticker_col=None),
]

ALL_TABLES: list[TableSpec] = REFERENCE_TABLES + EXTRACT_TABLES + AGGREGATE_TABLES
BY_NAME: dict[str, TableSpec] = {t.name: t for t in ALL_TABLES}
