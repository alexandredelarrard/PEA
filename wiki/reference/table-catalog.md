---
title: Table catalog
description: Canonical PostgreSQL table grains, temporal semantics, ownership, and lifecycle.
type: reference
tags:
  - wiki
  - reference
  - database
  - schema
---
# Table catalog

## Purpose

This is the canonical map of PostgreSQL table grain, ownership, temporal key, and lifecycle. The executable source of truth is the frozen `Table` registry in [schema.py](../../src/data_store/schema.py); all application access goes through [DataStore](../../src/data_store/store.py). Use this page before changing a table, then inspect the registry declaration and its producer.

> [!IMPORTANT]
> Never introduce a string table name or a parallel `*_TABLE` constant. Use `Tables.<name>`. Never infer availability from a period column when the registry declares a publication-date freshness column.

## Registry contract

| Field | Meaning |
| --- | --- |
| `name`, `pk` | Physical name and upsert/deduplication key. |
| `kind` | DDL grouping: reference, extract, aggregate, or part. It is distinct from the cube part registry's `PartKind`. |
| `date_col` | Default time column for `since`, `until`, `bounds`, and `max_date`. |
| `ticker_col` | Ticker-grain column, or none for market-wide series. |
| `date_type_cols` | Columns forced to SQL `DATE`. |
| `read_columns`, `optional_columns` | The one declaration behind `project=True`; optional fields may be absent in an older live table. |
| `freshness`, `freshness_date_col` | Expected cadence and, when different, the actual publication clock. |
| `vector_col`, `vector_prefix` | Wide embedding columns collapsed to a PostgreSQL `float8[]`. |
| `managed` | Managed tables retain generated DDL on replacement; unmanaged cube parts are dropped and recreated so removed features disappear. |

Derived registry views such as `ALL`, `BY_NAME`, `MANAGED`, `PARTS`, `by_kind()`, and `projection_report()` are computed from the declarations. Adding a managed table means adding one declaration and regenerating [sql/schema.sql](../../sql/schema.sql) through [generate_schema_sql.py](../../scripts/generate_schema_sql.py).

## Reference and identity tables

| Table | Primary grain | Contract |
| --- | --- | --- |
| `sp500_tickers` | ticker | Current research universe and the roster ticker-to-CIK mapping. Universe loading also applies the insufficient-history exclusions in [universe.py](../../src/utils/universe.py). |
| `superinvestor_roster` | snapshot date × Dataroma code | Point-in-time elite-manager roster. The code, not CIK, is the key because manager codes can rename and some managers never file 13F. |
| `symbol_tenure` | symbol × issuer CIK × valid-from | Dated symbol-to-issuer membership. Intervals are half-open and may overlap; callers perform membership tests rather than assuming a unique answer. |
| `entity_lineage` | CIK | Maps legal registrants into economic-company entities. Missing rows mean singleton entities, not unknown identity. |

The identity model has two axes: [registrant policy](../../src/data_extract/utils/common/registrant.py) controls which legal filers are valid for an SEC form, while `symbol_tenure` and `entity_lineage` control market-symbol ownership. See [source availability](../concepts/source-availability.md).

## Prices and market data

| Table | Primary grain | Time key | Important semantics |
| --- | --- | --- | --- |
| `prices` | ticker × date | date | Equity OHLCV only. `close_split` is split-adjusted for levels; `close_total` is total-return adjusted for returns. There is deliberately no ambiguous `close` column. |
| `dividends` | ticker × ex-date | date | Sparse cash distributions; absence is normal for non-payers. |
| `prices_splits` | ticker × ex-date | date | Sparse split/spinoff factors used in price-basis reconciliation. |
| `prices_macro` | named series × date | date | Long-form benchmark, volatility, commodity, energy, rate, credit, breakeven, FX, and derived macro series. These series never enter the equity `prices` cross-section. |
| `short_interest` | ticker × date | date | FINRA RegSHO tape. The source is market-wide per day, so incremental resume is global. |
| `sec_fails_to_deliver` | ticker × date | date | SEC semi-monthly settlement-fail files, kept separate from short volume because cadence and lag differ. |
| `cusip_ticker_map` | CUSIP | none | CUSIP-to-ticker resolution, including curated overrides. |
| `macro` | date | date | Older wide macro table retained in the registry/live database; new macro consumers use `prices_macro`. |

## Fundamentals

| Table | Primary grain | Contract |
| --- | --- | --- |
| `fundamentals_facts` | ticker × accession × field × fiscal labels × duration | Raw per-filing SEC XBRL facts. Originals and amendments coexist, preserving what was knowable on every filing date. |
| `fundamentals_history_sec` | ticker × publication event | Complete SEC replay snapshot at each filing-date event. It is append-only unless an explicit rebuild is requested and is the only fundamentals history consumed by validation. |
| `fundamentals_history` | ticker × as-of date | The merged consumer table. Sharadar owns declared field blocks for all history; SEC contributes a backward-as-of block whose names end in `_sec`. Consumers read this table. |
| `fundamentals_sharadar` | ticker × dimension × filing date × report period | Raw as-reported SF1 layer. It is wide and must be projected. Only AR dimensions are point-in-time. |
| `sharadar_tickers` | vendor table × permaticker × ticker | Mutable entity metadata; refreshed in full. |
| `sharadar_actions` | date × ticker × action × counter-ticker | Market-wide corporate actions. Literal `"N/A"` values in key fields must not be parsed as null. |
| `sharadar_sp500` | date × ticker × action | Historical index membership events. Currently ingested but not yet used to make the research universe point-in-time. |
| `fundamentals_reason_codes` | ticker × as-of × field × reason | Dense explanation for missing or qualified SEC history cells. |
| `fundamentals_employees` | ticker × as-of | Annual headcount parsed from 10-K prose, separated from XBRL history because its failure mode is textual. |
| `earnings_surprises` | ticker × earnings date | Consensus and actual EPS. Future scheduled dates are possible; realized features require a non-null actual. |
| `pension_facts` | CIK × tag × period × duration | Curated pension facts from SEC bulk datasets. Freshness follows filing date, not period date. |
| `notes_num`, `notes_text` | accession × tag × period × duration | Numeric and narrative SEC footnote datasets; freshness follows `filed`. |

The distinction between the two histories is load-bearing: [data extraction](../modules/data-extract.md) builds the SEC replay independently, then the Sharadar producer builds the merged consumer table.

## Ownership, institutional, and event data

| Table | Primary grain | Contract |
| --- | --- | --- |
| `sec13f_hr` | manager CIK × period × ticker × CUSIP | Universe-filtered quarterly holdings. Filing-date lag and manager-coverage quality must be applied before constructing deltas. Raw reported value units require per-filing repair. |
| `sec13f_manager_holdings` | manager CIK × period × CUSIP | Complete roster-manager books without an S&P 500 filter; use this denominator for portfolio weights. |
| `insider_transactions` | accession × security type × transaction key | Canonical bulk Forms 3/4/5 history. Ticker is resolved CIK-first; the derivative block is structurally null on non-derivative rows. |
| `insider_transactions_live` | accession × security type × XML row sequence | Provisional EDGAR tail. The canonical reader selects one whole source per accession rather than mixing bulk and live rows. |
| `insider_transactions_live_coverage` | ticker | Successful scan frontier, including legitimate zero-filing scans. The source-wide completeness frontier is the minimum over the requested universe. |
| `insider_footnotes` | accession × footnote id | Filing-level Form 3/4/5 prose. Joins through transactions; no ticker is required. |
| `insider_transactions_quarantine` | transaction key | Rejected identity rows retained as evidence. Its claimed ticker is not safe to join to prices. |
| `sec_13d`, `sec_13g` | ticker × accession × reporting-person sequence | Activist and passive beneficial-ownership filings. Pre-2024 structured-data numerics are unavailable, not zero. |
| `sec_13d_transactions` | ticker × accession × trade sequence | Schedule 13D Item 5(c) trade log, distinct from reporting-person grain. |
| `sec_8k` | ticker × accession × item | One row per 8-K item code; a filing commonly contributes several rows. |
| `sec_8k_votes` | ticker × accession × proposal sequence | Item 5.07 shareholder-meeting tallies. Director elections retain nominee-level JSON for recategorization. |
| `sec_filing_text` | ticker × accession × section | Selected 10-K/10-Q narrative sections after minimum-length guards. |

## Governance and proxy data

| Table | Primary grain | Contract |
| --- | --- | --- |
| `def14a_llm` | ticker × accession | Parent structured extraction for board, CEO, ownership, compensation, voting, governance, and auditor fields. Silence is represented as null for tri-state disclosures. |
| `def14a_executive_comp` | parent key × person × fiscal year | Summary Compensation Table rows; missing fiscal-year keys are rejected. |
| `def14a_director_comp` | parent key × person | Item 402(k) outside-director compensation, with fiscal year as payload. |
| `def14a_ownership` | parent key × holder × holder type | Item 403 ownership rows; SEC event sources remain preferred when as-of dates differ. |
| `def14a_directors` | parent key × person | Per-director attributes, including auditable gender basis and board-membership fields. |
| `sec_def14a` | ticker × accession | Deterministic inline-XBRL Pay-versus-Performance facts for the post-2022 regulatory regime. Live presence must be checked before use. |

The four prose child tables are flattened from the retained `def14a_json` payload rather than paid for again. See [SEC and LLM extraction](../flows/sec-llm-extraction.md).

## Behavioral, transcript, and embedding tables

| Table | Primary grain | Contract |
| --- | --- | --- |
| `google_trends`, `wiki_pageviews` | ticker × date | Weekly search interest and daily page views. |
| `earnings_call_sections` | ticker × quarter × section tag | Transcript prose; text is intentionally part of the read payload. |
| `earnings_call_sentiment` | ticker × quarter × section tag | Cached call-intrinsic FinBERT and lexicon scores. |
| `earning_calls_embedding` | ticker × quarter × speaker sequence | Speaker-turn embeddings with question/answer linkage; text is omitted from the default projection. |
| `notes_embedding` | ticker × accession × tag | Mean-pooled footnote embeddings. Currently populated but not consumed by a cube panel. |
| `ticker_descriptions`, `ticker_embeddings` | ticker | Business descriptions and similarity vectors used by peer deduction. |

## Aggregate products

| Table | Primary grain | Lifecycle |
| --- | --- | --- |
| `cube` | ticker × date | Wide feature table plus target columns. Features define the row set; forward labels are left-joined so the newest rows remain scoreable with null targets. |
| `predictions` | ticker × date | Backtest-time scores, replaced by training runs. |
| `cube_signal` | ticker × date | Blended cross-horizon signal. |
| `predictions_latest` | date × ticker × horizon × model | Long-form production scores with distinct as-of, prediction horizon, and production timestamp semantics. |
| `trend_asset_returns` | date | Net returns for the macro trend sleeve. |
| `strategy` | trading day × sleeve × ticker | Upserted trade ledger; opening rows are completed when exits occur. |
| `extraction_run` | table × run id | Durable extraction-run ledger. Different scopes on the same day remain distinct. |

## Cube parts

The unmanaged part tables are `cube_part_prices`, `cube_part_targets`, `cube_part_betas`, `cube_part_fundamentals`, `cube_part_momentum`, `cube_part_text`, `cube_part_institutionals`, and `cube_part_governance`. Their operational policy—command, kind, warm-up, and binding look-backs—lives in [parts.py](../../src/data_aggregate/utils/common/parts.py), not in the schema registry.

Every part uses `(date, ticker)` as its persisted key. Targets encode label and horizon in column names such as `target_rank_h30`; the horizon is not a row key. See [cube parts](../concepts/cube-parts.md) and the [cube-build flow](../flows/cube-build.md).

## Freshness and current state

Cadence names map to maximum ages in [constants.py](../../src/constants/constants.py). Publication-clock overrides matter for SEC facts, notes, pension data, and insider transactions. Freshness metadata describes source expectations; runtime gates such as insider bulk/live completeness add stricter operational checks.

For row counts, physical size, known holes, and tables registered but absent from the local database, use the [live database snapshot](./live-database.md). Re-measure it before operational decisions.

## Related

- [Data platform](../architecture/data-platform.md)
- [Data access and storage](../guides/data-access.md)
- [Data sources](./data-sources.md)
- [Live database](./live-database.md)
