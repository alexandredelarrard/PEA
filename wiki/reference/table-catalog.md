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
| `symbol_tenure` | symbol × issuer CIK × valid-from × `source` × `evidence_period` | Symbol evidence, one row per source. `form345` (Form 3/4/5 issuer symbols) and `manual` ([symbol_tenure_manual.json](../../configs/sec/symbol_tenure_manual.json)) are written by `identity-tables`; `dei` (10-K/10-Q cover-page `dei:TradingSymbol`, one row per Notes zip period in `evidence_period`) is written by `notes-download`. `evidence_period` is `''` for `form345` and `manual`. Each writer replaces only its own partition. Intervals are observed, half-open and may overlap, so callers run membership tests. Aggregate readers read `form345` and `manual` only. |
| `entity_lineage` | CIK × `role` × symbol × valid-from | The dated identity verdict and the only identity input of the fetchers. `role` `cik_window`: the CIK's consolidating filings (10-K/10-Q, DEF 14A) belong to the entity over `[valid_from, valid_to)`; `cik_event`: event forms only (co-registrant subsidiaries, CIKs with no dated window); `symbol`: a dated symbol interval with `status` `curated`, `corroborated`, `single_source`, `noise` or `conflict` (`symbol` is `''` on CIK rows). An open start is stored as the sentinel `1900-01-01` (a primary-key column cannot be NULL); an open end is NULL. Current = `role='symbol' AND valid_to IS NULL`. One CIK belongs to one entity, asserted by the build; `entity_id` is `E` plus the oldest CIK. Only universe entities and hand-adjudicated CIKs are stored; a CIK with no row is its own entity. `scope_changed_at` is the build time at which a ticker's rows last changed, stamped apart for CIK rows (drives the EDGAR relist) and symbol rows (their own stamp; drives the FTD re-resolution). |

`symbol_tenure` is evidence and `entity_lineage` is the verdict: fetchers read the lineage through `Identity` in [identity.py](../../src/data_extract/utils/common/identity.py), never the tenure. [Registrant policy](../../src/data_extract/utils/common/registrant.py) decides per form family which lineage CIKs a filing may come from: every CIK of the entity for event forms, the dated windows for consolidating forms. Both tables are fully derived; a primary-key change means dropping and recreating them before the next build (see the identity cutover in the [run guide](../guides/run-the-pipeline.md)). See [data sources](./data-sources.md) and [source availability](../concepts/source-availability.md).

## Prices and market data

| Table | Primary grain | Time key | Important semantics |
| --- | --- | --- | --- |
| `prices` | ticker × date | date | Equity OHLCV only. `close_split` is split-adjusted for levels; `close_total` is total-return adjusted for returns. There is deliberately no ambiguous `close` column. |
| `dividends` | ticker × ex-date | date | Sparse cash distributions; absence is normal for non-payers. |
| `prices_splits` | ticker × ex-date | date | Sparse split/spinoff factors used in price-basis reconciliation. |
| `prices_macro` | named series × date | date | Long-form benchmark, volatility, commodity, energy, rate, credit, breakeven, FX, and derived macro series. These series never enter the equity `prices` cross-section. |
| `short_interest` | ticker × date | date | FINRA RegSHO tape. The source is market-wide per day, so incremental resume is global. |
| `sec_fails_to_deliver` | ticker × settlement date | date | SEC semi-monthly settlement-fail rows. The persisted `period` identifies the source ZIP and drives feature availability on the fly; a `b` ZIP can contain a day-15 settlement, so settlement date cannot select the ZIP or serve as feature as-of. |

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
| `fundamentals_employees` | ticker × SEC filing date (`as_of`) | Annual issuer-wide headcount from a separate 10-K/10-K/A/10-K405 text and LLM walk. A filing with no supported count (image-only, not disclosed, ambiguous) is a NULL `employees` row, which marks it decided; consumers drop NULL rows. Hand-kept decisions live in `configs/sec/employees_manual_roster.json`. XBRL facts rebuilds do not write this table. |
| `earnings_surprises` | ticker × earnings date | Consensus and actual EPS. Future scheduled dates are possible; realized features require a non-null actual. |
| `pension_facts` | CIK × tag × period × duration × ZIP quarter | Curated pension facts from quarterly SEC Financial Statement Data Sets. `ddate` is fact time; date-typed `available_at` drives quarterly freshness. Historical ZIPs through 2026q2 use estimated quarter-end +12 calendar days (Monday if weekend); newly downloaded ZIPs from 2026q3 use the successful New York download date. A new ZIP vintage does not overwrite an earlier one. The pension feature starts at `max(filed, available_at)` and carries the latest eligible observation for at most 460 calendar days. The historical clock is an estimate, not verified SEC publication evidence. See [data sources](./data-sources.md). |
| `notes_num`, `notes_text` | accession × tag × period × duration | Numeric and narrative SEC footnote archives. `ddate` is fact time; date-typed `available_at` drives monthly freshness. Through the August 2026 archive it is an estimated next-month 12th (Monday if weekend); from September 2026 onward it is the New York completion date for a new download (cached files fall back to modification date). Numeric consumers start at `max(filed, available_at)` and can carry the latest annual fact for 460 calendar days. Historical dates are estimates, not verified SEC publication dates; see [data sources](./data-sources.md). |

The distinction between the two histories is load-bearing: [data extraction](../modules/data-extract.md) builds the SEC replay independently, then the Sharadar producer builds the merged consumer table.

## Ownership, institutional, and event data

| Table | Primary grain | Contract |
| --- | --- | --- |
| `sec13f_hr` | manager CIK × period × ticker × CUSIP | Universe-filtered quarterly holdings. Filing-date lag and manager-coverage quality must be applied before constructing deltas. Raw reported value units require per-filing repair. |
| `sec13f_manager_holdings` | manager CIK × period × CUSIP | Complete roster-manager books without an S&P 500 filter; use this denominator for portfolio weights. |
| `insider_transactions` | `accession_number` × `security_type` × `row_sequence` | The only Forms 3/4/5 transaction table. `row_sequence` is the 1-based row inside the filing's non-derivative or derivative table (XML order; the zip's surrogate key ranked per table gives the same number). Daily EDGAR rows are authoritative (`source='edgar'`); a quarterly zip adds only the filings EDGAR lacks (`source='zip'`), never both sources in one accession. `quarter` is the zip quarter that covered the filing, NULL until one has. Joint filings keep one row per trade: the primary owner (best role Officer < Director < 10% owner < Other, then lowest CIK) fills `owner_cik`/`owner_name`/`officer_title`, `owner_ciks` lists every reporting owner, `n_reporting_owners` counts them, and the four role flags are OR'ed across owners. Also `original_submission_date` (set on amendments), `document_type`, `fetched_at`; `footnote_ids` and `acceptance_datetime` are EDGAR-only (NULL on zip rows). Ticker is resolved CIK-first; identity rejects are not stored. Daily freshness on `filing_date`; the derivative block is structurally null on non-derivative rows. |
| `insider_footnotes` | accession × footnote id | Filing-level Form 3/4/5 prose. Joins through transactions; no ticker is required. |
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
| `earnings_call_sections` | ticker × quarter × paragraph | Raw defeatbeta transcript paragraphs: `as_of` DATE is the real call date (not null), plus `transcript_id`, `speaker`, `content`. `quarter` is the fiscal label (`2026Q2`). Nothing is split or cleaned here; quarterly freshness on `as_of`. `content` is the payload, so reads project the needed columns and scope by ticker. |
| `earnings_call_sentiment` | ticker × quarter × section tag | Cached call-intrinsic FinBERT and lexicon scores of the cleaned prepared remarks and the management-answer Q&A text; rows carry the cleaned-text cache version in `model`. Also holds the pending refresh markers of re-issued calls. |
| `earning_calls_embedding` | ticker × quarter × speaker sequence | Speaker-turn embeddings of the split's cleaned turns, with question/answer linkage (`exchange_idx`); `model` is the cache tag `<OpenAI model>:<cleaning version>`. Text is omitted from the default projection. |
| `notes_embedding` | ticker × accession × tag | Mean-pooled footnote embeddings. Currently populated but not consumed by a cube panel. |
| `ticker_descriptions`, `ticker_embeddings` | ticker | Business descriptions and similarity vectors used by peer deduction. |

Earnings-call `as_of` is stored as the call date with no +1 day. The text aggregate aligns calls to the trading calendar with `searchsorted(as_of, side="right")`, so a call dated D is first visible on the next trading session: a Wednesday call on Thursday, a Friday call on Monday. The source date is date-only, so this one-session lag is the earliest safe visibility.

## Aggregate products

| Table | Primary grain | Lifecycle |
| --- | --- | --- |
| `cube` | ticker × date | Wide feature table plus target columns. Features define the row set; forward labels are left-joined so the newest rows remain scoreable with null targets. |
| `predictions` | ticker × date | Backtest-time scores, replaced by training runs. |
| `cube_signal` | ticker × date | Blended cross-horizon signal. |
| `predictions_latest` | date × ticker × horizon × model | Long-form production scores with distinct as-of, prediction horizon, and production timestamp semantics. |
| `trend_asset_returns` | date | Net returns of the removed macro trend sleeve; unused since 2026-10 (registry entry kept until the next data-store change). |
| `strategy` | trading day × sleeve × ticker | Upserted trade ledger; opening rows are completed when exits occur. |
| `extraction_run` | table × run id | Durable extraction-run ledger. Different scopes on the same day remain distinct. |

## Cube parts

The unmanaged part tables are `cube_part_prices`, `cube_part_targets`, `cube_part_betas`, `cube_part_fundamentals`, `cube_part_momentum`, `cube_part_text`, `cube_part_institutionals`, and `cube_part_governance`. Their operational policy—command, kind, warm-up, and binding look-backs—lives in [parts.py](../../src/data_aggregate/utils/common/parts.py), not in the schema registry.

Every part uses `(date, ticker)` as its persisted key. Targets encode label and horizon in column names such as `target_rank_h30`; the horizon is not a row key. See [cube parts](../concepts/cube-parts.md) and the [cube-build flow](../flows/cube-build.md).

## Freshness and current state

Cadence names map to maximum ages in [constants.py](../../src/constants/constants.py). Publication-clock overrides matter for SEC facts, notes, pension data, and insider transactions. Freshness metadata describes source expectations; runtime gates such as the insider completeness frontier (the EDGAR run's manifest entry over the exact cube universe) add stricter operational checks.

For row counts, physical size, known holes, and tables registered but absent from the local database, use the [live database snapshot](./live-database.md). Re-measure it before operational decisions.

## Related

- [Data platform](../architecture/data-platform.md)
- [Data access and storage](../guides/data-access.md)
- [Data sources](./data-sources.md)
- [Live database](./live-database.md)
