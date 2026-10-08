---
title: Data extraction
description: Source-specific fetchers and the ordered orchestration that populates normalized extract tables.
type: module
tags:
  - wiki
  - module
---
# Data extraction

## Summary

`src/data_extract` owns external source ingestion. The source CLI and nightly DAG cover price, institutional, Sharadar-fundamental, SEC-fundamental, governance-structure, and earnings-call data in dependency order. Every run derives its work list from the table's own rows, its `Resume` contract in the central [table registry](../../src/data_store/schema.py) and the run date; after the fetchers, `extraction-status` reports the freshness of every table that declares a cadence and blocks nothing. Source-specific quirks and availability limits are documented in [data sources](../reference/data-sources.md).

## Responsibilities

- Plan each run from the database alone: no file under `data/` decides what to fetch, and no table is read in full to find where to continue.
- Normalize provider identifiers, dates, filing metadata, and source-specific schemas.
- Cache expensive bulk downloads while persisting tabular results through `context.store`.
- Save what each run read, re-read failed SEC documents in a few in-task rounds, log the coverage and exit 0; what could not be read is listed again on the next run.
- Report each schema-declared table's age against its cadence, and the per-ticker fresh share of the three prediction inputs.
- Maintain shared EDGAR, identity, registrant, lineage, retry and pacing plumbing, the local EDGAR filing index, and the work-list planners.
- Extract employee headcount through an independent annual-filing walk where a table row (counted or status-only) marks a filing decided; the cube reads that table directly.

## Public API / entry points

- `StepExtractAllData.run()` in [step_extract_all_data.py](../../src/data_extract/step_extract_all_data.py).
- The flat source-command group in [data_extract/cli.py](../../src/data_extract/cli.py).
- Sub-steps under `src/data_extract/transformers/`.

## Key files

- [step_extract_prices.py](../../src/data_extract/transformers/step_extract_prices.py) writes equity prices, dividends, splits, and named macro series.
- [step_extract_institutionals.py](../../src/data_extract/transformers/step_extract_institutionals.py) owns 13F, superinvestors, insiders, 13D/13G, 8-K, short-volume, and fails-to-deliver sources. The [FTD fetcher](../../src/data_extract/utils/institutionals/fetch_fails_to_deliver.py) persists each settlement row's source ZIP `period` in `sec_fails_to_deliver`; it does not write a separate availability table.
- [step_extract_fundamentals_sharadar.py](../../src/data_extract/transformers/step_extract_fundamentals_sharadar.py) builds the vendor layer and merged consumer history.
- [step_extract_fundamentals.py](../../src/data_extract/transformers/step_extract_fundamentals.py) builds SEC facts and the replay history.
- [fundamentals_employees.py](../../src/data_extract/utils/fundamentals/fundamentals_employees.py) independently lists annual filings across the dated CIK chain and skips decided dates; per filing it strips the iXBRL header, sends the workforce-number windows (or an annual-report exhibit when the primary document has none or incorporates the count by reference) to Luna, guards each returned component against its own quote, and writes one row per filing date to `fundamentals_employees`: components, basis, status, filer CIK, accession, source document and quotes, with SEC filing-date `as_of`. See [data sources](../reference/data-sources.md).
- [step_extract_structure.py](../../src/data_extract/transformers/step_extract_structure.py) handles filing text, DEF 14A, and Item 5.07 votes. In [votes/fetch.py](../../src/data_extract/utils/structure/votes/fetch.py), `fetch_8k_votes_llm` loads the stored 5.07 narratives, plans the accessions not yet in `sec_8k_votes` (markers included), builds each ticker's LLM tasks (a text the guard refuses is saved as a marker with no call), then extracts and saves ticker by ticker.
- [step_extract_behavioral.py](../../src/data_extract/transformers/step_extract_behavioral.py) runs the earnings-call extractor; there is no Wikipedia pageview or Google Trends fetcher.
- `src/data_extract/utils/behavioral/fetch_earnings_call_transcripts.py` (`extract_earnings_calls`, CLI `extract-earnings-calls [-F] [-t]`) writes raw defeatbeta paragraphs to `earnings_call_sections`. It pins every read to the dataset commit and checks the source schema. A run reads the parquet footer and keeps the row groups whose `symbol` statistics meet the scope and whose latest call reaches the stored frontier minus the table's `Resume` overlap (a new ticker's groups at any date; a cold table or `-F` reads every group). It reads the five index columns on `read_workers` threads, diffs them against stored calls, and reads the `transcripts` column only for row groups holding new or re-issued calls. Only `-F` deletes stored calls the source no longer has. All DB writes run on one thread, one row group per batch.
- [schema.py](../../src/data_store/schema.py) is the sole resume and freshness inventory. Each extracted table declares a `Resume` contract (mode, key, frontier column, overlap, EDGAR forms, archive period column, source start) and, where a read can find nothing, an `empty_marker`; `freshness_tables()` exposes each table's cadence and publication date column to `extraction-status` and the `predict` guard.
- `src/data_extract/utils/common/` centralizes EDGAR and identity mechanics; each mechanism exists once:
  - [resume.py](../../src/data_extract/utils/common/resume.py): the work-list planners. `series_windows` (dated per-key series), `document_worklist` (EDGAR documents: local index rows of the table's forms for each key's lineage, minus stored accessions, markers included; `stored_accessions` reads them once per table, per key for 13D, 13G and insider or one table-wide set elsewhere, and each key's set is cut to its own listing before the comparison) and `archive_worklist` (bulk archives: periods missing from any of the fetcher's tables, plus every cached period for a new ticker). `data_extract resume-plan` prints the document work lists read-only.
  - [edgar_index.py](../../src/data_extract/utils/common/edgar_index.py): the local copy of SEC's quarterly `full-index/master.gz`, one Parquet per quarter under `data/sec_edgar_index`, kept to the universe's forms. Every EDGAR document fetch refreshes it first (the current quarter, any missing one, and the previous one early in a quarter); `edgar-index --build` downloads every quarter again.
  - [sec_io.py](../../src/data_extract/utils/common/sec_io.py): the one SEC retry policy (`sec_retry`) and the shared per-process limiter for every SEC request; an exhausted retry raises `TransientReadError`.
  - [edgar_driver.py](../../src/data_extract/utils/common/edgar_driver.py): `EdgarFetch` declares one per-ticker EDGAR document fetch (8-K, 13D, 13G, DEF 14A ECD, filing text, SEC fundamentals, EDGAR insider) and `run_edgar_fetch` walks it: refresh the index, plan the work list, read each filing as one unit, re-read failed units in `retry_rounds`, log coverage and return. `runtime_floor` raises the listing floor (the insider zip floor) and `done_where` filters the done set (the insider fetch counts only `source='edgar'` rows); `FilingStamp` reads filing CIK, filing date, guarded `period_of_report`, and document URL once per filing.
  - [empty_markers.py](../../src/data_extract/utils/common/empty_markers.py): `marker_frame` builds an empty-filing marker; `drop_markers_over_data` drops any marker whose key already holds a real row, so a marker never replaces data.
  - [identity.py](../../src/data_extract/utils/common/identity.py): `Identity.filing_scope(ticker)` returns the precomputed `FilingScope` (entity, roster CIK, every entity CIK, symbols, same-CIK aliases) used for registrant resolution.
  - [registrant.py](../../src/data_extract/utils/common/registrant.py): one registrant walk for index rows and `Company` listings, and the filing-header subject check that tells a 13D/13G subject from its filer.
  - [bulk_cache.py](../../src/data_extract/utils/common/bulk_cache.py): `read_zip_tables` for every SEC bulk zip, the cached-archive listing, and archive availability dates.
  - [symbol_tenure.py](../../src/data_extract/utils/common/symbol_tenure.py): `scan_form345_cache` reads the Form 3/4/5 cache once for both `symbol_tenure` and [entity_lineage](../../src/data_extract/utils/common/entity_lineage.py).
  - [parallel_fetch.py](../../src/data_extract/utils/common/parallel_fetch.py), [item_carve.py](../../src/data_extract/utils/common/item_carve.py), [frame_sanitize.py](../../src/data_extract/utils/common/frame_sanitize.py), and [incremental.py](../../src/data_extract/utils/common/incremental.py): guarded thread pool, item-heading carving, primary-key de-duplication, and stored-value helpers. `stored_values` is the one reader of stored keys and accessions (`SELECT DISTINCT`), shared by the document and archive planners and the DEF 14A done sets.
- Institutional sharing: [fetch_13f.py](../../src/data_extract/utils/institutionals/fetch_13f.py) parses each 13F-HR once for `sec13f_hr` and `sec13f_manager_holdings` (its information-table downloads go through `sec_io`, and every 13F writer pads the CIK to 10 digits), while [fetch_13f_managers.py](../../src/data_extract/utils/institutionals/fetch_13f_managers.py) catches roster CIKs up on the filings their stored (period, filing date) pairs do not show yet; [fetch_13f_backfill.py](../../src/data_extract/utils/institutionals/fetch_13f_backfill.py) fills a new ticker's `sec13f_hr` history from the cached SEC 13F data sets, then one EDGAR walk over the filings after them (a `sp500_tickers` without `added_on` has no new tickers); [fetch_superinvestors.py](../../src/data_extract/utils/institutionals/fetch_superinvestors.py) lists a manager's 13F filings through `sec_io.company` and `company_filings`, and a throttle that outlasts the retries raises `EdgarListingError`; [schedule_rows.py](../../src/data_extract/utils/institutionals/schedule_rows.py) is the one 13D/13G row builder; [insider_common.py](../../src/data_extract/utils/institutionals/insider_common.py) is the insider contract shared by the zip and EDGAR paths: one encoding, the joint-filing primary-owner rule, and the identity-exclusion summary. Both paths write the one `insider_transactions` table: EDGAR is authoritative, a zip quarter adds only the filings EDGAR lacks and logs how many it found.

## Dependencies

The module depends on [DataStore](./data-store.md), shared runtime utilities, yfinance, SEC/FINRA/FRED endpoints, Sharadar, the HuggingFace Hub (earnings-call transcripts), and the reusable [GPT extraction module](./gpt-extract.md).

## Participates in

- [Nightly data refresh](../flows/nightly-data-refresh.md)
- [SEC and LLM extraction](../flows/sec-llm-extraction.md)
- [Point-in-time data](../concepts/point-in-time-data.md)

## Related

- [Add a data source](../guides/add-a-data-source.md)
- [Source availability](../concepts/source-availability.md)
- [Table catalog](../reference/table-catalog.md)
