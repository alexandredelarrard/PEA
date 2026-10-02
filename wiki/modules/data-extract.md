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

`src/data_extract` owns external source ingestion. The source CLI and nightly DAG cover price, institutional, Sharadar-fundamental, SEC-fundamental, governance-structure, and earnings-call data in dependency order. Before aggregation, the extraction status command checks exactly the tables that declare freshness in the central [table registry](../../src/data_store/schema.py). Source-specific quirks and availability limits are documented in [data sources](../reference/data-sources.md).

## Responsibilities

- Resume each source from database frontiers rather than rereading complete tables.
- Normalize provider identifiers, dates, filing metadata, and source-specific schemas.
- Cache expensive bulk downloads while persisting tabular results through `context.store`.
- Keep completed per-ticker work durable, but fail completeness-sensitive SEC walks when any requested ticker fails so Airflow retries the source.
- Gate aggregation on each schema-declared table's maximum publication date and cadence tolerance.
- Maintain shared EDGAR, identity, registrant, lineage, pacing, and manifest plumbing, including per-ticker identity-scope fingerprints.
- Extract employee headcount through an independent annual-filing walk with accession-level terminal and pending outcomes.

## Public API / entry points

- `StepExtractAllData.run()` in [step_extract_all_data.py](../../src/data_extract/step_extract_all_data.py).
- The flat source-command group in [data_extract/cli.py](../../src/data_extract/cli.py).
- Sub-steps under `src/data_extract/transformers/`.

## Key files

- [step_extract_prices.py](../../src/data_extract/transformers/step_extract_prices.py) writes equity prices, dividends, splits, and named macro series.
- [step_extract_institutionals.py](../../src/data_extract/transformers/step_extract_institutionals.py) owns 13F, superinvestors, insiders, 13D/13G, 8-K, short-volume, and fails-to-deliver sources. The [FTD fetcher](../../src/data_extract/utils/institutionals/fetch_fails_to_deliver.py) persists each settlement row's source ZIP `period` in `sec_fails_to_deliver`; it does not write a separate availability table.
- [step_extract_fundamentals_sharadar.py](../../src/data_extract/transformers/step_extract_fundamentals_sharadar.py) builds the vendor layer and merged consumer history.
- [step_extract_fundamentals.py](../../src/data_extract/transformers/step_extract_fundamentals.py) builds SEC facts and the replay history.
- [fundamentals_employees.py](../../src/data_extract/utils/fundamentals/fundamentals_employees.py) independently lists annual filings across the dated CIK chain, validates Luna's quoted headcount against filing text, and writes only `fundamentals_employees` with SEC filing-date `as_of`.
- [step_extract_structure.py](../../src/data_extract/transformers/step_extract_structure.py) handles filing text, DEF 14A, and Item 5.07 votes.
- [step_extract_behavioral.py](../../src/data_extract/transformers/step_extract_behavioral.py) handles earnings-call transcripts; there is no Wikipedia pageview or Google Trends fetcher.
- [schema.py](../../src/data_store/schema.py) is the sole freshness inventory: `freshness_tables()` exposes each checked table, cadence, and publication date column to the CLI gate.
- `src/data_extract/utils/common/` centralizes EDGAR and identity mechanics; each mechanism exists once:
  - [edgar_driver.py](../../src/data_extract/utils/common/edgar_driver.py): `EdgarFetch` declares one per-ticker EDGAR fetch (8-K, 13D, 13G, DEF 14A ECD, filing text, SEC fundamentals, live insider) and `run_edgar_fetch` walks it; `FilingStamp` reads filing CIK, filing date, guarded `period_of_report`, and document URL once per filing; `build_filing_rows` is the generic single-table builder.
  - [identity.py](../../src/data_extract/utils/common/identity.py): `Identity.filing_scope(ticker)` returns the precomputed `FilingScope` used for registrant resolution and the per-ticker identity-scope fingerprint.
  - [registrant.py](../../src/data_extract/utils/common/registrant.py) and [sec_atom.py](../../src/data_extract/utils/common/sec_atom.py): one registrant walk, the Schedule 13D/13G subject search, and the owner-inclusive Atom feed it shares with the live insider listing.
  - [bulk_cache.py](../../src/data_extract/utils/common/bulk_cache.py): `read_zip_tables` for every SEC bulk zip, `pending_periods` / `mark_processed` for bulk incremental state, and archive availability dates.
  - [symbol_tenure.py](../../src/data_extract/utils/common/symbol_tenure.py): `scan_form345_cache` reads the Form 3/4/5 cache once for both `symbol_tenure` and [entity_lineage](../../src/data_extract/utils/common/entity_lineage.py).
  - [parallel_fetch.py](../../src/data_extract/utils/common/parallel_fetch.py), [run_manifest.py](../../src/data_extract/utils/common/run_manifest.py), [item_carve.py](../../src/data_extract/utils/common/item_carve.py), [frame_sanitize.py](../../src/data_extract/utils/common/frame_sanitize.py), and [incremental.py](../../src/data_extract/utils/common/incremental.py): guarded thread pool, manifest window and atomic manifest writes, item-heading carving, primary-key de-duplication, and stored-value resume helpers.
- Institutional sharing: [fetch_13f.py](../../src/data_extract/utils/institutionals/fetch_13f.py) parses each 13F-HR once for `sec13f_hr` and `sec13f_manager_holdings`, while [fetch_13f_managers.py](../../src/data_extract/utils/institutionals/fetch_13f_managers.py) catches roster CIKs up from their stored frontier; [schedule_rows.py](../../src/data_extract/utils/institutionals/schedule_rows.py) is the one 13D/13G row builder; [insider_common.py](../../src/data_extract/utils/institutionals/insider_common.py) is the insider contract shared by the bulk and live paths.

## Dependencies

The module depends on [DataStore](./data-store.md), shared runtime utilities, yfinance, SEC/FINRA/FRED endpoints, Sharadar, and the reusable [GPT extraction module](./gpt-extract.md).

## Participates in

- [Nightly data refresh](../flows/nightly-data-refresh.md)
- [SEC and LLM extraction](../flows/sec-llm-extraction.md)
- [Point-in-time data](../concepts/point-in-time-data.md)

## Related

- [Add a data source](../guides/add-a-data-source.md)
- [Source availability](../concepts/source-availability.md)
- [Table catalog](../reference/table-catalog.md)
