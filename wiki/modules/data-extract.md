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
- [step_extract_behavioral.py](../../src/data_extract/transformers/step_extract_behavioral.py) handles earnings-call transcripts; retired Wikipedia and Google Trends sources are not part of the nightly contract.
- [schema.py](../../src/data_store/schema.py) is the sole freshness inventory: `freshness_tables()` exposes each checked table, cadence, and publication date column to the CLI gate.
- `src/data_extract/utils/common/` centralizes EDGAR and identity mechanics.

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
