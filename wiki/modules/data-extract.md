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

`src/data_extract` owns external source ingestion. A super-step resolves the equity universe once, then executes price, institutional, Sharadar-fundamental, SEC-fundamental, governance-structure, and behavioral sub-steps in dependency order. Source-specific quirks and availability limits are documented in [data sources](../reference/data-sources.md).

## Responsibilities

- Resume each source from database frontiers rather than rereading complete tables.
- Normalize provider identifiers, dates, filing metadata, and source-specific schemas.
- Cache expensive bulk downloads while persisting tabular results through `context.store`.
- Continue per ticker when a provider fails and keep completed work durable.
- Maintain shared EDGAR, identity, registrant, lineage, pacing, and manifest plumbing.

## Public API / entry points

- `StepExtractAllData.run()` in [step_extract_all_data.py](../../src/data_extract/step_extract_all_data.py).
- The flat source-command group in [data_extract/cli.py](../../src/data_extract/cli.py).
- Sub-steps under [data_extract/transformers](../../src/data_extract/transformers/).

## Key files

- [step_extract_prices.py](../../src/data_extract/transformers/step_extract_prices.py) writes equity prices, dividends, splits, and named macro series.
- [step_extract_institutionals.py](../../src/data_extract/transformers/step_extract_institutionals.py) owns 13F, superinvestors, insiders, 13D/13G, 8-K, short-volume, and fails-to-deliver sources.
- [step_extract_fundamentals_sharadar.py](../../src/data_extract/transformers/step_extract_fundamentals_sharadar.py) builds the vendor layer and merged consumer history.
- [step_extract_fundamentals.py](../../src/data_extract/transformers/step_extract_fundamentals.py) builds SEC facts and the replay history.
- [step_extract_structure.py](../../src/data_extract/transformers/step_extract_structure.py) handles filing text, DEF 14A, and Item 5.07 votes.
- [step_extract_behavioral.py](../../src/data_extract/transformers/step_extract_behavioral.py) handles pageviews, trends, and earnings-call transcripts.
- [data_extract/utils/common](../../src/data_extract/utils/common/) centralizes EDGAR and identity mechanics.

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
