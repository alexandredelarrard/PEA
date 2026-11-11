---
source_commit: 15df4358b2a4ea81c34f02c71076256aed23544a
title: System overview
description: Boundaries and major components of the stock_pick_strat quantitative pipeline.
type: architecture
tags:
  - wiki
  - architecture
---
# System overview

## Summary

The system is a staged quantitative pipeline: source-specific extractors persist normalized observations, cube builders create point-in-time feature panels, model code turns the cube into predictions, the portfolio layer constructs sector-neutral long/short positions and simulates them alongside S&P 500 and cash. The package map and production entry points are implemented by the orchestrators under [src](../../src/) and detailed in [orchestration and runtime](./orchestration-and-runtime.md).

## Diagram

~~~mermaid
flowchart TD
  Providers[Market SEC vendor and text providers] --> Extract[Data extraction]
  Extract --> DB[(PostgreSQL tables)]
  DB --> Peer[Peer baskets]
  DB --> CubeParts[Cube part builders]
  Peer --> CubeParts
  CubeParts --> Cube[(Wide point in time cube)]
  Cube --> Train[Per horizon ensemble]
  Train --> Predict[Latest predictions]
  Train --> Portfolio[Model long short market and cash backtest]
  DB --> Portfolio
  Portfolio --> Report[Offline portfolio report]
  Airflow[Airflow] --> Extract
  Airflow --> CubeParts
  Airflow --> Train
~~~

## Key components

- The traded universe is [configs/universe.yml](../../configs/universe.yml), the only hand-edited ticker list, each ticker with its GICS sector and sub-industry. `StepExtractUniverse` (`data_extract universe`) refreshes `sec_company_tickers` and `sharadar_tickers`, then syncs it into `universe_tickers` ([sync.py](../../src/utils/tickers_universe/sync.py)); every stage reads tickers, CIKs, GICS and entry dates through `context.universe` ([universe.py](../../src/utils/tickers_universe/universe.py)).
- [Data extraction](../modules/data-extract.md) is coordinated by `StepExtractAllData` in [step_extract_all_data.py](../../src/data_extract/step_extract_all_data.py), which fetches `sharadar_tickers` and syncs the universe first. It composes functional steps: `StepExtractZips` (SEC bulk zips: download, then ingest), `StepExtractIdentity` (the offline identity build, read from tables and the Form 3/4/5 and Notes zips, and its SQL-only sync), `StepExtractEdgar` (the eight per-filing EDGAR walks, each one `EdgarFetch` walked by the shared `run_edgar_fetch` in [edgar_driver.py](../../src/data_extract/utils/edgar_extract/edgar_driver.py)) and `StepExtractLlm` (the provider-backed extractions, which read only stored text: `sec_def14a_text`, the `sec_8k` narratives and `sec_employees_text`), beside the prices, institutionals, fundamentals and behavioral steps; bulk SEC zips go through one reader in [bulk_cache.py](../../src/data_extract/utils/common/bulk_cache.py). FTD, short-volume, 13F and insider rows are stored as filed, at security grain; the cube maps them to tickers with the dated `universe_cusip` / `universe_cik_symbol` rows, applies the class ratios and sums to ticker (13F per manager × ticker after the vintage resolver) before the feature builders.
- [Peer deduction](../modules/data-peers.md) persists business-similarity and return-based peer baskets through `StepDeducePeers` in [step_deduce_peers.py](../../src/data_peers/step_deduce_peers.py).
- [Cube aggregation](../modules/data-aggregate.md) owns nine persisted parts and the final assembly in [step_build_cube.py](../../src/data_aggregate/step_build_cube.py).
- [Modelling](../modules/modelling.md) supplies saved models; [portfolio construction](../modules/strategies.md) and the quantity engine live inside [portfolio](../modules/portfolio.md).
- [Validation](../modules/validation.md) is a read-only subsystem with reusable checks and explicit result gates.

## Design decisions

The repository uses package-level orchestration rather than a single script. Every major stage has a narrow entry point, persists its output, and can be invoked independently by the CLI or Airflow. That separation keeps expensive source work resumable and allows the cube, training, and prediction schedules to run at different cadences; the four schedules are implemented in [src/dags](../../src/dags/dag_strat_prediction.py).

Tabular state belongs in PostgreSQL, while models, diagnostics, caches, and peer dictionaries are non-tabular artifacts resolved through `Context.paths` in [context.py](../../src/context.py). SQL is isolated behind the [data-store boundary](../concepts/table-registry-and-store-boundary.md).

## Related

- [Nightly data refresh](../flows/nightly-data-refresh.md)
- [Cube build](../flows/cube-build.md)
- [Feature-to-portfolio architecture](./feature-to-portfolio.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
