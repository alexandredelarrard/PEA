---
title: System overview
description: Boundaries and major components of the stock_pick_strat quantitative pipeline.
type: architecture
tags:
  - wiki
  - architecture
---
# System overview

## Summary

The system is a staged quantitative pipeline: source-specific extractors persist normalized observations, cube builders create point-in-time feature panels, model code turns the cube into predictions, strategy sleeves turn signals into return streams and holdings, and the portfolio layer blends those sleeves into trades. The package map and production entry points are implemented by the orchestrators under [src](../../src/) and detailed in [orchestration and runtime](./orchestration-and-runtime.md).

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
  Predict --> Sleeves[Strategy sleeves]
  DB --> Sleeves
  Sleeves --> Blend[ERC blend and volatility target]
  Blend --> Trades[(Strategy ledger)]
  Airflow[Airflow] --> Extract
  Airflow --> CubeParts
  Airflow --> Train
~~~

## Key components

- [Data extraction](../modules/data-extract.md) is coordinated by `StepExtractAllData` in [step_extract_all_data.py](../../src/data_extract/step_extract_all_data.py).
- [Peer deduction](../modules/data-peers.md) persists business-similarity and return-based peer baskets through `StepDeducePeers` in [step_deduce_peers.py](../../src/data_peers/step_deduce_peers.py).
- [Cube aggregation](../modules/data-aggregate.md) owns eight persisted parts and the final assembly in [step_build_cube.py](../../src/data_aggregate/step_build_cube.py).
- [Modelling](../modules/modelling.md), [strategies](../modules/strategies.md), and [portfolio](../modules/portfolio.md) form distinct signal, sleeve, and allocation layers.
- [Validation](../modules/validation.md) is a read-only subsystem with reusable checks and explicit result gates.

## Design decisions

The repository uses package-level orchestration rather than a single script. Every major stage has a narrow entry point, persists its output, and can be invoked independently by the CLI or Airflow. That separation keeps expensive source work resumable and allows the cube, training, and prediction schedules to run at different cadences; the four schedules are implemented in [src/dags](../../src/dags/).

Tabular state belongs in PostgreSQL, while models, diagnostics, caches, and peer dictionaries are non-tabular artifacts resolved through `Context.paths` in [context.py](../../src/context.py). SQL is isolated behind the [data-store boundary](../concepts/table-registry-and-store-boundary.md).

## Related

- [Nightly data refresh](../flows/nightly-data-refresh.md)
- [Cube build](../flows/cube-build.md)
- [Feature-to-portfolio architecture](./feature-to-portfolio.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
