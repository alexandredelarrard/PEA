---
title: Nightly data refresh
description: Airflow extraction fan-out, peer deduction, sequential cube parts, assembly, status, and prediction trigger.
type: flow
tags:
  - wiki
  - flow
---
# Nightly data refresh

## Summary

The nightly path refreshes source tables in parallel, then triggers a memory-bounded aggregation chain. Extraction tasks are grouped by operational resource pools rather than application package, while cube tasks run sequentially and derive their commands from the part registry.

## Trigger

The `data_extraction` DAG is scheduled daily by [dag_data_extraction.py](../../src/dags/dag_data_extraction.py). On completion it triggers the unscheduled aggregation DAG in [dag_data_aggregation.py](../../src/dags/dag_data_aggregation.py).

## Sequence diagram

~~~mermaid
sequenceDiagram
  participant Airflow
  participant Extract as Data extraction CLI
  participant Store as PostgreSQL
  participant Peers as Peer deduction
  participant Cube as Cube builders
  participant Predict as Prediction DAG
  Airflow->>Extract: seed universe
  Airflow->>Extract: run source tasks by pool
  Extract->>Store: incremental upserts
  Airflow->>Peers: deduce peers
  Peers-->>Airflow: persist peer dictionary
  Airflow->>Cube: run registered parts sequentially
  Cube->>Store: write cube parts and cube
  Airflow->>Cube: run cube status
  Airflow->>Predict: trigger daily scoring
~~~

## Steps

1. Seed or load the S&P 500 universe through [data extraction](../modules/data-extract.md).
2. Fan out source commands across SEC bulk, SEC API, scrape, and default pools.
3. Persist every completed entity or source chunk through [DataStore](../modules/data-store.md).
4. Trigger [peer deduction](../modules/data-peers.md), then the [cube-build flow](./cube-build.md).
5. Run the part-status gate and trigger [model prediction](./model-training-and-prediction.md).

## Failure modes

The extraction completion gate is configured to remain visible even when a source task fails, allowing downstream data to build from the latest available state. Each fetcher therefore needs correct resume semantics and source freshness metadata. Aggregation is sequential because the largest part determines peak memory; parallelizing part builds would add their working sets.

## Related

- [DAGs and infrastructure](../modules/dags-and-infrastructure.md)
- [Source availability](../concepts/source-availability.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
