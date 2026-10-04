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

The nightly path refreshes source tables and passes a final schema-driven freshness gate before triggering the memory-bounded aggregation chain. Extraction tasks are grouped by operational resource pools rather than application package, while cube tasks run sequentially and derive their commands from the part registry.

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
  Airflow->>Extract: refresh insiders
  Airflow->>Extract: build identity tables
  Airflow->>Extract: run identity-consuming SEC tasks
  Airflow->>Extract: facts and employees as sibling walks
  Extract->>Store: incremental upserts
  Airflow->>Extract: run schema-driven freshness gate
  Extract->>Store: read each declared table frontier
  Extract-->>Airflow: green only when every declared table is current
  Airflow->>Peers: deduce peers
  Peers-->>Airflow: persist peer dictionary
  Airflow->>Cube: run registered parts sequentially
  Cube->>Store: write cube parts and cube
  Airflow->>Cube: run cube status
  Airflow->>Predict: trigger daily scoring
~~~

## Steps

1. Seed or load the S&P 500 universe through [data extraction](../modules/data-extract.md).
2. Refresh insider evidence (pending zip quarters fill the filings EDGAR missed, then EDGAR lists each ticker from its own latest stored filing date minus 7 days, all into `insider_transactions`; only the EDGAR run records the completeness manifest entry), build `identity-tables`, then fan out every issuer-identity SEC consumer. The 13F manager chain remains independent because it does not resolve issuer filing history.
3. Run SEC facts/history and standalone employee headcount as sibling tasks. The Sharadar merge waits for both, but either task can be repaired and rerun without replaying the other.
4. Run `extract-earnings-calls` as one task in the default pool. It compares the defeatbeta transcripts file's content hash with the one recorded in the run manifest and exits in seconds when the daily build has not changed and no `reconcile_days` full comparison is due; otherwise it reads only the row groups holding new or re-issued calls. The split, sentiment and embeddings run later in `build-text`.
5. Retry an extractor that raises up to three times. Per-ticker SEC API walks consume the whole SEC pool so their process-local rate limiters cannot exceed the shared request budget.
6. Run the final extraction gate. Its complete table inventory, cadence, and publication date column come from `freshness_tables()` in [schema.py](../../src/data_store/schema.py); cadence tolerances come from [constants.py](../../src/constants/constants.py). The gate also inherits three retries. `earnings_call_sections` declares quarterly freshness on `as_of`. Google Trends is not scheduled and declares no freshness; there is no Wikipedia pageviews task.
7. On green only, trigger [peer deduction](../modules/data-peers.md), then the [cube-build flow](./cube-build.md).
8. Run the cube part-status gate and trigger [model prediction](./model-training-and-prediction.md) only when that gate succeeds.

## Failure modes

A source task that raises is retried, so extraction itself runs again. Completeness-sensitive EDGAR walks keep successful idempotent rows but do not advance their manifest when any ticker fails. Identity-scope fingerprints rewind only changed tickers on the next complete attempt, while employee accession outcomes remain independent of XBRL fact completion. The final gate compares each schema-declared table's maximum publication date with its cadence tolerance; an absent or stale table makes the gate retry and blocks aggregation if it remains red. The gate does not maintain a second source-to-table map, so its retry rechecks database state rather than guessing which upstream extractor owns a table. The cube status gate likewise blocks prediction. Aggregation remains sequential because the largest part determines peak memory; parallelizing part builds would add their working sets.

## Related

- [DAGs and infrastructure](../modules/dags-and-infrastructure.md)
- [Source availability](../concepts/source-availability.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
