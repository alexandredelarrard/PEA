---
title: DAGs and infrastructure
description: Airflow scheduling, Docker services, isolated runtime dependencies, and database containers.
type: module
tags:
  - wiki
  - module
source_commit: 1b5f33a2a210336c14758ce4897008c72da1d42c
---
# DAGs and infrastructure

## Summary

Infrastructure combines a PostgreSQL 16 pipeline database with an Airflow deployment. Airflow schedules the same package CLI commands used locally, while its image keeps pipeline dependencies in a separate virtual environment to avoid conflicts with Airflow's own environment.

## Responsibilities

- Start the extraction night with `[sec_company_tickers, sharadar_tickers, sharadar_actions, sec_submissions] >> universe_sync >>` every fetcher (five `universe --only` parts of `StepExtractUniverse`; the four evidence tasks refresh in parallel, `sec_submissions` in the `sec_bulk` pool): the universe sync writes `universe_tickers` from [universe.yml](../../configs/universe.yml), so a failed sync stops the night.
- Run the nightly extraction fan-out with source-specific pools, placing `zips_download >> identity` before every issuer-identity SEC consumer and one `edgar_index` refresh before the seven EDGAR walks.
- Keep SEC facts/history and employee headcount as independent tasks; the Sharadar merge waits for SEC facts/history only.
- Keep `sharadar_actions` an independent fetcher (no edge beyond the universe sync) Fundamentals are two independent producers with no path between them: `identity >> edgar_index >> sec_fundamentals_facts >> sec_fundamentals_history`, and `identity >> fundamentals_sharadar` (an identity consumer: its history is keyed through the lineage). `fundamentals_check` runs after both histories on ALL_DONE, so either side failing still runs it. The `fundamentals_sharadar` task fetches the actions itself before its history build, so an actions failure never blocks it. The actions are therefore fetched twice a night by two unordered tasks; accepted, since the upsert is idempotent and the cost is one duplicate Sharadar request (at worst a lock wait). A refresh of `sharadar_actions` or `prices_splits` never corrects a restated price basis; only a re-pull of the ticker's history does. The extraction schedule is `0 1 * * *`, its `start_date` is 2026-11-11 with `catchup=False`, so no scheduled run exists before that date.
- Trigger a sequential, memory-bounded cube build.
- Retrain models weekly and score the latest cube daily.
- Initialize and operate the database, Airflow metadata database, scheduler, and webserver.
- Mount source, configs, data, and logs into the scheduled runtime.

## Public API / entry points

- [dag_data_extraction.py](../../src/dags/dag_data_extraction.py) schedules the step commands: `zips_download` (`zips-download`, `sec_bulk` pool) `>> identity` (`identity`, ALL_DONE) `>>` the identity consumers; the zip ingests as two `sec_bulk` tasks, `zips_ingest_insider` (`zips-ingest --only insider`) and `zips_ingest_datasets` (`zips-ingest --only notes --only ftd`); `identity >> edgar_index` (`edgar-index`, `sec_api` pool) `>>` the seven EDGAR walks (`insider_edgar`, `sec_13d`, `sec_13g`, `sec_8k_items`, `filing_text`, `def14a_edgar`, `sec_fundamentals_facts`), each in the `sec_api` pool so no two EDGAR walks run together; and `zips_ingest_insider >> insider_edgar`, the two halves of the one `insider_transactions` table; its standalone employee task is a sibling of SEC fundamentals rather than a side effect.
- [dag_data_aggregation.py](../../src/dags/dag_data_aggregation.py) chains peer deduction, registered cube parts, assembly, status, and downstream prediction. The extraction DAG's freshness report and aggregation trigger, and every aggregation task, run on `ALL_DONE`, so a failed source or part never stops the night; `modelling predict` alone refuses stale inputs. The 13F new-ticker backfill is a task of the extraction DAG, after the nightly 13F walk.
- [dag_modelling.py](../../src/dags/dag_modelling.py) runs training, fresh assessment, portfolio backtest and full-history production fitting with the same generated run ID.
- [dag_strat_prediction.py](../../src/dags/dag_strat_prediction.py) scores the newest cube; its obsolete ledger task has been removed.
- [docker-compose.yml](../../docker-compose.yml) is the deployment manifest.

## Key files

- [airflow/Dockerfile.airflow](../../airflow/Dockerfile.airflow) creates `/opt/pipeline` and installs runtime dependencies.
- [docker-compose.yml](../../docker-compose.yml) defines `db`, `airflow-db`, `airflow-init`, `airflow-scheduler`, and `airflow-webserver`.
- [test_dag_matches_part_registry.py](../../tests/dags/test_dag_matches_part_registry.py) proves that the aggregation chain derives from the cube-part registry.

## Dependencies

The DAGs call [runtime and CLI](./runtime-and-shared-utils.md) commands. Docker Compose provides both PostgreSQL services and the shared environment and volume wiring.

## Participates in

- [Nightly data refresh](../flows/nightly-data-refresh.md)
- [Model training and daily prediction](../flows/model-training-and-prediction.md)
- [Orchestration and runtime architecture](../architecture/orchestration-and-runtime.md)

## Related

- [Run the pipeline](../guides/run-the-pipeline.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
- [Cube parts](../concepts/cube-parts.md)
