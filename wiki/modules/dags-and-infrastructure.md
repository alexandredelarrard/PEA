---
title: DAGs and infrastructure
description: Airflow scheduling, Docker services, isolated runtime dependencies, and database containers.
type: module
tags:
  - wiki
  - module
source_commit: 55d6e2d199fa9a83aca7d5442549eeb66eedd189
---
# DAGs and infrastructure

## Summary

Infrastructure combines a PostgreSQL 16 pipeline database with an Airflow deployment. Airflow schedules the same package CLI commands used locally, while its image keeps pipeline dependencies in a separate virtual environment to avoid conflicts with Airflow's own environment.

## Responsibilities

- Run the nightly extraction fan-out with source-specific pools, placing `identity-tables` before every issuer-identity SEC consumer.
- Keep SEC facts/history and employee headcount as independent tasks; the Sharadar merge waits for SEC facts/history only.
- Keep `sharadar_actions` an independent fetcher (no edge beyond the universe seed) and keep `fundamentals >> fundamentals_sharadar`: the merged `fundamentals_history` reads `fundamentals_history_sec`, and `fundamentals_sharadar` fetches the actions itself before the merge, so an actions failure never blocks it and a failed SEC `fundamentals` skips only that night's `fundamentals_sharadar`. The actions are therefore fetched twice a night by two unordered tasks; accepted, since the upsert is idempotent and the cost is one duplicate Sharadar request (at worst a lock wait). A refresh of `sharadar_actions` or `prices_splits` never corrects a restated price basis; only a re-pull of the ticker's history does. The extraction schedule is `0 1 * * *`, its `start_date` is 2026-11-11 with `catchup=False`, so no scheduled run exists before that date.
- Trigger a sequential, memory-bounded cube build.
- Retrain models weekly and score the latest cube daily.
- Initialize and operate the database, Airflow metadata database, scheduler, and webserver.
- Mount source, configs, data, and logs into the scheduled runtime.

## Public API / entry points

- [dag_data_extraction.py](../../src/dags/dag_data_extraction.py) schedules insiders (two tasks into the one `insider_transactions` table: `insider-zip`, the pending zip quarters, in the `sec_bulk` pool; `insider-edgar`, the EDGAR walk, in the `sec_api` pool after it, so it never runs beside another EDGAR walk; both are identity consumers), after the identity downloads, identity tables and propagation, like every identity-consuming SEC task; its standalone employee task is a sibling of SEC fundamentals rather than a side effect.
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
