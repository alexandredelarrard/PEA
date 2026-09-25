---
title: Orchestration and runtime
description: How Context, Step classes, the dynamic CLI, Airflow, Docker, and the Streamlit app run the pipeline.
type: architecture
tags:
  - wiki
  - architecture
---
# Orchestration and runtime

## Summary

A shared `Context` constructs configuration, logging, random state, paths, environment access, and the data store. Package-level `Step` classes expose one public `run()` method, Click discovers package CLIs dynamically, and Airflow invokes those same CLI commands inside an isolated pipeline virtual environment. The supported commands and environment assumptions are recorded in [run the pipeline](../guides/run-the-pipeline.md).

## Diagram

~~~mermaid
flowchart TD
  Config[OmegaConf files] --> Context[Context]
  Env[Environment variables] --> Context
  Context --> Step[Step subclass]
  ModuleCLI[Package cli.py] --> Step
  RootCLI[python -m src] --> ModuleCLI
  Airflow[Airflow BashOperator] --> RootCLI
  Compose[Docker Compose] --> Airflow
  Compose --> DB[(PostgreSQL)]
  Streamlit[Streamlit app] --> Context
~~~

## Key components

- [src/__main__.py](../../src/__main__.py) delegates to the dynamic `CLI` in [src/cli.py](../../src/cli.py).
- Shared option declarations live in [command_line_interface.py](../../src/constants/command_line_interface.py).
- `Context` and `get_config_context` live in [context.py](../../src/context.py); the base [Step](../concepts/step-pattern.md) lives in [step.py](../../src/utils/step.py).
- Four DAG modules under [src/dags](../../src/dags/) schedule extraction, aggregation, weekly training, and daily prediction.
- [docker-compose.yml](../../docker-compose.yml) defines the pipeline database, Airflow metadata database, initializer, scheduler, and webserver. [Dockerfile.airflow](../../airflow/Dockerfile.airflow) installs the pipeline into `/opt/pipeline` separately from Airflow's own dependencies.
- [app/app.py](../../app/app.py) is the Streamlit portfolio dashboard. [main.py](../../main.py) is a scratch driver and does not execute a stage unless a `run()` call is explicitly enabled.

## Design decisions

The CLI is plugin-shaped: any immediate `src/*/cli.py` can become a command group without editing a central registry. Airflow calls the same commands developers use locally, reducing divergence between scheduled and manual execution.

Airflow uses Python 3.12 while the application requires Python 3.13 on the host. The Docker image isolates the pipeline dependencies in a separate virtual environment because Airflow's dependency versions differ; this split is explicit in [airflow/Dockerfile.airflow](../../airflow/Dockerfile.airflow).

## Related

- [Runtime and shared utilities](../modules/runtime-and-shared-utils.md)
- [DAGs and infrastructure](../modules/dags-and-infrastructure.md)
- [Nightly data refresh](../flows/nightly-data-refresh.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
