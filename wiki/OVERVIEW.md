---
title: Codebase Wiki — Overview
description: Canonical internal hub for architecture, modules, flows, concepts, guides, references, and backlog.
profile: internal/standard
source_commit: 054f0e1d4eb6ffde1bc1b9a48dc3d34170c12c01
tags:
  - wiki
  - overview
  - stock-pick-strat
---
# stock_pick_strat codebase wiki

## What this is

`stock_pick_strat` is an internal quantitative research and production pipeline for S&P 500 equities. It extracts market, fundamental, ownership, governance, and alternative data into PostgreSQL; turns those sources into a point-in-time, peer-relative feature cube; trains cross-sectional models; and blends independent strategy sleeves into a tradeable portfolio.

This wiki is the canonical documentation surface for code navigation, data contracts, operations, testing, and deferred work. Start with the task map below, then inspect the linked source before changing code. Package metadata lives in [pyproject.toml](../pyproject.toml); [docker-compose.yml](../docker-compose.yml) defines PostgreSQL and Airflow services.

## Architecture at a glance

~~~mermaid
flowchart LR
  Sources[Market SEC vendor and text sources] --> Extract[Extraction steps]
  Extract --> Store[(PostgreSQL)]
  Store --> Peers[Peer deduction]
  Store --> Parts[Cube part builders]
  Peers --> Parts
  Parts --> Cube[(Point in time cube)]
  Cube --> Models[Cross sectional ensemble]
  Models --> Sleeves[Strategy sleeves]
  Store --> Sleeves
  Sleeves --> Portfolio[ERC portfolio blend]
  Portfolio --> Ledger[(Strategy trade ledger)]
  Airflow[Airflow DAGs] --> Extract
  Airflow --> Parts
  Airflow --> Models
  CLI[Dynamic Click CLI] --> Extract
  CLI --> Parts
  CLI --> Models
  App[Streamlit] --> Portfolio
~~~

The narrow production boundary is deliberate: all tabular application I/O passes through [DataStore](./modules/data-store.md), while manual and scheduled orchestration invoke the same package-level CLI commands.

## Task map

| Task | Read first |
| --- | --- |
| Understand boundaries, stages, and entry points | [System overview](./architecture/system-overview.md) and [orchestration](./architecture/orchestration-and-runtime.md) |
| Touch a table, grain, PK, date column, projection, or freshness rule | [Table catalog](./reference/table-catalog.md) |
| Know what is physically populated in the local database | [Live database](./reference/live-database.md) |
| Add or debug a fetcher | [Data sources](./reference/data-sources.md) and [add a data source](./guides/add-a-data-source.md) |
| Read or write tabular data | [Data access and storage](./guides/data-access.md) |
| Find or add a configuration knob | [Configuration](./reference/configuration.md) |
| Change models, sleeves, portfolio allocation, or the ledger | [Modelling, strategies, and portfolio](./reference/modelling-and-portfolio.md) |
| Write Python under `src/` | [Coding standards](./guides/coding-standards.md) |
| Write or execute tests | [Testing and validation](./guides/testing.md) |
| Execute the CLI, Docker, Airflow, or routine validators | [Run the pipeline](./guides/run-the-pipeline.md) |
| Run destructive or high-cost historical recovery | [Large backfills and recovery](./guides/large-backfills-and-recovery.md) |
| Review accepted deferred data work | [TODO](./TODO.md) |
| Verify documentation migration and deletion readiness | [Documentation coverage](./reference/documentation-coverage.md) |

## Navigation

### Architecture

- [System overview](./architecture/system-overview.md)
- [Data platform](./architecture/data-platform.md)
- [Orchestration and runtime](./architecture/orchestration-and-runtime.md)
- [Feature-to-portfolio pipeline](./architecture/feature-to-portfolio.md)
- [Point-in-time and quality controls](./architecture/point-in-time-and-quality.md)

### Modules

- [Runtime and shared utilities](./modules/runtime-and-shared-utils.md)
- [Constants and configuration](./modules/constants-and-configuration.md)
- [Data store](./modules/data-store.md)
- [Data extraction](./modules/data-extract.md)
- [GPT extraction](./modules/gpt-extract.md)
- [Peer deduction](./modules/data-peers.md)
- [Cube aggregation](./modules/data-aggregate.md)
- [Modelling](./modules/modelling.md)
- [Strategies](./modules/strategies.md)
- [Portfolio](./modules/portfolio.md)
- [Validation](./modules/validation.md)
- [DAGs and infrastructure](./modules/dags-and-infrastructure.md)
- [Application and scripts](./modules/application-and-scripts.md)
- [Tests](./modules/tests.md)

### Flows

- [Nightly data refresh](./flows/nightly-data-refresh.md)
- [SEC and LLM extraction](./flows/sec-llm-extraction.md)
- [Cube build](./flows/cube-build.md)
- [Model training and daily prediction](./flows/model-training-and-prediction.md)
- [Portfolio to trade ledger](./flows/portfolio-to-trade-ledger.md)

### Concepts

- [Step pattern](./concepts/step-pattern.md)
- [Table registry and store boundary](./concepts/table-registry-and-store-boundary.md)
- [Point-in-time data](./concepts/point-in-time-data.md)
- [Cube parts](./concepts/cube-parts.md)
- [Peer-relative features](./concepts/peer-relative-features.md)
- [Source availability](./concepts/source-availability.md)
- [Strategy sleeves](./concepts/strategy-sleeves.md)

### Guides

- [Run the pipeline](./guides/run-the-pipeline.md)
- [Large backfills and recovery](./guides/large-backfills-and-recovery.md)
- [Data access and storage](./guides/data-access.md)
- [Coding standards](./guides/coding-standards.md)
- [Testing and validation](./guides/testing.md)
- [Add a data source](./guides/add-a-data-source.md)
- [Add a cube feature](./guides/add-a-cube-feature.md)
- [Add a model or sleeve](./guides/add-a-model-or-sleeve.md)
- [Validate a change](./guides/validate-a-change.md)

### Reference

- [Table catalog](./reference/table-catalog.md)
- [Data sources](./reference/data-sources.md)
- [Live database](./reference/live-database.md)
- [Configuration](./reference/configuration.md)
- [Modelling, strategies, and portfolio](./reference/modelling-and-portfolio.md)
- [Documentation coverage](./reference/documentation-coverage.md)

### Project records

- [TODO](./TODO.md)
- [Wiki generation log](./log.md)

## Documentation contract

The wiki is structured by intent:

- architecture pages explain boundaries and design decisions;
- module pages map packages, APIs, dependencies, and participating flows;
- flow pages trace end-to-end execution and failure modes;
- concept pages define load-bearing abstractions;
- guides are task-oriented procedures; and
- reference pages hold dense contracts and dated operational facts.

Every durable claim should link to an inspected source file or another canonical wiki page. When behavior changes, update the narrowest affected pages and append one entry to [the wiki log](./log.md).
