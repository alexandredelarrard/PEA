---
title: Application and scripts
description: Streamlit dashboard, scratch entry point, schema generator, diagnostics, evidence, and one-off operational tools.
type: module
tags:
  - wiki
  - module
---
# Application and scripts

## Summary

The repository includes a Streamlit dashboard and a broad set of operational scripts around the core package. The dashboard runs the configured portfolio over existing model artifacts. Scripts generate DDL, profile data, produce feature catalogues and evidence, investigate source defects, manage registrant and DEF 14A migrations, and create definition-of-done reports.

## Responsibilities

- Present portfolio and sleeve results interactively.
- Offer an intentionally manual scratch driver for local investigation.
- Generate managed SQL from the table registry.
- Produce diagnostics, baselines, catalogues, and acceptance evidence.
- Execute bounded backfills, identity research, or cutover verification that does not belong in the recurring package API.

## Public API / entry points

- Run [app/app.py](../../app/app.py) with Streamlit for the dashboard.
- [main.py](../../main.py) constructs a context and selected steps but leaves execution calls commented until explicitly enabled.
- Individual files under [scripts](../../scripts/) expose command-line `main()` functions where applicable.

## Key files

- [app/app.py](../../app/app.py) renders portfolio-first KPIs, allocation, correlations, equity curves, and sleeve tabs.
- [scripts/generate_schema_sql.py](../../scripts/generate_schema_sql.py) renders [sql/schema.sql](../../sql/schema.sql) from the registry.
- [scripts/dod/data_profile.py](../../scripts/dod/data_profile.py), [scripts/dod/modelling_report.py](../../scripts/dod/modelling_report.py), and [scripts/dod/refactor_metrics.py](../../scripts/dod/refactor_metrics.py) produce completion evidence.
- [scripts/cube_feature_catalogue.py](../../scripts/cube_feature_catalogue.py), [scripts/cube_governance_catalogue.py](../../scripts/cube_governance_catalogue.py), and [scripts/cube_institutionals_catalogue.py](../../scripts/cube_institutionals_catalogue.py) maintain feature inventories.
- Registrant, identity, and DEF 14A research scripts remain isolated under [scripts](../../scripts/).

## Dependencies

The app and operational scripts reuse [Context](./runtime-and-shared-utils.md), [DataStore](./data-store.md), [modelling](./modelling.md), [portfolio](./portfolio.md), and [validation](./validation.md) rather than introducing another application layer.

## Participates in

The dashboard is a consumer of the [feature-to-portfolio pipeline](../architecture/feature-to-portfolio.md). Scripts support schema evolution, source investigations, and the [validate-a-change guide](../guides/validate-a-change.md).

## Related

- [Run the pipeline](../guides/run-the-pipeline.md)
- [Data platform](../architecture/data-platform.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
