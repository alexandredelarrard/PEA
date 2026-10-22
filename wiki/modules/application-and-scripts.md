---
source_commit: 55d6e2d199fa9a83aca7d5442549eeb66eedd189
title: Operational scripts
description: Scratch entry point, diagnostics, evidence and bounded operational tools.
type: module
tags:
  - wiki
  - module
---
# Operational scripts

## Summary

The repository includes operational scripts around the core package. Portfolio analysis is a standalone offline HTML report, with no application server. Scripts generate DDL, profile data, produce feature catalogues and evidence, investigate source defects, manage registrant and DEF 14A migrations, and create definition-of-done reports.

## Responsibilities

- Offer an intentionally manual scratch driver for local investigation.
- Generate managed SQL from the table registry.
- Produce diagnostics, baselines, catalogues, and acceptance evidence.
- Execute bounded backfills, identity research, or registrant-cutover verification that does not belong in the recurring package API.

## Public API / entry points

- [main.py](../../main.py) constructs a context and selected steps but leaves execution calls commented until explicitly enabled.
- Individual files under `scripts/` expose command-line `main()` functions where applicable.

## Key files

- [scripts/generate_schema_sql.py](../../scripts/generate_schema_sql.py) renders [sql/schema.sql](../../sql/schema.sql) from the registry.
- [scripts/cube_feature_catalogue.py](../../scripts/cube_feature_catalogue.py), [scripts/cube_governance_catalogue.py](../../scripts/cube_governance_catalogue.py), and [scripts/cube_institutionals_catalogue.py](../../scripts/cube_institutionals_catalogue.py) maintain feature inventories.
- [institutionals_feature_quality.py](../../scripts/institutionals_feature_quality.py) runs read-only quality gates on the institutional feature part against a frozen input manifest. Its source-table list names the one `insider_transactions` table, so a manifest frozen before the single-table insider change is rejected with `manifest source tables differ`; re-freeze it before reuse.
- Registrant, identity, and DEF 14A research scripts remain isolated under `scripts/`.

## Dependencies

Operational scripts reuse [Context](./runtime-and-shared-utils.md), [DataStore](./data-store.md), [modelling](./modelling.md), [portfolio](./portfolio.md), and [validation](./validation.md) rather than introducing another application layer.

## Participates in

Scripts support schema evolution, source investigations, and the [validate-a-change guide](../guides/validate-a-change.md).

## Related

- [Run the pipeline](../guides/run-the-pipeline.md)
- [Data platform](../architecture/data-platform.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
