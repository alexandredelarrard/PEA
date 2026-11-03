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

The repository includes operational scripts around the core package. Portfolio analysis is a standalone offline HTML report, with no application server. Scripts profile data, investigate source defects, manage registrant and DEF 14A migrations, and create definition-of-done reports. `scripts/` is deletable: nothing in `src/` or `tests/` imports it. The tools that tests import live in `tests/support/`, and the DDL generator lives in `sql/`.

## Responsibilities

- Offer an intentionally manual scratch driver for local investigation.
- Generate managed SQL from the table registry.
- Produce diagnostics, baselines, catalogues, and acceptance evidence.
- Execute bounded backfills, identity research, or registrant-cutover verification that does not belong in the recurring package API.

## Public API / entry points

- [main.py](../../main.py) constructs a context and selected steps but leaves execution calls commented until explicitly enabled.
- Individual files under `scripts/` and `tests/support/` expose command-line `main()` functions where applicable.

## Key files

- [sql/generate_schema_sql.py](../../sql/generate_schema_sql.py) renders [sql/schema.sql](../../sql/schema.sql) from the registry (`python sql/generate_schema_sql.py`).
- [cube_feature_catalogue.py](../../tests/support/cube_feature_catalogue.py), [cube_governance_catalogue.py](../../tests/support/cube_governance_catalogue.py), [cube_institutionals_catalogue.py](../../tests/support/cube_institutionals_catalogue.py) and [cube_sec_thirteen_catalogue.py](../../tests/support/cube_sec_thirteen_catalogue.py) in `tests/support/` maintain feature inventories.
- [institutionals_feature_quality.py](../../tests/support/institutionals_feature_quality.py) runs read-only quality gates on the institutional feature part against a frozen input manifest. Its source-table list names the one `insider_transactions` table, so a manifest frozen before the single-table insider change is rejected with `manifest source tables differ`; re-freeze it before reuse.
- [detect_registrant_cutovers.py](../../tests/support/detect_registrant_cutovers.py), [identity_regression_gate.py](../../tests/support/identity_regression_gate.py) and [def14a_legacy_purge.py](../../tests/support/def14a_legacy_purge.py) are the test-imported registrant, identity and DEF 14A tools; other research scripts remain isolated under `scripts/`.

## Dependencies

Operational scripts reuse [Context](./runtime-and-shared-utils.md), [DataStore](./data-store.md), [modelling](./modelling.md), [portfolio](./portfolio.md), and [validation](./validation.md) rather than introducing another application layer.

## Participates in

Scripts support schema evolution, source investigations, and the [validate-a-change guide](../guides/validate-a-change.md).

## Related

- [Run the pipeline](../guides/run-the-pipeline.md)
- [Data platform](../architecture/data-platform.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
