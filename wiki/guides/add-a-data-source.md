---
title: Add a data source
description: Wire a resumable source fetcher through the table registry, extraction step, CLI, tests, and validation.
type: guide
tags:
  - wiki
  - guide
---
# Add a data source

## Goal

Add a source without bypassing the store boundary, losing point-in-time provenance, or creating a full-table resume path. Read [data sources](../reference/data-sources.md), the [table catalog](../reference/table-catalog.md), and [data access and storage](./data-access.md) before editing.

## Steps

1. Define the source's grain, identifiers, publication clock, primary key, date column, and freshness semantics.
2. Add exactly one `Table` declaration to [schema.py](../../src/data_store/schema.py), including the smallest safe read projection.
3. Regenerate [sql/schema.sql](../../sql/schema.sql) through [scripts/generate_schema_sql.py](../../scripts/generate_schema_sql.py) when the table is managed.
4. Implement the fetcher under the owning [data_extract utils package](../../src/data_extract/utils/) and accept `Context` rather than constructing a store.
5. Resume from `store.max_date` or `store.max_date_by` using the source's actual retrieval grain.
6. Save bounded work per ticker, filing, or source chunk so interruption loses minimal progress.
7. Add the fetcher to the correct transformer step and expose a source command in [data_extract/cli.py](../../src/data_extract/cli.py).
8. Add an Airflow task only when scheduled refresh is appropriate, choosing the resource pool by operational constraint.
9. Add parsing tests with known-truth fixtures and economic/incremental checks with bounded real data.
10. Add validation declarations or a dedicated check when availability, bounds, or reconciliation needs an explicit contract.

## Relevant code

- Store registry: [src/data_store/schema.py](../../src/data_store/schema.py)
- Store facade: [src/data_store/store.py](../../src/data_store/store.py)
- Incremental helpers: [data_extract/utils/common/incremental.py](../../src/data_extract/utils/common/incremental.py)
- EDGAR driver: [data_extract/utils/common/edgar_driver.py](../../src/data_extract/utils/common/edgar_driver.py)
- Extraction orchestrator: [step_extract_all_data.py](../../src/data_extract/step_extract_all_data.py)

## Gotchas

Do not hardcode table names, SQL, URLs, thresholds, or credentials in the fetcher. Keep identifiers such as CIK and CUSIP as text. A source start date does not make every ticker-date eligible; model missingness through [source availability](../concepts/source-availability.md). Changes to the registry, constants, SQL, configs, or data are risk-zone edits under [coding standards](./coding-standards.md).

## Related

- [Data extraction](../modules/data-extract.md)
- [Table registry and store boundary](../concepts/table-registry-and-store-boundary.md)
- [SEC and LLM extraction](../flows/sec-llm-extraction.md)
