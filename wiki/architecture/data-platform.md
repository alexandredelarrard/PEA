---
title: Data platform
description: PostgreSQL storage, table registry, access facade, and artifact boundaries.
type: architecture
tags:
  - wiki
  - architecture
---
# Data platform

## Summary

PostgreSQL is the authoritative store for tabular pipeline state. A frozen table registry describes grain, primary keys, incremental date columns, projections, freshness, and managed-versus-part lifecycle; all application reads and writes pass through one `DataStore` facade. The detailed contracts are documented in [table catalog](../reference/table-catalog.md) and [data access and storage](../guides/data-access.md).

## Diagram

~~~mermaid
flowchart LR
  Code[Pipeline code] --> Store[DataStore facade]
  Registry[Tables registry] --> Store
  Store --> PG[(PostgreSQL 16)]
  Registry --> DDL[DDL generator]
  DDL --> Schema[sql schema]
  Parts[Unmanaged cube parts] --> PG
  Artifacts[Models caches plots peers] --> Disk[(data artifacts)]
~~~

## Key components

- `Table`, `Tables`, `ALL`, `MANAGED`, and `PARTS` are defined in [schema.py](../../src/data_store/schema.py).
- `DataStore` exposes bounded introspection, projected reads, streaming reads, upserts, replacement, tail rewrites, and schema evolution in [store.py](../../src/data_store/store.py).
- Managed DDL is rendered from the registry by [ddl.py](../../src/data_store/ddl.py) into [sql/schema.sql](../../sql/schema.sql).
- The engine factory in [db.py](../../src/utils/db.py) selects `DATABASE_URL` or compose-compatible PostgreSQL settings and also supports SQLite for tests.
- [Context](../modules/runtime-and-shared-utils.md) owns the store instance and the separate filesystem artifact paths.

## Design decisions

The [table registry and store boundary](../concepts/table-registry-and-store-boundary.md) prevents SQL dialect and query construction from leaking into feature code. Large tables require projection and row scoping; cube-sized reads use `iter_load` so the driver does not buffer the complete result.

Managed source and aggregate tables keep generated DDL and are emptied on replacement. Unmanaged `cube_part_*` tables are dropped and recreated because their columns legitimately change with the feature set. The final cube is streamed in chunks by [step_assemble_cube.py](../../src/data_aggregate/transformers/step_assemble_cube.py).

Non-tabular products do not masquerade as tables: models, diagnostics, downloaded SEC archives, transcripts, and the peer dictionary use paths resolved by [context.py](../../src/context.py).

## Related

- [Data store module](../modules/data-store.md)
- [Point-in-time data](../concepts/point-in-time-data.md)
- [Cube parts](../concepts/cube-parts.md)
- [Add a data source](../guides/add-a-data-source.md)
