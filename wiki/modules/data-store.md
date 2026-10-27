---
title: Data store
description: The sole SQL boundary, table registry, generated DDL, and bounded DataFrame I/O facade.
type: module
tags:
  - wiki
  - module
source_commit: aa9ac35ff94b6992ac43106230f63e3d50082cab
---
# Data store

## Summary

`src/data_store` is the only application package allowed to issue SQL or translate DataFrames to database operations. Its registry declares table identity and lifecycle; its facade implements generic bounded reads, transactions, streaming, upserts, tail rewrites, replacement, deletion, and explicit schema changes.

## Responsibilities

- Resolve every table through a typed registry.
- Enforce primary keys, date columns, projections, and freshness metadata.
- Compile safe server-side filters and inclusive date bounds.
- Support PostgreSQL production and SQLite-backed tests.
- Render managed DDL and keep existing managed schemas stable during ingestion.

## Public API / entry points

`DataStore` in [store.py](../../src/data_store/store.py) exposes `exists`, `columns`, `row_count`, `bounds`, `max_date`, `max_date_by`, `key_stats`, `distinct`, `load`, `iter_load`, `summarize`, `load_cohort`, `same_rows`, `transaction`, `temporary`, `insert`, `update`, `save`, `bulk_seed`, `append_tail`, `delete`, `replace`, `drop`, and `ensure_columns`.

Filters accept a dictionary, a complete non-null primary-key DataFrame, or a flat union of those shapes. `summarize` groups counts/minima/maxima and filtered counts on the server; `load_cohort` expands touched groups without loading an entire source table. Cold names-only checks use lightweight inspection. Other operations reuse generic SQLAlchemy schema metadata, shared with bound transactions and invalidated by explicit schema changes or rollback. Exhaust or close a stream to release its connection.

`with store.transaction(lock_key) as bound:` makes reads, strict inserts, upserts, updates, deletes and COPY share one connection and commit. The owner captures fresh input at the start under its lock and then validates and publishes it. Nested transactions are rejected. Ordinary unbound `replace` and `append_tail` retain their existing separate commits; owners requiring atomicity explicitly use the bound store.

Frame bindings preserve SQL NULLs, carried fields and omitted-field upsert behavior. Existing managed tables never gain columns or legacy defaults during ordinary writes. `ensure_columns` remains an explicit generic administration helper. Fresh databases use [canonical SQL](../../sql/schema.sql); selected temporary source schemas use `DataStore.temporary(tables)`.

Table rules live with their users: identity changes in [identity_store.py](../../src/data_extract/utils/common/identity_store.py), insider sources in [insider_store.py](../../src/data_extract/utils/institutionals/insider_store.py), FTD and FINRA in their existing fetchers, immutable filings beside `_Saver` in [fetch_13f.py](../../src/data_extract/utils/institutionals/fetch_13f.py), and completeness/output interpretation in the aggregate owners. The store contains no concrete-table dispatch or migration preparation.

`Tables` and the frozen `Table` dataclass live in [schema.py](../../src/data_store/schema.py). EDGAR tables declare `desc`, their complete ordered `retrieval_columns`, and empty-marker stamps there; `Resume.forms` owns their filing scope. The extraction driver and frame builders consume this metadata without repeating it in fetch declarations. `read_columns` remains the separate bounded consumer projection. Missing and empty reads use explicit errors from [errors.py](../../src/data_store/errors.py).

## Key files

- [src/data_store/schema.py](../../src/data_store/schema.py) is the single table registry.
- [src/data_store/store.py](../../src/data_store/store.py) is the application I/O facade.
- [src/data_store/ddl.py](../../src/data_store/ddl.py) renders registry definitions into SQL.
- [sql/schema.sql](../../sql/schema.sql) is generated managed-table DDL.
- [test_store_boundary.py](../../tests/data_store/test_store_boundary.py) rejects SQL leakage elsewhere.

## Dependencies

The package depends on pandas, NumPy, and SQLAlchemy 2.0. Engine construction remains in [utils/db.py](../../src/utils/db.py), which is the one permitted non-store SQLAlchemy boundary.

## Participates in

All pipeline stages persist through this package. It underpins [data extraction](./data-extract.md), [cube aggregation](./data-aggregate.md), [modelling](./modelling.md), [portfolio](./portfolio.md), and read-only [validation](./validation.md).

## Related

- [Data platform architecture](../architecture/data-platform.md)
- [Table registry and store boundary](../concepts/table-registry-and-store-boundary.md)
- [Data access and storage](../guides/data-access.md)
