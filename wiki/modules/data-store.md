---
title: Data store
description: The sole SQL boundary, table registry, generated DDL, and bounded DataFrame I/O facade.
type: module
tags:
  - wiki
  - module
---
# Data store

## Summary

`src/data_store` is the only application package allowed to issue SQL or translate DataFrames to database operations. Its registry declares table identity and lifecycle; its facade implements bounded reads, streaming, upserts, tail rewrites, replacement, deletion, and schema evolution.

## Responsibilities

- Resolve every table through a typed registry.
- Enforce primary keys, date columns, projections, and freshness metadata.
- Compile safe server-side filters and inclusive date bounds.
- Support PostgreSQL production and SQLite-backed tests.
- Generate managed DDL and evolve newly observed columns.

## Public API / entry points

`DataStore` in [store.py](../../src/data_store/store.py) exposes `exists`, `columns`, `row_count`, `bounds`, `max_date`, `max_date_by`, `distinct`, `load`, `iter_load`, `save`, `bulk_seed`, `append_tail`, `delete`, `replace`, and `ensure_columns`.

`Tables` and the frozen `Table` dataclass live in [schema.py](../../src/data_store/schema.py). Missing and empty reads use explicit errors from [errors.py](../../src/data_store/errors.py).

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
