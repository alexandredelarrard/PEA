---
title: Table registry and store boundary
description: The single source of truth for table metadata and the only application SQL facade.
type: concept
tags:
  - wiki
  - concept
---
# Table registry and store boundary

## Definition

Every database table is declared once as a frozen `Table` in [schema.py](../../src/data_store/schema.py). The declaration carries its name, primary key, kind, date and ticker columns, date coercions, projection, optional columns, freshness, vector shape, and managed lifecycle. Application code passes that declaration to `DataStore` rather than writing table names or SQL.

## Why it matters

A central registry makes grain and incremental semantics inspectable and prevents a feature builder from silently disagreeing with DDL or another reader. The store boundary concentrates parameter binding, projection, streaming, dialect differences, upsert behavior, replacement, and schema evolution in one place. [test_store_boundary.py](../../tests/data_store/test_store_boundary.py) enforces the rule mechanically.

## Where it lives

- Registry: [src/data_store/schema.py](../../src/data_store/schema.py)
- Facade: [src/data_store/store.py](../../src/data_store/store.py)
- Generated DDL: [sql/schema.sql](../../sql/schema.sql)
- Generator: [src/data_store/ddl.py](../../src/data_store/ddl.py)
- Full conventions: [data access and storage](../guides/data-access.md)

## Related

- [Data store](../modules/data-store.md)
- [Data platform](../architecture/data-platform.md)
- [Add a data source](../guides/add-a-data-source.md)
- [Cube parts](./cube-parts.md)
