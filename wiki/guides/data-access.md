---
title: Data access and storage
description: Rules for bounded DataStore reads, safe writes, point-in-time data, and incremental processing.
type: guide
tags:
  - wiki
  - guide
  - database
  - point-in-time
---
# Data access and storage

## Goal

Read and write tabular pipeline data without bypassing the SQL boundary, exhausting memory, breaking point-in-time semantics, or creating a second table contract.

## One SQL boundary

[src/data_store](../../src/data_store/) is the only package that may know SQL exists. Outside it, do not import SQLAlchemy, call pandas SQL methods, reach through `store.engine`, open raw connections, or query `information_schema`. [test_store_boundary.py](../../tests/data_store/test_store_boundary.py) enforces this boundary.

If [DataStore](../../src/data_store/store.py) cannot express a required operation, extend its API and test both PostgreSQL and SQLite behavior. Do not add a one-off escape hatch.

## DataStore surface

| Group | Operations |
| --- | --- |
| Introspection | `exists`, `columns`, `row_count`, `bounds`, `max_date`, `distinct` |
| Reads | `load`, `iter_load` |
| Writes | `save`, `replace`, `append_tail`, `bulk_seed`, `delete`, `drop`, `ensure_columns` |

Pass a `Table` from [schema.py](../../src/data_store/schema.py). New code must not pass a string name even when the resolver accepts legacy strings.

There is no `existing_dates` method. Resume with a registry date column, `max_date`, `max_date_by`, `bounds`, or `distinct` as appropriate.

## Reading safely

Always combine projection and row scope on a large table:

```python
df = store.load(
    Tables.sec13f_hr,
    project=True,
    where={"ticker": ["AAPL", "MSFT"]},
    since="2024-01-01",
)
```

Projection rules:

- use `columns=[...]` for an explicit consumer contract;
- use `project=True` for the registry's `read_columns`;
- never pass both;
- declare tolerated live-schema gaps in `optional_columns`;
- use `iter_load` for cube-sized or vector-heavy reads.

`iter_load` requires a projection and enables server-side streaming. Exhaust or explicitly close its iterator because it owns a pooled connection for its lifetime.

Row filters are parameterized and validated against known columns. Scalar values compile to equality, collections to deterministic `IN`, `None` to `IS NULL`, and `store.NOT_NULL` to `IS NOT NULL`. An empty collection matches nothing, never everything. `since` and `until` operate on the registry date column; `until` includes the requested day even for timestamps.

Do not constrain the same date column through both `where` and `since`/`until`; the store raises rather than hide the error behind an empty result.

## Missing and empty tables

`load` raises `TableMissingError` or `TableEmptyError` by default. This keeps an unrun source visible.

Use `optional=True` only when absence is a legitimate state, such as a cold-start resume probe or an explicitly optional feature source:

```python
df = store.load(Tables.def14a_llm, optional=True)
if df is None:
    ...
```

Branch on `is None`, not `.empty`.

## Writing safely

| Method | Use it for | Contract |
| --- | --- | --- |
| `save` | Normal fetcher output | Upsert on registry PK after table/column checks. |
| `replace` | Full rebuild | Managed tables retain DDL; unmanaged cube parts are dropped and recreated. |
| `append_tail` | Incremental aggregate part | Delete and rewrite from a day boundary; inclusive mode lets forward labels mature. |
| `bulk_seed` | Later chunks of a streamed replacement | Append to an already-created schema; unknown columns raise. |
| `delete` | Explicit scoped cleanup | A non-empty `where` clause is mandatory. |

Schema evolution can add columns but never silently remove or retype them. This is why part replacement drops unmanaged tables: a deleted feature must disappear physically, or every later incremental write will see a schema mismatch.

Tabular data belongs in PostgreSQL. Do not substitute CSV or Parquet as an application table. Models, plots, caches, transcripts, and the peer dictionary are non-tabular artifacts and belong under `context.paths`.

## Incremental extraction

Resume from the database frontier, never from a full table read.

- Per-entity providers use `max_date_by` and the shared resume helper. The batch starts at the oldest entity frontier, and upsert makes already-current entities no-ops.
- Market-wide files use one global `max_date`; one downloaded day contains every security.
- Save completed entities or durable chunks immediately.
- Catch provider failures per ticker/filing and continue, but re-raise programming errors in repository code.
- Periodically force full source listings to recover filings missed by a previous bug or out-of-order publication.

The source grain, not the destination primary key alone, determines the correct frontier.

## Point-in-time rules

1. Make information available on its publication or filing date, never the period it describes.
2. Keep originals and amendments distinguishable.
3. Join publication snapshots backward, never forward.
4. Apply source-specific reporting lags and completeness frontiers before feature construction.
5. Do not forward-fill across a period where the source was not observable.
6. Preserve the difference between unavailable, missing, and observed zero.

The fundamentals path demonstrates the pattern: accession-grain facts retain amendments, the SEC replay emits complete filing-date snapshots, and the merged consumer history joins declared SEC fields backward without switching field ownership mid-series. See [point-in-time data](../concepts/point-in-time-data.md).

## Incremental cube parts

[incremental.py](../../src/data_aggregate/utils/common/incremental.py) is the one lifecycle implementation:

```python
window = plan_window(
    store,
    Tables.cube_part_institutionals,
    full=full,
    warmup=self._warmup(),
    trading_index=load_trading_calendar(store),
)
result = write_part(store, Tables.cube_part_institutionals, panel, window, drop_empty=True)
if result == COLUMNS_CHANGED:
    return self.run(full=True)
```

The part registry declares warm-up days and binding look-backs. A trailing recompute is valid only when every bounded daily look-back fits inside that warm-up and cross-sectional transforms are date-local. Tests prove full-versus-incremental equivalence for representative builders.

A declaration can still be wrong. Governance required a longer warm-up because a daily five-year self-history feature lived on top of annual filing inputs. Institutional features with expanding or event-age state compute against the full calendar and slice only at write time; no finite warm-up can reconstruct them.

## SEC and XBRL handling

- Coalesce a priority-ordered candidate tag set per period.
- Derive missing accounting identities only from explicit components.
- Apply non-negative guards and per-issuer deny rules as rejection layers, not hardcoded tag pins.
- Convert booleans to numeric indicators when they become model features.
- Join entities by CIK/CUSIP/accession, not free-text names.
- Keep identifiers as text.
- Treat multi-class shares, consolidated income, Up-C ownership, amendments, and dimensioned facts as specialized contracts; consult [data sources](../reference/data-sources.md) before changing them.

## Artifact boundary

[Context](../../src/context.py) resolves the root, data store, output, models, peer dictionary, and diagnostic paths. Bulk-cache sidecars live beside their own cache directory rather than in a shared metadata file.

Both `data/` and the PostgreSQL volume are risk zones. Ask before modifying or deleting either.

## Checklist

Before merging a data-path change:

1. confirm the table declaration and grain in the [table catalog](../reference/table-catalog.md);
2. use `Tables.<name>` and the store facade;
3. project and scope every large read;
4. model absence explicitly;
5. choose the frontier that matches provider grain;
6. preserve publication time and amendment history;
7. declare look-backs and test incremental equivalence;
8. run a targeted real-data sanity check; and
9. update [live database state](../reference/live-database.md) only after re-measuring it.

## Related

- [Data platform](../architecture/data-platform.md)
- [Data store module](../modules/data-store.md)
- [Table registry and store boundary](../concepts/table-registry-and-store-boundary.md)
- [Add a data source](./add-a-data-source.md)
- [Add a cube feature](./add-a-cube-feature.md)
