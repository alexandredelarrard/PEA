---
title: Point-in-time data
description: The rule that a feature may use only information public by its observation date.
type: concept
tags:
  - wiki
  - concept
---
# Point-in-time data

## Definition

Point-in-time data answers what the system could have known on a given date, not what later restatements say about that period. Filing-derived values become available on filing or publication events, amendments remain separate events, and joins move backward from a trading date to the latest eligible observation.

## Why it matters

Using a fiscal period end, a later amendment, or an exact-match shortcut can leak future information or drop valid observations. The SEC facts table keeps original 10-K and 10-Q filings by accession (amendments are not read, as in Sharadar), the SEC replay emits a complete snapshot at each filing date, and the cube joins three SEC columns to the Sharadar history on the exact `(ticker, as_of)` key, never across dates. These contracts are described in [table catalog](../reference/table-catalog.md).

Cube utilities forward values only from past observations. Targets are forward-looking by definition but remain separate label columns and are excluded from latest-date prediction inputs.

## Where it lives

- SEC facts and replay: [build_history.py](../../src/data_extract/utils/fundamentals/build_history.py)
- Sharadar history: [sharadar_history.py](../../src/data_extract/utils/fundamentals_sharadar/sharadar_history.py)
- Cube join of the two: [source_join.py](../../src/data_aggregate/utils/fundamentals/source_join.py)
- Daily PIT accessors: [pit.py](../../src/data_aggregate/utils/common/pit.py)
- Leakage validation: [validate/checks/leakage.py](../../src/validate/checks/leakage.py)
- Data rules: [data access and storage](../guides/data-access.md)

## Related

- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Source availability](./source-availability.md)
- [Cube build](../flows/cube-build.md)
