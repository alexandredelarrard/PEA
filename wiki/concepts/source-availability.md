---
title: Source availability
description: The distinction between unavailable, missing, and observed-zero data.
type: concept
tags:
  - wiki
  - concept
---
# Source availability

## Definition

Source availability describes whether a source could have produced an observation for a particular ticker, field, and date. It is distinct from a null caused by parsing failure and from an observed zero. Publication start dates, filing lags, listing or identity boundaries, required denominators, and minimum history all contribute.

## Why it matters

Treating unavailable data as zero fabricates negative evidence and changes cross-sectional ranks. The institutional feature system carries value-plus-availability payloads, while validation uses source clocks and a narrow observed-zero declaration to decide whether a feature appeared too early.

Configuration dates are only outer bounds; they do not fill missing cells or prove ticker eligibility. Deferred source-history repairs are tracked in [TODO](../TODO.md).

## Where it lives

- Availability configuration: [configs/data.yml](../../configs/data.yml)
- Validation declarations: [configs/validate.yml](../../configs/validate.yml)
- Availability utilities: [availability.py](../../src/data_aggregate/utils/institutionals/availability.py)
- Completeness-frontier resolution: [frontiers.py](../../src/data_aggregate/utils/institutionals/frontiers.py), including all-ticker live insider coverage and complete-universe 13D/13G manifest checks
- Conditioning sink: [sink.py](../../src/data_aggregate/utils/institutionals/sink.py)
- Leakage checks: [validate/checks/leakage.py](../../src/validate/checks/leakage.py)
- Source constraints: [data sources](../reference/data-sources.md)

## Related

- [Point-in-time data](./point-in-time-data.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Add a data source](../guides/add-a-data-source.md)
