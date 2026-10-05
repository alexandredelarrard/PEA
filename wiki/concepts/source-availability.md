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

Configuration dates are only outer bounds; they do not fill missing cells or prove ticker eligibility. FTD settlement dates are likewise not availability dates: the [source table](../reference/table-catalog.md) persists each row's ZIP `period`, and the feature builder derives that ZIP's shared clock at build time. Historical ZIPs use period end plus 15 calendar days past weekends. Only the newest stored period within 60 days can use its cached ZIP's New York modification date when it is today or yesterday; this local, short-lived observation can change after a copy or later rebuild. Some `b` ZIPs contain day-15 settlements, so the half cannot be inferred safely from settlement date. Feature publication moves to the next trading session when necessary; missing FTD evidence remains null rather than a fabricated zero. Deferred source-history repairs are tracked in [TODO](../TODO.md).

## Where it lives

- Availability configuration: [configs/data.yml](../../configs/data.yml)
- Validation declarations: [configs/validate.yml](../../configs/validate.yml)
- Availability utilities: [availability.py](../../src/data_aggregate/utils/institutionals/availability.py)
- Completeness-frontier resolution: [frontiers.py](../../src/data_aggregate/utils/institutionals/frontiers.py), one `schedule_complete_through` rule for insider, 13D and 13G, read from the table alone: absence reads as zero through the last price session while the table's newest `filing_date` (markers included) lies within its 7-day resume overlap of that session, otherwise through that newest filing date; an empty table has no frontier. A late filing therefore reads as zero for at most 7 days, until the 7-session cube tail rewrites it
- Conditioning sink: [sink.py](../../src/data_aggregate/utils/institutionals/sink.py)
- Leakage checks: [validate/checks/leakage.py](../../src/validate/checks/leakage.py)
- Source constraints: [data sources](../reference/data-sources.md)

## Related

- [Point-in-time data](./point-in-time-data.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Add a data source](../guides/add-a-data-source.md)
