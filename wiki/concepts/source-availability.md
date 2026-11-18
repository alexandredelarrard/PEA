---
title: Source availability
description: The distinction between unavailable, missing, and observed-zero data.
type: concept
tags:
  - wiki
  - concept
source_commit: 77d1016e79b9d08c289c5894b1282f30a84ea733
integration_parents:
  - f06e3781ccb5756a0a1406895182de37329e282b
  - 5f7b40bc4dc8379f2e58e8d32acd82a30143b9d1
---
# Source availability

## Definition

Source availability describes whether a source could have produced an observation for a particular ticker, field, and date. It is distinct from a null caused by parsing failure and from an observed zero. Publication start dates, filing lags, listing or identity boundaries, required denominators, and minimum history all contribute.

## Why it matters

Treating unavailable data as zero fabricates negative evidence and changes cross-sectional ranks. The institutional feature system applies explicit availability masks, while validation uses source clocks and a narrow observed-zero declaration to decide whether a feature appeared too early.

Source `start_date` bounds extraction; it neither fills cells nor proves ticker eligibility. [data.yml](../../configs/data.yml) owns each source's exact `pit.column` and calendar `lag_days`. Filing-derived fundamentals, governance, insider and Schedule13D/13G use one day;13F uses filing date plus2; calls use call date plus3; short volume uses one day. Both notes tables use period end plus12 and FTD half-month end plus21. [point_in_time](../../src/utils/publication.py) applies that clock once before sources combine and rolls forward to the first actual session. Raw acceptance/capture dates do not override it.

Cube `warmup` declares read history relative to the actual rewrite start; it does not establish economic coverage. Operational freshness settings and age guards are removed. Economic book age, source completeness, identity, supported intervals and denominator rules remain; unavailable values never become zero merely because a date is configured.

FTD availability uses the persisted ZIP `period` to determine half-month end, then adds the 21-calendar-day lag configured for `sec_fails_to_deliver_security` in [data.yml](../../configs/data.yml) and rolls to a market session. This estimated release clock uses neither the cache modification time nor the settlement day to infer the half; a `b` ZIP can contain day15 settlements. Historical actual SEC release dates remain unverified. See [accepted remaining research](../TODO.md#institutionals-follow-up-validation).

## Where it lives

- Availability configuration: [configs/data.yml](../../configs/data.yml)
- Validation declarations: [configs/validate.yml](../../configs/validate.yml)
- Availability utilities: [availability.py](../../src/data_aggregate/utils/common/availability.py)
- Insider completeness: [inputs.py](../../src/data_aggregate/utils/institutionals/inputs.py) reads one table-wide frontier D, the newest `filing_date` in `insider_transactions` (empty-filing markers included). A quiet window is 0 when the ticker is listed, after its first stored insider filing and on or before D; after D it is NaN. `fetched_at` is stored as data and is not a gate. A filing that arrives late is repaired by the institutional writer's diff rewrite on the next build. A Form 4 is visible one calendar day after its `filing_date`, rolled to the next session (the `insider_transactions.pit` policy in [data.yml](../../configs/data.yml), applied by [inputs.py](../../src/data_aggregate/utils/institutionals/inputs.py)). [frontiers.py](../../src/data_aggregate/utils/institutionals/frontiers.py) retains the existing independent Schedule13D/13G rule.
- Short/FTD completeness: successfully parsed and reconciled source cohorts persist `source_checked_at` on existing ticker rows. Within dated supported security intervals, missing on a successful fetch is zero. Failed/incomplete fetches retry; legacy observed values remain usable without stamps, while absent rows require proof. Missing/zero denominators remain NULL, including0/0. Operational check dates never change public availability.
- Leakage checks: [validate/checks/leakage.py](../../src/validate/checks/leakage.py)
- Source constraints: [data sources](../reference/data-sources.md)

## Related

- [Point-in-time data](./point-in-time-data.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Add a data source](../guides/add-a-data-source.md)
