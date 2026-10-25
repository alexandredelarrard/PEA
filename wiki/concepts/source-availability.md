---
title: Source availability
description: The distinction between unavailable, missing, and observed-zero data.
type: concept
tags:
  - wiki
  - concept
source_commit: 1c36d16f5e951e61999a0b11ac33c50cf6381198
integration_parents:
  - 2a3fd08aec2245ec688f671a9d2e1894e4dd1ad8
  - fe27d5cd3fadc3736144e7a0c0216edc63eb1b1f
---
# Source availability

## Definition

Source availability describes whether a source could have produced an observation for a particular ticker, field, and date. It is distinct from a null caused by parsing failure and from an observed zero. Publication start dates, filing lags, listing or identity boundaries, required denominators, and minimum history all contribute.

## Why it matters

Treating unavailable data as zero fabricates negative evidence and changes cross-sectional ranks. The institutional feature system applies explicit availability masks, while validation uses source clocks and a narrow observed-zero declaration to decide whether a feature appeared too early.

Configuration dates are outer bounds; they neither fill cells nor prove ticker eligibility. FTD settlement dates are not publication dates: the persisted source ZIP `period` determines half-period end, then `build_cube.institutionals.ftd_availability_calendar_days` adds21 calendar days and snaps forward to a trading session. This `estimated_policy_21d` policy never reads cache timestamps; historical SEC first-publication dates remain unverified. A `b` ZIP can contain day15 settlements, so settlement day cannot infer the half. See [accepted remaining research](../TODO.md#institutionals-follow-up-validation).

## Where it lives

- Availability configuration: [configs/data.yml](../../configs/data.yml)
- Validation declarations: [configs/validate.yml](../../configs/validate.yml)
- Availability utilities: [availability.py](../../src/data_aggregate/utils/common/availability.py)
- Insider completeness: [inputs.py](../../src/data_aggregate/utils/institutionals/inputs.py) derives per-ticker frontiers from `inventory_checked_through` and fetched accession evidence in the existing transaction table. The newest filing date alone does not prove that older filings were discovered or read. [frontiers.py](../../src/data_aggregate/utils/institutionals/frontiers.py) retains the existing independent Schedule13D/13G rule.
- Short/FTD completeness: successfully parsed and reconciled source cohorts persist `source_checked_at` on existing ticker rows. Within dated supported security intervals, missing on a successful fetch is zero. Failed/incomplete fetches retry; legacy observed values remain usable without stamps, while absent rows require proof. Missing/zero denominators remain NULL, including0/0. Operational check dates never change public availability.
- Leakage checks: [validate/checks/leakage.py](../../src/validate/checks/leakage.py)
- Source constraints: [data sources](../reference/data-sources.md)

## Related

- [Point-in-time data](./point-in-time-data.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Add a data source](../guides/add-a-data-source.md)
