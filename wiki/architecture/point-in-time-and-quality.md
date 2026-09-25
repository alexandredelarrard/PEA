---
title: Point-in-time and quality controls
description: Cross-cutting safeguards for temporal integrity, incremental rebuilds, validation, and regression testing.
type: architecture
tags:
  - wiki
  - architecture
---
# Point-in-time and quality controls

## Summary

Temporal correctness and data quality are architectural concerns rather than final-stage cleanup. Filing dates drive availability, amendments remain separately observable, source-specific lags are explicit, cube parts rebuild a guarded trailing window, and validation checks can abstain when their contract is not applicable. The detailed data rules live in [data access and storage](../guides/data-access.md), while test conventions live in [testing and validation](../guides/testing.md).

## Diagram

~~~mermaid
flowchart TD
  Filing[Source event or filing] --> Raw[(Raw extract table)]
  Raw --> PIT[Point in time publication state]
  PIT --> Part[Incremental cube part]
  Part --> Checks[Coverage leakage bounds and grain checks]
  Checks --> Cube[(Cube)]
  Cube --> Fingerprint[Regression fingerprint]
  Cube --> Model[Time series cross validation]
  Tests[Unit and real data tests] --> Checks
~~~

## Key components

- The [point-in-time data](../concepts/point-in-time-data.md) contract is implemented across [fundamentals history construction](../../src/data_extract/utils/fundamentals/build_history.py) and [PIT frame utilities](../../src/data_aggregate/utils/common/pit.py).
- [incremental.py](../../src/data_aggregate/utils/common/incremental.py) plans warm-up windows and rewrites an inclusive tail; [parts.py](../../src/data_aggregate/utils/common/parts.py) declares the binding look-backs.
- [src/validate](../../src/validate/) provides grain, coverage, profile, redundancy, leakage, clipping, time-series, bounds, catalogue, and insider-parity checks.
- Architectural tests pin the SQL boundary in [test_store_boundary.py](../../tests/data_store/test_store_boundary.py), DAG/part consistency in [test_dag_matches_part_registry.py](../../tests/dags/test_dag_matches_part_registry.py), incremental equivalence in [test_cube_incremental.py](../../tests/data_aggregate/test_cube_incremental.py), and numeric outputs in [test_aggregate_regression.py](../../tests/data_aggregate/test_aggregate_regression.py).

## Design decisions

Missingness is not automatically zero. Availability is declared by source, publication lag, ticker eligibility, and field dependencies; the validation layer distinguishes an unavailable observation from an observed absence. This is summarized in [source availability](../concepts/source-availability.md).

Economic and feature tests normally use small real-data slices so they encounter actual gaps, amendments, and corporate events. Synthetic fixtures are reserved for known-truth parser or mathematical identities, as enforced by project practice in [tests/conftest.py](../../tests/conftest.py).

## Related

- [Validation module](../modules/validation.md)
- [Tests module](../modules/tests.md)
- [Validate a change](../guides/validate-a-change.md)
- [Add a cube feature](../guides/add-a-cube-feature.md)
