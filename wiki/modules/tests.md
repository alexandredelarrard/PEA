---
title: Tests
description: Pytest fixtures, architectural guards, real-data checks, numeric fingerprints, and pipeline contract tests.
type: module
tags:
  - wiki
  - module
---
# Tests

## Summary

The test suite mirrors the application packages and mixes fast SQLite or fake-store contract tests with bounded real-data economic checks. Synthetic fixtures are reserved for known-truth parsing and mathematics; feature behavior is normally exercised against actual stored data so corporate actions, gaps, and late histories remain visible.

## Responsibilities

- Pin store semantics, SQL isolation, filtering, projections, and partial-upsert behavior.
- Prove cube-part incremental tails match full builds.
- Guard the part registry, DAG chain, final cube grain, and streamed write order.
- Test extraction parsing, source resumption, identity resolution, and source-specific edge cases.
- Test model reproducibility, persistence, constraints, diagnostics, and latest prediction.
- Test sleeve construction, blending, position ledgers, and portfolio behavior.
- Require human-readable sanity-check conclusions in new tests.

## Public API / entry points

Pytest discovers tests under [tests](../../tests/). Shared fixtures live in [tests/conftest.py](../../tests/conftest.py); commands and environment assumptions are documented in [testing and validation](../guides/testing.md).

## Key files

- [test_store_boundary.py](../../tests/data_store/test_store_boundary.py) enforces the sole SQL boundary.
- [test_read_equivalence.py](../../tests/data_store/test_read_equivalence.py) pins bounded store reads against their legacy SQL shapes.
- [test_partial_upsert_merges.py](../../tests/data_store/test_partial_upsert_merges.py) prevents omitted columns from being nulled.
- [test_cube_incremental.py](../../tests/data_aggregate/test_cube_incremental.py) tests warm-ups, projections, and full-versus-incremental equivalence.
- [test_assemble_cube.py](../../tests/data_aggregate/test_assemble_cube.py) pins grain, left-join behavior, and chunked write order.
- [test_aggregate_regression.py](../../tests/data_aggregate/test_aggregate_regression.py) compares frozen numeric fingerprints.

## Dependencies

Tests use pytest, pandas, NumPy, SQLite, fakes, and bounded production data. Optional real-data fixtures skip when their prerequisites are unavailable instead of fabricating replacements.

## Participates in

The suite is the executable half of [point-in-time and quality controls](../architecture/point-in-time-and-quality.md) and the primary verification step in [validate a change](../guides/validate-a-change.md).

## Related

- [Validation](./validation.md)
- [Add a cube feature](../guides/add-a-cube-feature.md)
- [Testing and validation](../guides/testing.md)
