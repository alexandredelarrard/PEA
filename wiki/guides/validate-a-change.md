---
title: Validate a change
description: Select targeted tests and read-only validation evidence, then report a concrete sanity conclusion.
type: guide
tags:
  - wiki
  - guide
---
# Validate a change

## Goal

Prove that a change preserves structural contracts and produces economically sensible data. Test execution rules, interpreter details, and the sanity-print requirement are documented in [testing and validation](./testing.md) and [run the pipeline](./run-the-pipeline.md).

## Steps

1. Identify the narrowest affected package, table, part, model family, or sleeve.
2. Run the directly affected pytest target from the repository root with verbose output and `-s`.
3. Include a printed sanity-check conclusion in any new test.
4. Run the relevant architectural guard when the change touches storage, parts, DAGs, projections, or package boundaries.
5. For a cube change, run registry/projection tests, incremental equivalence, the targeted feature test, and the aggregate fingerprint guard.
6. For a model change, inspect cross-validation, persistence, diagnostics, constraints, and ensemble output.
7. For a data-producing change, run the matching read-only command from [src/validate](../../src/validate/).
8. Retain the validation report and state what was measured, what passed, what abstained, and what remains out of scope.

## Relevant code

- Shared fixtures: [tests/conftest.py](../../tests/conftest.py)
- Architectural storage guard: [test_store_boundary.py](../../tests/data_store/test_store_boundary.py)
- Part and projection guard: [test_cube_incremental.py](../../tests/data_aggregate/test_cube_incremental.py)
- Numeric fingerprint: [test_aggregate_regression.py](../../tests/data_aggregate/test_aggregate_regression.py)
- Validation CLI: [validate/cli.py](../../src/validate/cli.py)
- Structured results: [validate/result.py](../../src/validate/result.py)

## Gotchas

A skipped real-data fixture is not evidence that the behavior passed. Do not run a large-table validation without projection or scoping. Do not regenerate a baseline simply because output changed. Validation may abstain when the table or declaration is unavailable; report that state rather than calling it a pass.

## Related

- [Tests](../modules/tests.md)
- [Validation](../modules/validation.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Run the pipeline](./run-the-pipeline.md)
