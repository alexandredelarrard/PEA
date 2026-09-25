---
title: Validation
description: Read-only table checks, explicit findings, gates, cache-aware I/O, and insider reconciliation.
type: module
tags:
  - wiki
  - module
---
# Validation

## Summary

`src/validate` is the read-only validation subsystem for pipeline tables and feature panels. It separates table contracts, projected I/O, check execution, structured findings, and pass/fail gates so validation can be reused from the CLI and task-finishing reports.

## Responsibilities

- Resolve table-specific validation declarations from configuration.
- Read only required columns, optionally through a reusable cache.
- Check grain, universe coverage, distributions, redundancy, leakage, clipping, time-series continuity, bounds, and catalogue alignment.
- Reconcile bulk and live-tail insider data at transaction and feature levels.
- Emit JSON-serializable findings, severities, metrics, and gate results.
- Abstain explicitly when a check is not applicable or evidence is unavailable.

## Public API / entry points

- The validation command group and dynamic check commands in [validate/cli.py](../../src/validate/cli.py).
- `TableSpec` and `load_spec()` in [validate/spec.py](../../src/validate/spec.py).
- `Finding`, `CheckResult`, and `gate()` in [validate/result.py](../../src/validate/result.py).
- Cache-aware readers and writers in [validate/io.py](../../src/validate/io.py).

## Key files

- [validate/checks](../../src/validate/checks/) contains the reusable check implementations.
- [validate/checks/leakage.py](../../src/validate/checks/leakage.py) tests label horizons and publication clocks.
- [validate/checks/timeseries.py](../../src/validate/checks/timeseries.py) detects jumps, holes, and frozen legs per ticker.
- [validate/insider_reconciliation.py](../../src/validate/insider_reconciliation.py) runs completed-quarter bulk-versus-EDGAR reconciliation.
- [validate/utils/prices.py](../../src/validate/utils/prices.py) implements the price-basis invariants used to gate cube builds.
- [configs/validate.yml](../../configs/validate.yml) owns table contracts and thresholds.

## Dependencies

Validation reads through [DataStore](./data-store.md) and shares table metadata from the registry. It does not mutate production tables; reports and optional caches are separate outputs.

## Participates in

The price-basis gate runs before [cube aggregation](./data-aggregate.md). Validation is also the final evidence step in [validate a change](../guides/validate-a-change.md) and supports [point-in-time quality controls](../architecture/point-in-time-and-quality.md).

## Related

- [Tests](./tests.md)
- [Source availability](../concepts/source-availability.md)
- [Testing and validation](../guides/testing.md)
