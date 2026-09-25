---
title: Cube parts
description: Persisted intermediate tables that isolate feature families and make incremental builds memory-bounded.
type: concept
tags:
  - wiki
  - concept
---
# Cube parts

## Definition

Cube parts are persisted intermediate tables, one per price, target, beta, or feature family. The registry in [parts.py](../../src/data_aggregate/utils/common/parts.py) binds each table to a CLI command, semantic kind, incremental warm-up, and declared look-backs.

## Why it matters

Parts prevent one process from holding every source and feature frame at once. They let Airflow run builders sequentially, allow targeted retries, and give the status gate a concrete freshness surface. Incremental correctness depends on each part rebuilding enough history to reproduce a full build's tail.

Part tables are unmanaged: replacement drops and recreates them so removed feature columns do not linger. The final cube assembler reads the persisted parts and streams output chunks.

## Where it lives

- Registry: [src/data_aggregate/utils/common/parts.py](../../src/data_aggregate/utils/common/parts.py)
- Window planning and writes: [incremental.py](../../src/data_aggregate/utils/common/incremental.py)
- Orchestrator: [step_build_cube.py](../../src/data_aggregate/step_build_cube.py)
- Status gate: [part_status.py](../../src/data_aggregate/utils/common/part_status.py)
- Registry tests: [test_part_registry.py](../../tests/data_aggregate/test_part_registry.py)

## Related

- [Cube aggregation](../modules/data-aggregate.md)
- [Cube build](../flows/cube-build.md)
- [Add a cube feature](../guides/add-a-cube-feature.md)
- [Table registry and store boundary](./table-registry-and-store-boundary.md)
