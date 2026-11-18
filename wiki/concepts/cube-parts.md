---
title: Cube parts
description: Persisted intermediate tables that isolate feature families and make incremental builds memory-bounded.
type: concept
tags:
  - wiki
  - concept
source_commit: cbc13eb78abd2cfa59465ea780c1092a2b7ecb2e
---
# Cube parts

## Definition

Cube parts are persisted intermediate tables, one per price, target, beta, or feature family. The registry in [parts.py](../../src/data_aggregate/utils/common/parts.py) binds each table to a CLI command, semantic kind, incremental warm-up, and declared look-backs.

## Why it matters

Parts prevent one process from holding every source and feature frame at once. They let Airflow run builders sequentially, allow targeted retries, and give the status gate a concrete freshness surface. Incremental correctness depends on each part rebuilding enough history to reproduce a full build's tail.

Part tables are unmanaged: replacement drops and recreates them so removed feature columns do not linger. There are nine parts owned by eight builders. `cube_part_institutionals` contains insider, ownership and short-flow; `cube_part_sec_thirteen` contains the 80 per-cluster 13F columns (`f_ic_cluster_<id>_<feature>`). The institutionals part computes from full history and the 13F part carries published quarters; both rewrite from their own last stored session minus seven trading-session intervals, inclusively (eight existing dates). Their 390-session warmup metadata is not an input cutoff. Corrections older than that output tail require a full rebuild. The final cube assembler reads the persisted parts and streams output chunks. Seven existing insider cap names change to sums of action-date per-trade ratios. Full replacement of both parts and cube reassembly are necessary for adoption even when column names are unchanged; compatible fresh models are required. The 13F part refuses a changed column set on a normal run and names `build-sec-thirteen -F`. [Definitions](../modules/data-aggregate.md#cluster-features) and the [cluster tests](../guides/testing.md#cluster-13f-features) distinguish source support from implementation correctness.

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
