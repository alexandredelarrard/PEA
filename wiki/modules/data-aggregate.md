---
title: Cube aggregation
description: Incremental cube-part builders, feature families, target construction, and streamed final assembly.
type: module
tags:
  - wiki
  - module
---
# Cube aggregation

## Summary

`src/data_aggregate` converts normalized extract tables into a wide, point-in-time modelling cube. Seven domain sub-steps persist eight intermediate parts, and a final assembler merges features, betas, peers, and wide targets at one row per date and ticker.

## Responsibilities

- Normalize prices and define the trading grid.
- Build forward targets and rolling factor loadings.
- Build fundamental, momentum, text, institutional, and governance feature panels.
- Apply peer-relative and cross-sectional transformations without future leakage.
- Recompute guarded trailing windows and detect schema drift.
- Stream the final cube to the store without materializing all output chunks twice.

## Public API / entry points

- `StepBuildCube.run()` and `cube_parts_status()` in [step_build_cube.py](../../src/data_aggregate/step_build_cube.py).
- Eight part commands plus status in [data_aggregate/cli.py](../../src/data_aggregate/cli.py).
- Domain sub-step classes live under `src/data_aggregate/transformers`; [step_cube_prices.py](../../src/data_aggregate/transformers/step_cube_prices.py) and [step_cube_institutionals.py](../../src/data_aggregate/transformers/step_cube_institutionals.py) show the shared orchestration contract.
- `StepCubeInstitutionals.run()` and `build_panel()` in [step_cube_institutionals.py](../../src/data_aggregate/transformers/step_cube_institutionals.py) are the institutional part's persistence and in-memory panel entry points.

## Key files

- [parts.py](../../src/data_aggregate/utils/common/parts.py) is the single part registry.
- [incremental.py](../../src/data_aggregate/utils/common/incremental.py) plans refresh windows and writes inclusive tails.
- [price_frames.py](../../src/data_aggregate/utils/common/price_frames.py), [pit.py](../../src/data_aggregate/utils/common/pit.py), [panel.py](../../src/data_aggregate/utils/common/panel.py), and [xs.py](../../src/data_aggregate/utils/common/xs.py) hold shared contracts.
- [step_cube_institutionals.py](../../src/data_aggregate/transformers/step_cube_institutionals.py) keeps the ordered institutional merge and persistence contract; [inputs.py](../../src/data_aggregate/utils/institutionals/inputs.py) owns projected and universe-scoped reads, [frontiers.py](../../src/data_aggregate/utils/institutionals/frontiers.py) resolves completeness boundaries, and [sink.py](../../src/data_aggregate/utils/institutionals/sink.py) carries source events and availability into the two derived panels.
- [step_assemble_cube.py](../../src/data_aggregate/transformers/step_assemble_cube.py) left-joins wide targets onto the feature-led base and writes chunks.
- [configs/build_cube.yml](../../configs/build_cube.yml) owns windows, targets, feature settings, and output switches.

## Dependencies

The module reads through [DataStore](./data-store.md), loads peer dictionaries from [peer deduction](./data-peers.md), and consumes the full set of normalized extract tables from [data extraction](./data-extract.md).

## Participates in

The module is the core of the [cube-build flow](../flows/cube-build.md). Its output feeds [modelling](./modelling.md), while its registry and incremental rules define the [cube-parts concept](../concepts/cube-parts.md).

## Related

- [Add a cube feature](../guides/add-a-cube-feature.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Data access and storage](../guides/data-access.md)
