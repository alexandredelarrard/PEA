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

`src/data_aggregate` converts normalized extract tables into a wide, point-in-time modelling cube. Seven domain sub-steps persist eight intermediate parts through the price maximum date, and a final assembler reports any part-edge mismatch before merging features, betas, peers, and wide targets at one row per date and ticker.

## Responsibilities

- Normalize prices and define the trading grid.
- Build forward targets and rolling factor loadings.
- Build fundamental, momentum, text, institutional, and governance feature panels.
- Apply peer-relative and cross-sectional transformations without future leakage, preserving binary states and structural zeros.
- Match financial changes by fiscal period, enforce publication-time visibility, and expire stale point-in-time values by source cadence.
- Recompute guarded trailing windows and detect schema drift.
- Warn on every missing, behind, or ahead registered part during assembly, while `cube-status` fails closed on any non-exact price-edge alignment.
- Stream the final cube to the store without materializing all output chunks twice.

## Public API / entry points

- `StepBuildCube.run()` and `cube_parts_status()` in [step_build_cube.py](../../src/data_aggregate/step_build_cube.py).
- Eight part commands plus status in [data_aggregate/cli.py](../../src/data_aggregate/cli.py).
- Domain sub-step classes live under `src/data_aggregate/transformers`; [step_cube_prices.py](../../src/data_aggregate/transformers/step_cube_prices.py) and [step_cube_institutionals.py](../../src/data_aggregate/transformers/step_cube_institutionals.py) show the shared orchestration contract.
- `StepCubeInstitutionals.run()` and `build_panel()` in [step_cube_institutionals.py](../../src/data_aggregate/transformers/step_cube_institutionals.py) are the institutional part's persistence and in-memory panel entry points.

## Key files

- [parts.py](../../src/data_aggregate/utils/common/parts.py) is the single part registry.
- [incremental.py](../../src/data_aggregate/utils/common/incremental.py) plans refresh windows and writes inclusive tails.
- [price_frames.py](../../src/data_aggregate/utils/common/price_frames.py), [pit.py](../../src/data_aggregate/utils/common/pit.py), [panel.py](../../src/data_aggregate/utils/common/panel.py), and [xs.py](../../src/data_aggregate/utils/common/xs.py) hold shared contracts. `pit.py` owns fiscal-period matching and source-age projection; `panel.py` applies declared continuous, binary, and structural-zero transform semantics.
- [step_cube_institutionals.py](../../src/data_aggregate/transformers/step_cube_institutionals.py) keeps the ordered institutional merge and persistence contract; [inputs.py](../../src/data_aggregate/utils/institutionals/inputs.py) owns projected and universe-scoped reads, [frontiers.py](../../src/data_aggregate/utils/institutionals/frontiers.py) resolves completeness boundaries, and [sink.py](../../src/data_aggregate/utils/institutionals/sink.py) carries source events and availability into the two derived panels.
- [step_assemble_cube.py](../../src/data_aggregate/transformers/step_assemble_cube.py) logs registry-wide part-versus-price edge warnings, left-joins wide targets onto the feature-led base, and writes chunks. [part_status.py](../../src/data_aggregate/utils/common/part_status.py) is the exact-alignment readiness gate.
- [short_flow_features.py](../../src/data_aggregate/utils/institutionals/short_flow_features.py) publishes FTD values atomically by the persisted source ZIP period, deriving availability from period end plus 15 days or a fresh latest-ZIP cache timestamp. It advances weekend/holiday availability to the next trading session and preserves unavailable `NaN` cells.

- [configs/build_cube.yml](../../configs/build_cube.yml) owns windows, targets, feature settings, and output switches.

## Earnings-call feature contract

`StepCubeText` combines cached sentiment and embedding call-grain metrics into exactly 12 model-facing columns: eight raw values and four prior-only issuer-history scores. Earnings-call features deliberately do not use peer or cross-sectional normalization because transcript availability is sparse and non-synchronous. Current cleaned prepared remarks and Q&A are the quality authority: both must be present, cached sentiment must be complete, cheap word/uncertainty metrics are refreshed from the current text, and embeddings are left-joined only onto quality-valid calls. A valid call becomes visible on the next trading session, remains available for 66 trading sessions, and then expires to null. Missing or malformed quarters remain null; a mathematically observed zero remains zero. Full and incremental runs both calculate on the complete trading calendar, while the shared writer alone slices the incremental refresh tail. Sentiment cache provenance includes the cleaned-text preprocessing version; legacy raw-text scores are rejected and become eligible for one-time rescoring. If a refreshed source becomes malformed, a durable pending marker carries its call date across the extract/aggregate boundary and is acknowledged only after the inclusive cube repair succeeds.

## Dependencies

The module reads through [DataStore](./data-store.md), loads peer dictionaries from [peer deduction](./data-peers.md), and consumes the full set of normalized extract tables from [data extraction](./data-extract.md).

## Participates in

The module is the core of the [cube-build flow](../flows/cube-build.md). Its output feeds [modelling](./modelling.md), while its registry and incremental rules define the [cube-parts concept](../concepts/cube-parts.md).

## Related

- [Add a cube feature](../guides/add-a-cube-feature.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Data access and storage](../guides/data-access.md)
