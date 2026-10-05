---
title: Cube build
description: Price-basis gate, registered incremental parts, point-in-time features, and streamed wide-cube assembly.
type: flow
tags:
  - wiki
  - flow
---
# Cube build

## Summary

The cube build converts extract tables into independently persisted price, target, beta, fundamental, momentum, text, institutional, and governance parts. Each builder uses one registry-defined incremental window, every registered output is expected to end exactly at the price-part maximum date, and the assembler streams a feature-led left join into the final wide cube.

## Trigger

The `data_aggregate` CLI in [data_aggregate/cli.py](../../src/data_aggregate/cli.py) exposes each part separately and a composite build. Airflow chains the same registered commands in [dag_data_aggregation.py](../../src/dags/dag_data_aggregation.py).

## Sequence diagram

~~~mermaid
sequenceDiagram
  participant Driver as CLI or Airflow
  participant Gate as Price validation
  participant Registry as Part registry
  participant Builder as Part builder
  participant Store as DataStore
  participant Assemble as Cube assembler
  Driver->>Gate: validate price basis
  Driver->>Registry: resolve command and warmup
  Registry-->>Builder: part contract
  Builder->>Store: read projected sources
  Builder->>Store: rewrite incremental tail
  Driver->>Assemble: assemble cube
  Assemble->>Store: stream feature chunks
  Assemble->>Store: left join wide targets
~~~

## Steps

1. `StepBuildCube` in [step_build_cube.py](../../src/data_aggregate/step_build_cube.py) runs the price-basis gate unless explicitly skipped.
2. [parts.py](../../src/data_aggregate/utils/common/parts.py) supplies each part's table, command, kind, warm-up, and binding look-backs.
3. [incremental.py](../../src/data_aggregate/utils/common/incremental.py) plans a full build or an inclusive trailing refresh: at least 7 sessions for every part, institutionals included (45 for fundamentals, the longest horizon for targets). A source correction older than that tail needs `--full`.
4. Domain builders load projected source columns and emit one panel at the date/ticker grain. Fundamentals are restricted to the price-supported universe, fiscal changes match the prior fiscal period rather than a fixed daily lag, and quarterly/TTM versus annual-only values expire after their declared source-age limits. The institutional builder reads through [inputs.py](../../src/data_aggregate/utils/institutionals/inputs.py), resolves observed-zero boundaries through [frontiers.py](../../src/data_aggregate/utils/institutionals/frontiers.py), and runs 13F, superinvestor, insider, short-flow, ownership, conditioning, then cross-source panels in that explicit order with one shared [conditioning sink](../../src/data_aggregate/utils/institutionals/sink.py). The insider panel loads the one `insider_transactions` table; its observed-zero frontier, like 13D's and 13G's, is read from the table itself (see [source availability](../concepts/source-availability.md)). [insider_quality.py](../../src/data_aggregate/utils/institutionals/insider_quality.py) drops later repeat copies of a trade and supersedes amended Form 4/A cells point in time before the aggregators run.
5. Before joining, [step_assemble_cube.py](../../src/data_aggregate/transformers/step_assemble_cube.py) compares every registered part maximum with `cube_part_prices.max_date` and emits an explicit missing, behind, or ahead warning for each mismatch. Assembly then merges feature parts, betas, peers, and GICS metadata and left-joins wide targets.
6. The first output chunk replaces the cube and later chunks use `bulk_seed`. `cube-status` reports readiness: every registered part and the final cube must exist at exactly the price edge, and the insider source must be within `insider_max_lag_days` of the institutional part. A red status fails its own task; the prediction trigger still runs (`ALL_DONE`), and `predict` refuses a cube that ends before `prices`.

## Failure modes

A missing price-basis gate can propagate adjustment seams into returns, betas, and labels. A look-back longer than its declared warm-up makes incremental and full tails diverge; the fundamentals part therefore rewrites an inclusive recent tail rather than strict-appending. Feature-name collisions or duplicate date/ticker keys are rejected by the merge layer, and long or duplicate targets are refused before they can multiply rows. Assembly warnings do not abort the write by themselves, so `cube-status` must pass before the cube is treated as ready.

## Related

- [Cube aggregation](../modules/data-aggregate.md)
- [Cube parts](../concepts/cube-parts.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Add a cube feature](../guides/add-a-cube-feature.md)
