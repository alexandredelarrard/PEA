---
title: Cube build
description: Price-basis gate, registered incremental parts, point-in-time features, and streamed wide-cube assembly.
type: flow
tags:
  - wiki
  - flow
source_commit: 5fb2fa5e192b2997762a26756c3cd68ab63c0312
---
# Cube build

## Summary

The cube build converts extract tables into eight registered parts. The assembler reads the configured columns, uses every persisted price date/ticker as its row grid, and left-attaches features and wide targets. Readiness compares compulsory and populated optional parts with the latest stored source-price date for the analysis universe. See the shared [assembly policy](../../src/data_aggregate/utils/common/assembly.py) and [readiness report](../../src/data_aggregate/utils/common/part_status.py).

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
5. [StepAssembleCube](../../src/data_aggregate/transformers/step_assemble_cube.py) resolves `build_cube.assembly` from [build_cube.yml](../../configs/build_cube.yml). Prices, targets, betas, fundamentals, momentum and governance are compulsory. Text and institutionals skip only when absent or empty; populated broken inputs fail. Payload reads are projected and price-scoped; a separate key scan prevents hidden key corruption. All price rows survive, including latest unlabelled prediction rows.
6. All selected-column, key, collision, target-schema and metadata checks finish before output mutation. Old output columns are physically removed through the existing store facade after the first chunk is ready; `replace` writes the first chunk and `bulk_seed` later chunks. Final schema and row count are checked.
7. [cube-status](../../src/data_aggregate/utils/common/part_status.py) compares compulsory parts, populated optional parts and cube with the latest stored `prices` date for the analysis universe. Diagnostics name expected/actual edges and refresh commands. Skipped institutionals skip the dependent insider check; populated institutionals retain its existing completeness rule. A red status fails its own task; the prediction trigger still runs (`ALL_DONE`).

## Failure modes

A missing price-basis gate can propagate adjustment seams into returns, betas, and labels. A look-back longer than its declared warm-up makes incremental and full tails diverge. Assembly refuses missing compulsory parts, invalid keys, selected-column collisions and legacy long targets before changing output. Date-edge mismatches remain warnings during assembly and failures at readiness. Streamed replacement is non-atomic: an interruption can leave a partial cube; resolve the failure and rerun assembly from the preserved parts. Existing source/prediction freshness checks detect market-age staleness; aligned tables alone cannot establish that source prices are current.

## Related

- [Cube aggregation](../modules/data-aggregate.md)
- [Cube parts](../concepts/cube-parts.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Add a cube feature](../guides/add-a-cube-feature.md)
