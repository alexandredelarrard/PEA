---
title: Cube build
description: Price-basis gate, registered incremental parts, point-in-time features, and streamed wide-cube assembly.
type: flow
tags:
  - wiki
  - flow
source_commit: 8e1277bbf02cd5995a61a6c1fd13b087ef9cba76
---
# Cube build

## Summary

The cube build converts extract tables into nine registered parts. The assembler reads the configured columns, uses every persisted price date/ticker as its row grid, and left-attaches features and wide targets. Readiness compares compulsory and populated optional parts with the latest stored source-price date for the analysis universe. See the shared [assembly policy](../../src/data_aggregate/utils/common/assembly.py) and [readiness report](../../src/data_aggregate/utils/common/part_status.py).

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
3. [incremental.py](../../src/data_aggregate/utils/common/incremental.py) plans a full build or an inclusive trailing refresh: 7 session intervals for ordinary parts, institutionals and SEC thirteen included (45 for fundamentals, the longest horizon for targets); text has its own 30-calendar-day rewrite from its stored frontier. A source correction older than that tail needs `--full`.
4. Domain builders load projected source columns and emit one panel at the date/ticker grain. Fundamentals are restricted to the price-supported universe, fiscal changes match the prior fiscal period rather than a fixed daily lag, and quarterly/TTM versus annual-only values expire after their declared source-age limits. [StepCubeInstitutionals](../../src/data_aggregate/transformers/step_cube_institutionals.py) runs insider, short-flow and ownership panels. [StepCubeSecThirteen](../../src/data_aggregate/transformers/step_cube_sec_thirteen.py) then runs broad institutional and elite superinvestor panels. Each owns a separate part and CLI task; the two derived panels and their sink have been removed. Shared inputs are loaded once; the broad 13F stream and smaller ordinary manager packet are processed sequentially through shared cleaning and reconciliation. The insider panel loads the one `insider_transactions` table; its observed-zero frontier, like 13D's and 13G's, is read from the table itself (see [source availability](../concepts/source-availability.md)). [insider_quality.py](../../src/data_aggregate/utils/institutionals/insider_quality.py) drops later repeat copies of a trade and supersedes amended Form 4/A cells point in time before the aggregators run.
5. [StepAssembleCube](../../src/data_aggregate/transformers/step_assemble_cube.py) resolves `build_cube.assembly` from [build_cube.yml](../../configs/build_cube.yml). Prices, targets, betas, fundamentals, momentum and governance are compulsory. Text, institutionals and SEC thirteen skip only when absent or empty; populated broken inputs fail. Payload reads are projected and price-scoped; a separate key scan prevents hidden key corruption. All price rows survive, including latest unlabelled prediction rows.
6. All selected-column, key, collision, target-schema and metadata checks finish before output mutation. Old output columns are physically removed through the existing store facade after the first chunk is ready; `replace` writes the first chunk and `bulk_seed` later chunks. Final schema and row count are checked.
7. [cube-status](../../src/data_aggregate/utils/common/part_status.py) compares compulsory parts, populated optional parts and cube with the latest stored `prices` date for the analysis universe. Diagnostics name expected/actual edges and refresh commands. Skipped institutionals skip the dependent insider check; populated institutionals retain its existing completeness rule. A red status fails its own task; the prediction trigger still runs (`ALL_DONE`).

## Failure modes

A missing price-basis gate can propagate adjustment seams into returns, betas, and labels. A look-back longer than its declared warm-up makes incremental and full tails diverge. Assembly refuses missing compulsory parts, invalid keys, selected-column collisions and legacy long targets before changing output. Date-edge mismatches remain warnings during assembly and failures at readiness. Streamed replacement is non-atomic: an interruption can leave a partial cube; resolve the failure and rerun assembly from the preserved parts. Existing source/prediction freshness checks detect market-age staleness; aligned tables alone cannot establish that source prices are current.

## Related

- [Cube aggregation](../modules/data-aggregate.md)
- [Cube parts](../concepts/cube-parts.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Add a cube feature](../guides/add-a-cube-feature.md)

## Institutional stage sequence

[StepCubeInstitutionals.run](../../src/data_aggregate/transformers/step_cube_institutionals.py) follows saved-output planning → shared input loading → insider loading/cleaning/features/masks → short-flow loading/preparation/features → ownership loading/events/features/masks → one panel merge → inclusive output-tail write. The family stages process sequentially and reuse their loaded frames. The global FTD stream keeps coverage before ticker filtering, while insider/13D/13G completeness metadata remains independent of payload rows.

The source/calendar history remains complete even for an incremental output tail. Schema recovery writes the already-computed full panel once, without recursive source reads or feature rebuilding. The refactor retains existing feature mathematics, publication rules and missing-versus-zero behavior; see [institutional build stages](../modules/data-aggregate.md#institutional-build-stages).

## SEC 13F stage sequence

[StepCubeSecThirteen.run](../../src/data_aggregate/transformers/step_cube_sec_thirteen.py) plans the saved output window and loads shared resources once. The broad input uses one projected stream ordered by CIK, report quarter and accession, carrying complete CIK-quarter groups across 200,000-row chunks. The smaller manager input uses one ordinary projected load scoped to historical roster CIKs and the filing cutoff; both preserve completion markers.

[Shared cleaning](../../src/data_aggregate/utils/sec_thirteen/holdings_clean.py) attaches CUSIPs, prepares publication dates and repairs values per accession before reconciliation. Off-universe positions remain in whole-book denominators. [Compact broad books](../../src/data_aggregate/utils/sec_thirteen/institutional_books.py) release raw positions after retaining ticker amounts, security evidence and book states. The manager publication scheduler retains changed-book statistics and conviction; the step exposes selection, public state, contributions, levels and the separate institutional/superinvestor panels. Full sparse history and original calendar ordinals remain dependencies, while normal daily projection uses the effective window: currently 398 computation sessions and eight inclusive rewrite sessions when the saved frontier is current.

Bounded historical probes establish emitted column support; genuine schema drift reprojects the same prepared facts over the full calendar without reloading sources or rebuilding books. A failed previously saved family or a missing saved calendar date inside the actual deletion footprint refuses persistence. A successful nonempty known-schema grid clears an entirely unknown incremental tail; genuinely empty output and all-null full rebuilds preserve saved data. Selection/cohort/CIK/continuity science and non-atomic tail replacement remain [follow-ups](../TODO.md#13f-refactor-follow-ups--completed-review-23-october-2026). See the [detailed stages](../modules/data-aggregate.md#sec-13f-build-stages); no complete 100M computation or 22 GB capacity certification was performed.

## Text stage sequence

[StepCubeText](../../src/data_aggregate/transformers/step_cube_text.py) runs calendar → plan_window → prepare_sections → run_sentiments → run_embeddings → run_features. Compact source keys are compared independently with each saved model table across history; selected paragraphs are read/prepared once. Shared cleaning removes provider/roster rows and balanced parenthetical content, preserves financial opening speech and keeps Operator handoffs only as split cues. Sentiment saves prepared, question and answer roles independently of embeddings; failed roles remain pending and available partner features remain usable. See the [exact ten formulas](../modules/data-aggregate.md#exact-ten-earnings-call-features).

After both model stages, feature creation reads bounded saved scores/vectors and tail-only prices. The output rewrite is 30 calendar days; required payload history is active emitter calls plus their actual immediately preceding fiscal quarters, including old symbols. Retired history z-scores no longer require five years. Stored embedding date-only/timestamp forms normalize before completion and joins. Availability remains call date plus two calendar days and expiry remains 66 original trading sessions. Older changes require a full feature rebuild, reusing compatible model outputs. See [current ten-feature validation](../guides/testing.md#ten-feature-follow-up-2026-10-22) and the separately dated [earlier flow acceptance](../guides/testing.md#text-cube-acceptance-2026-10-22).
