---
title: Cube build
description: Price-basis gate, registered incremental parts, point-in-time features, and streamed wide-cube assembly.
type: flow
tags:
  - wiki
  - flow
source_commit: 77d1016e79b9d08c289c5894b1282f30a84ea733
integration_parents:
  - f06e3781ccb5756a0a1406895182de37329e282b
  - 5f7b40bc4dc8379f2e58e8d32acd82a30143b9d1
---
# Cube build

## Summary

The cube build converts extract tables into nine registered parts. The assembler reads the parts listed in `data_aggregate.assembly`, uses every persisted price date/ticker as its row grid, and left-attaches features and wide targets. Readiness compares compulsory and populated optional parts with the latest stored source-price date for the analysis universe. See the [part probe](../../src/data_aggregate/utils/common/assembly.py) and [readiness report](../../src/data_aggregate/utils/common/part_status.py).

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
3. [incremental.py](../../src/data_aggregate/utils/common/incremental.py) plans a full build or an inclusive trailing refresh: the calendar-day rewrite of each part's `data.tables.<part>.overlap` in [data.yml](../../configs/data.yml), anchored on the part's own saved frontier (30 days for prices, momentum and institutionals, 100 for fundamentals and SEC thirteen, 150 for targets, 7 for text, 380 for governance). A correction older than that tail needs `--full`. The price part compares its old saved rows with the recomputed history and rebuilds only the tickers whose old rows changed; no consumer part widens its tail for a price change.
4. Domain builders load projected source columns and emit one panel at the date/ticker grain. Fundamentals are restricted to the price-supported universe, fiscal changes match the prior fiscal period rather than a fixed daily lag, and quarterly/TTM versus annual-only values expire after their declared source-age limits. [StepCubeInstitutionals](../../src/data_aggregate/transformers/step_cube_institutionals.py) runs insider, short-flow and ownership panels. [StepCubeSecThirteen](../../src/data_aggregate/transformers/step_cube_sec_thirteen.py) then builds the per-cluster 13F panel from the manager labels `build-cluster` wrote. Each owns a separate part and CLI task; the two derived panels and their sink have been removed. Shared inputs are loaded once. The insider panel loads the one `insider_transactions` table; its observed-zero frontier is read from per-ticker filing-inventory proof in that table;13D and13G retain their separate existing frontier rules (see [source availability](../concepts/source-availability.md)). [insider_quality.py](../../src/data_aggregate/utils/institutionals/insider_quality.py) drops later repeat copies of a trade and supersedes amended Form 4/A cells point in time before the aggregators run.
5. [StepAssembleCube](../../src/data_aggregate/transformers/step_assemble_cube.py) resolves `data_aggregate.assembly` from [data_aggregate.yml](../../configs/data_aggregate.yml): `compulsory_parts` and `optional_parts` are the parts to merge (betas are not assembled), `columns` projects each (null = all). It loads the price part as the root (float64), attaches the peer basket and GICS codes, then reads each listed part once on its projected columns (float32 payload) and merges it through `PanelMerger`, which refuses duplicate keys and feature-name collisions and drops rows off the price grid. Text, institutionals and SEC thirteen skip only when absent or empty; populated broken inputs fail. The merged frame is saved with one chunked `store.replace`. All price rows survive, including latest unlabelled prediction rows.
6. All selected-column, key, collision, target-schema and metadata checks finish before output mutation. Old output columns are physically removed through the existing store facade after the first chunk is ready; `replace` writes the first chunk and `bulk_seed` later chunks. Final schema and row count are checked.
7. [cube-status](../../src/data_aggregate/utils/common/part_status.py) compares compulsory parts, populated optional parts and cube with the latest stored `prices` date for the analysis universe. Diagnostics name expected/actual edges and refresh commands. Skipped institutionals skip the dependent insider check; populated institutionals retain its existing completeness rule. A red status fails its own task; the prediction trigger still runs (`ALL_DONE`).

## Failure modes

A missing price-basis gate can propagate adjustment seams into returns, betas, and labels. A look-back longer than its declared warm-up makes incremental and full tails diverge. Assembly refuses missing compulsory parts, invalid keys, selected-column collisions and legacy long targets before changing output. Date-edge mismatches remain warnings during assembly and failures at readiness. Streamed replacement is non-atomic: an interruption can leave a partial cube; resolve the failure and rerun assembly from the preserved parts. Existing source/prediction freshness checks detect market-age staleness; aligned tables alone cannot establish that source prices are current.

## Related

- [Cube aggregation](../modules/data-aggregate.md)
- [Cube parts](../concepts/cube-parts.md)
- [Point-in-time and quality controls](../architecture/point-in-time-and-quality.md)
- [Add a cube feature](../guides/add-a-cube-feature.md)

## Price and security units stage

[StepCubePrices](../../src/data_aggregate/transformers/step_cube_prices.py) reads Yahoo bars, Sharadar reconciliation/share evidence and complete action/predecessor history once. It cleans volume and guarded quote/total repairs, resolves the level history, computes the three PIT outputs, cleans null returns, checks the whole candidate and publishes the existing price part. A zero opening quote becomes null in the candidate; negative or infinite price inputs still refuse publication. Raw sources stay unchanged.

`close_pit` is the historical quoted-security price; `shares_outstanding_pit` is the issuer count in canonical-class equivalents (the public source count times the vendor class coefficient: BRK-A counts as 1,500 BRK-B, V's B and C at their conversion rates) in dated canonical-security units, so `mcap_pit = close_pit × shares_outstanding_pit`; `mcap_pit` is issuer USD capitalization with independent scope evidence. The fields have independent nulls. Consumers must not derive a missing count from cap divided by quote or rebuild a missing cap locally. Consumers convert quantities only through the saved forward basis `security_basis` (`q · F_to / F_from`); the genuine split/quoted-unit adapters remain for the price build and fundamentals. Weighted financial share denominators keep their source definitions. Per-observation vendor-vintage and predecessor-owner proof is required before normalization; a nominal level factor cannot establish a raw count's representation. Financial working comparisons require coherent four-quarter source windows, while raw dividend cash reconciliation keeps its original history.

Adjusted close/return/candle fields remain necessary for momentum and labels. Archive EPS and Yahoo dividends retain documented matched adjusted pairs; momentum's dollar volume reads the saved `close_pit × volume_pit`. These operations do not rerun price reconciliation, reconstruct a second daily PIT series or reapply a price repair. Detailed canonical-price event decisions belong to price creation. Secondary-class raw tape outside that saved universe still needs its existing volume correction once at the institutional input boundary. Consumers retain actionable source-support, seam and feature-population diagnostics.

The price builder compares the saved rows dated before the refresh window with the recomputed history. Each ticker with a differing old row is deleted and rewritten whole, the others get the tail refresh, in one store transaction (an interruption rolls back). `build-prices -t X,Y -F` forces that rebuild for named tickers. Consumers read only PIT columns, which a new split or unit event leaves unchanged before its date, so a price-history change reruns no consumer part; source gap repairs and production adoption are separate reviewed work.

## Institutional stage sequence

[StepCubeInstitutionals.run](../../src/data_aggregate/transformers/step_cube_institutionals.py) follows saved-output planning → shared input loading → insider loading/cleaning/features/masks → short-flow loading/preparation/features → ownership loading/events/features/masks → one panel merge → atomic tail replacement. The family stages process sequentially and reuse their loaded frames. The global FTD stream keeps coverage before ticker filtering, while insider/13D/13G completeness metadata remains independent of payload rows.

The source/calendar history remains complete even for an incremental output tail. `run(full)` selects the effective window before calling `build_panel()` with no unused full flag. Schema recovery writes the already-computed full panel once, without recursive source reads or feature rebuilding. The seven insider market-cap fields sum per-trade action-date ratios. Short/FTD quantities convert through the saved `security_basis` (F_T/F_d), FTD uses the fixed21-calendar-day ZIP estimate, source proof gates quiet zeros, and account/role support gates management fields. Ownership event definitions remain unchanged. See [institutional build stages](../modules/data-aggregate.md#institutional-build-stages).

## SEC 13F stage sequence

[StepCubeSecThirteen.run](../../src/data_aggregate/transformers/step_cube_sec_thirteen.py) plans from the saved part and the manager labels through the shared Window planner, loads the shared resources once, computes only the report quarters published after the saved frontier, checks the newest saved quarter against a recomputation, carries each quarter's values daily and rewrites one window. A quarter `t` is published on the first session on or after `t + 45` days plus the `sec13f_hr` publication lag of [data.yml](../../configs/data.yml) (7 days, so `t + 52`), and takes its members from the labels of `t-1`; the labels come from `build-cluster`, which the DAG runs before the part (`build_manager_profile` >> `build_cluster` beside `build_momentum`, then the cube steps). A changed column set or a relabelled newest saved quarter stops the run before any write and names `build-sec-thirteen -F`. See [cluster features](../modules/data-aggregate.md#sec-13f-cluster-features) for the stages and feature definitions and the [run guide](../guides/run-the-pipeline.md#cluster-labels-and-the-13f-cube-part) for the refit and rebuild sequence.

The cluster features are quarterly and held per ticker: a quarter's frame is the value of every ticker it has a row for from its publication until the next quarter is published, at most `cluster.max_age_calendar_days` (100) days. The 80 columns follow from the `clusters` block of [cluster.yml](../../configs/cluster.yml).

The declared inputs of `data.tables.cube_part_sec_thirteen.warmup` (prices, `sec13f_hr`, manager details, CUSIP identity, Sharadar tickers) are all null, read in full; a finite budget is refused. The rewrite window is the 100-calendar-day overlap before the saved frontier; the saved rows of that window are re-emitted as saved, so a night without a due quarter reads no 13F line.

A stored column set other than the configured 80, a relabelled newest saved quarter, missing labels for a quarter before it, or a saved-date gap in the replacement footprint refuse persistence. After a refit or a feature change, pause the DAG and run `build-sec-thirteen -F` (see the [run guide](../guides/run-the-pipeline.md#cluster-labels-and-the-13f-cube-part)). See [cluster features](../modules/data-aggregate.md#sec-13f-cluster-features) and [validation](../guides/testing.md#cluster-13f-features).

## Text stage sequence

[StepCubeText](../../src/data_aggregate/transformers/step_cube_text.py) runs calendar → plan_window → prepare_sections → run_sentiments → run_embeddings → run_features. Compact source keys are compared independently with each saved model table across history; selected paragraphs are read/prepared once. Shared cleaning removes provider/roster rows and balanced parenthetical content, preserves financial opening speech and keeps Operator handoffs only as split cues. Sentiment saves prepared, question and answer roles independently of embeddings; failed roles remain pending and available partner features remain usable. See the [exact ten formulas](../modules/data-aggregate.md#exact-ten-earnings-call-features).

After the NLP stages, feature creation reads saved scores/vectors and prices for the configured output scope. The normal feature rewrite is seven calendar days; payload history retains active emitters and their actual immediately preceding fiscal quarters, including old symbols. Stored date-only/timestamp forms normalize before completion checks and joins. Availability is call date +3 calendar days rolled forward to the first market session on or after that day, as owned by [data.yml](../../configs/data.yml); expiry uses the configured signal-session horizon. Older corrections require an explicit full feature rebuild and subsequent assembly, reusing valid cached NLP outputs. The [earlier ten-feature acceptance](../guides/testing.md#ten-feature-follow-up-2026-10-22) remains a separately dated historical checkpoint.
