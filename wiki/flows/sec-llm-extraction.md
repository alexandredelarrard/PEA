---
title: SEC and LLM extraction
description: SEC filing discovery, five focused proxy requests, bounded retries and main-thread persistence.
type: flow
source_commit: 63e7d8f9b5331f56a95d8b4815ed208c7dab5653
tags:
  - wiki
  - flow
---

# SEC and LLM extraction

SEC narrative extraction shares discovery, identity, clients, prompts, concurrency and usage tracking. DEF 14A uses five independent requests. Item 5.07 votes retain their own existing schema and grouped persistence.

## DEF 14A daily flow

The `def14a` CLI invokes `fetch_def14a_llm` in [fetch.py](../../src/data_extract/utils/structure/def14a/fetch.py). Every run lists each accepted CIK window over the configured rolling history. [work.py](../../src/data_extract/utils/structure/def14a/work.py) reads each scoped fact table once, loads candidate archives separately, and replays already-paid answers before considering another request.

| Request/action | Output and counter owner |
| --- | --- |
| `def14a_executive_pay` | `Tables.def14a_executive_comp`; parent CEO/NEO pay summaries are deterministic copies |
| `def14a_director_pay` | `Tables.def14a_director_comp` |
| `def14a_audit` | Auditor identity and all six fee amounts in `Tables.def14a_llm`; this table's retry belongs only to audit |
| `def14a_ownership` | `Tables.def14a_ownership`, plus parent insider/CEO summaries |
| `def14a_governance` | `Tables.def14a_directors`, plus parent board, CEO age/identity and governance disclosures |

Each action has its own schema in [def14a_schema.py](../../src/data_extract/utils/schemas/def14a_schema.py), prompt pair under [prompt_templates](../../src/gpt_extract/prompt_templates), and existing-size budget in [gpt.yml](../../configs/gpt.yml). [carve.py](../../src/data_extract/utils/structure/def14a/carve.py) classifies the filing once and selects domain-specific tables/prose, nearby source units and relevant footnotes. Named historical director payments can be retained without a modern Item 402(k) table; there is no pre-2008 exclusion.

~~~mermaid
sequenceDiagram
  participant Daily as Daily driver
  participant SEC as Filing discovery
  participant LLM as Matching LLM action
  participant Store as DataStore
  Daily->>SEC: List the full rolling filing window
  Store-->>Daily: Scoped facts, counters and candidate archives
  Daily->>Daily: Replay already paid answers
  loop Unread section or disclosed missing key at attempt 1
    Daily->>Store: Reserve the owned retry value
    Daily->>LLM: Domain payload and its schema
    LLM-->>Daily: Ordered parsed answer
    Daily->>Store: Append raw answer and source evidence
    Daily->>Daily: Normalize units and preserve valid partial facts
    Daily->>Store: Save owned facts and summaries
  end
~~~

## Retry and recovery

[retry.py](../../src/data_extract/utils/structure/def14a/retry.py) binds the five schemas/tables. An unattempted section uses `retry=1`; a saved first attempt with a source-present missing key can run once more at `retry=2`. Two is excluded from further calls, including `-F`. There is no call cap, new retry parameter or separate repair command.

[evidence.py](../../src/data_extract/utils/structure/def14a/evidence.py) requires a disclosed value tied to a row or label before repairing an attempted section. Headers, optional silence, dashes and less-than bounds do not prove an exact missing value. An unreadable filing consumes no attempt; a provider failure retains its reserved counter. Empty child sections use hidden `_empty` rows with complete primary keys. Legacy whole-proxy answers/markers prove a first attempt for all domains; a new split archive does not complete its siblings.

[postprocess.py](../../src/data_extract/utils/structure/def14a/postprocess.py) applies only declared currency/scale, converts percentages to fractions, preserves fiscal years, zeros, negatives and real changes, and keeps existing non-null facts during partial repair. Unsupported units remain unknown; historical option counts are never priced as dollar awards. No additional cross-filing smoothing is introduced. Existing gender consensus and downstream imputation retain their own contracts.

The original paid JSON is retained inside the versioned `_def14a_split` archive member alongside raw section responses and source text; the outer object remains readable by legacy consumers. Raw answers are saved before postprocessing. A restart can replay them for free. Parent patches contain only their owned columns, so one call cannot clear another block or change another counter.

## Storage and consumers

`DataStore.prepare_def14a_retries()` in [store.py](../../src/data_store/store.py) transactionally adds missing BIGINT counters and initializes existing rows to one exactly once. Repeated preparation preserves two and does not refill later nulls. Cold tables use canonical DDL. All writes run on the main thread, with existing worker concurrency retained.

The parent default projection excludes large JSON and retry metadata. A shared evidence mask in [filer_tables.py](../../src/utils/filer_tables.py) excludes retry-only rows from governance and prior-CEO vote-role lookup while retaining genuine sparse observations, including audit zero. The legacy paid replacement script refuses writes after the retry contract is installed.

Airflow's DEF 14A task consumes both slots of the SEC API pool, serializing ordinary daily runs. Manually overlapping invocations are outside the exactly-once billing guarantee; a crash after reservation may consume an attempt before a response is saved.

## Other SEC paths

Item 5.07 retains `LLMExtractor.run_extraction()` and the separate vote flattener. Deterministic `sec_def14a` inline-XBRL, 8-K ingestion, tally rules, filing identity guards and point-in-time clocks are unchanged.

## Related

- [Data extraction](../modules/data-extract.md)
- [GPT extraction](../modules/gpt-extract.md)
- [Table catalog](../reference/table-catalog.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
- [Point-in-time data](../concepts/point-in-time-data.md)
