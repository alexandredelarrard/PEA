---
title: SEC and LLM extraction
description: SEC filing discovery, five focused proxy requests, bounded retries and main-thread persistence.
type: flow
source_commit: 58c33f8622d33c4654b139d83677c92e019d15e0
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

[evidence.py](../../src/data_extract/utils/structure/def14a/evidence.py) requires a disclosed value tied to a row or label before repairing an attempted section. Missing plus `retry=1` alone does not schedule a request: the source check must recognize the missing key. Supported layouts include named plaintext Summary Compensation Table salary rows, prefixed/repeated year headers, consecutive fiscal-year ranges and an explicitly labelled transposed audit-fee matrix. The complete header must identify Year/Fiscal Year, optionally with the recorded single-letter column or FY/Summary Compensation Table caption prefix. Pay or quantity labels ending in “year” do not identify a fiscal-year column. This check schedules the matching LLM action; it does not insert regex-derived amounts into fact tables. Unknown layouts can remain unresolved at one. Headers alone, optional silence, dashes and less-than bounds do not prove an exact missing value. An unreadable filing consumes no attempt; a provider failure retains its reserved counter. Empty child sections use hidden `_empty` rows with complete primary keys. Marker stamps retain only the table's registered keys, filing date and permitted CIK metadata. Legacy whole-proxy answers/markers prove a first attempt for all domains; a new split archive does not complete its siblings.

[postprocess.py](../../src/data_extract/utils/structure/def14a/postprocess.py) applies declared currency/scale, converts percentages to fractions, preserves fiscal years, zeros, negatives and real changes, and normally retains saved non-null facts during partial repair. [basis.py](../../src/data_extract/utils/structure/def14a/basis.py) permits a correction, an explicit null or recovery of a labelled monetary SAR award only when the same filing's named, year-matched, labelled source cells prove the value or conflict. Ownership uses the reported beneficial/common-stock basis rather than a final total that adds cash-settled or additional units. Historical option counts and stock-accounting expense are not monetary option awards; an annual-compensation subtotal is not a full SCT total. Foreign or ambiguous currencies/scales remain unresolved.

A 0.001/1000 pay correction needs both the literal source cell and a same-person, same-field median from other years printed in that same filing. The median corroborates the correction; it never replaces source evidence. No later filing or additional cross-filing smoothing supplies an earlier amount. Existing gender consensus and downstream imputation retain their own contracts.

[holders.py](../../src/data_extract/utils/structure/def14a/holders.py) provides one name identity for reconciliation and source matching: remove complete/nested parenthetical text, normalize case and accents, remove punctuation and collapse whitespace. Postal address tails are removed only with explicit address evidence. Aliases reconcile within the exact ticker/accession/as_of/holder-type key, preserving readable display names and saved comparable non-null cells. A bare issuer row takes precedence over a parenthetical foreign-security peer, preventing parent-company shares from filling issuer quantities. A separately supported disjoint ownership table can also replace a proven foreign-class quantity with the printed issuer-component sum when the filing's company name matches exactly; an existing issuer component is retained. Survivors are saved before exact redundant primary keys are deleted through the store; parent/facade counts refresh from the survivors.

For a missing monetary SAR award, source rows must bind the full person name and fiscal year to an explicit dollar column. Equal-length printed year/value vectors can be paired; a director caption supplies its year only to its immediately owned fee table and cannot supply annual executive/grant facts. Counts and ambiguous years remain unknown.

The carver preserves fixed-width compensation blanks and rejects CD&A performance-year summaries as SCTs. Complete pay and ownership domains may replay their archived answers for free only within the candidate filings whose archives are already loaded by the ordinary daily plan. Fully complete filings outside that plan are not scanned again, and archived answers without a matching source payload cannot prove a basis correction. No retry counter is reset and no history-wide paid backfill is triggered.

A missing director fiscal year can be completed before building the keyed row only from the locally owned annual-table caption and matching named monetary cells; a consecutive FY 2019–20 disclosure identifies 2020. A yearless table may correct an existing stock/option category without changing the saved year only when one named row and its cash fees/total prove the same monetary amount and the stock cell is explicitly blank. Unknown source text does not establish a blank.

A common-class group percentage does not establish combined economic ownership when actual preferred-class group holdings are also disclosed. Source corrections preserve a different valid saved all-class percentage and the voting leg. A dash accompanied by an omission threshold does not establish zero compensation; a genuinely printed zero remains zero. Surviving holder display names also drop terminal letter footnotes.

The original paid JSON is retained inside the versioned `_def14a_split` archive member alongside raw section responses and source text; the outer object remains readable by legacy consumers. Raw answers are saved before postprocessing. A restart can replay them for free. Each section preserves independently owned facts and counters. Recovering CEO identity also refreshes parent CEO pay and ownership copies from the same filing's already-saved child rows. A matching person's unsupported copies remain unknown, so values selected for another CEO cannot survive the identity repair.

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
