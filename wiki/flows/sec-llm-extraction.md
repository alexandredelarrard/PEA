---
title: SEC and LLM extraction
description: From SEC filing discovery and text preparation to ordered structured output and main-thread persistence.
type: flow
tags:
  - wiki
  - flow
---
# SEC and LLM extraction

## Summary

SEC narrative extraction combines shared EDGAR discovery and identity resolution with domain-specific text preparation and a reusable structured-output service. DEF 14A governance and Item 5.07 vote tasks share clients, prompt handling, concurrency, usage tracking, and save discipline while retaining separate schemas and flatteners.

## Trigger

Commands in [data_extract/cli.py](../../src/data_extract/cli.py) invoke the structure or institutional fetchers. `StepExtractAllData` also runs institutionals before structure so the vote parser can read newly stored 8-K narratives.

## Sequence diagram

~~~mermaid
sequenceDiagram
  participant CLI
  participant Edgar as EDGAR driver
  participant Text as Domain preparer
  participant LLM as LLMExtractor
  participant Worker as Provider worker
  participant Store as DataStore
  CLI->>Edgar: list unresolved filings
  Edgar->>Text: filing text and metadata
  Text->>LLM: ordered LlmTask objects
  LLM->>Worker: parallel structured requests
  Worker-->>LLM: schema filled results
  LLM->>LLM: restore submission order
  LLM->>Store: save grouped rows on main thread
~~~

## Steps

1. Resolve allowed registrants and unseen accessions through shared code under [data_extract/utils/common](../../src/data_extract/utils/common/).
2. Download or reuse filing text through [fetch_filing_text.py](../../src/data_extract/utils/structure/fetch_filing_text.py) or read stored 8-K Item 5.07 narratives.
3. Carve domain-specific sections and build `LlmTask` objects with destination tables.
4. `LLMExtractor.run_extraction()` in [gpt_getter.py](../../src/gpt_extract/transformers/gpt_getter.py) executes tasks through providers defined in [providers.py](../../src/gpt_extract/utils/providers.py).
5. Flatten answers and persist rows on the main thread, once per group key.

## Failure modes

Registrant mistakes can hide valid predecessor or successor filings; prompt truncation can omit a required table; and concurrent worker writes can race table creation. The implementation addresses these by central registrant policy, table-aware text carving, action-specific character budgets, ordered results, and main-thread persistence. Fabrication guards remain domain-specific and live beside the DEF 14A or vote extractors.

## Related

- [Data extraction](../modules/data-extract.md)
- [GPT extraction](../modules/gpt-extract.md)
- [Point-in-time data](../concepts/point-in-time-data.md)
- [Add a data source](../guides/add-a-data-source.md)
