---
title: Nightly data refresh
description: Airflow extraction fan-out, peer deduction, sequential cube parts, assembly, status, and prediction trigger.
type: flow
tags:
  - wiki
  - flow
---
# Nightly data refresh

## Summary

The nightly path refreshes source tables, reports their freshness, and triggers the memory-bounded aggregation chain whatever the fetchers' outcome; only `modelling predict` refuses stale inputs. Extraction tasks are grouped by operational resource pools rather than application package, while cube tasks run sequentially and derive their commands from the part registry.

## Trigger

The `data_extraction` DAG is scheduled daily by [dag_data_extraction.py](../../src/dags/dag_data_extraction.py). Once every fetcher has finished, failed or not, it triggers the unscheduled aggregation DAG in [dag_data_aggregation.py](../../src/dags/dag_data_aggregation.py). Every gate in both DAGs runs on `ALL_DONE`. `sharadar_actions` is refreshed by its own task and by the `fundamentals_sharadar` task, `prices_splits` by `price_history`; a refresh cannot restate stored prices, so `price_history` also re-pulls a ticker whose first-year `close_split` disagrees with Yahoo (see the [price-basis contract](../reference/data-sources.md#price-basis-contract)).

## Sequence diagram

~~~mermaid
sequenceDiagram
  participant Airflow
  participant Extract as Data extraction CLI
  participant Store as PostgreSQL
  participant Peers as Peer deduction
  participant Cube as Cube builders
  participant Predict as Prediction DAG
  Airflow->>Extract: SEC + Sharadar ticker lists, sync universe
  Airflow->>Extract: download SEC bulk zips
  Airflow->>Extract: build and propagate identity
  Airflow->>Extract: ingest zips, refresh EDGAR index
  Airflow->>Extract: run identity-consuming SEC tasks
  Airflow->>Extract: SEC facts walk, then SEC history
  Airflow->>Extract: Sharadar history
  Airflow->>Extract: fundamentals check after both histories
  Airflow->>Extract: EDGAR text walks, then their LLM passes
  Extract->>Store: incremental upserts
  Airflow->>Extract: run schema-driven freshness report
  Extract->>Store: read each declared table frontier
  Extract-->>Airflow: per-table report, exit 0
  Airflow->>Peers: deduce peers
  Peers-->>Airflow: persist peer dictionary
  Airflow->>Cube: run registered parts sequentially
  Cube->>Store: write cube parts and cube
  Airflow->>Cube: run cube status
  Airflow->>Predict: trigger daily scoring
~~~

## Steps

1. Seed or load the S&P 500 universe through [data extraction](../modules/data-extract.md).
2. Cache the SEC bulk zips (`zips_download`: Form 3/4/5, Notes, FTD and the `sec_company_tickers` snapshot), build and propagate the identity layer offline (`identity`: the Notes capture, the FTD lines read from the zip cache, `symbol_tenure`, `universe_cik_symbol`, `universe_cusip`, propagation), then fan out every issuer-identity SEC consumer. The zip ingests are two `sec_bulk` tasks: `zips_ingest_insider` (`zips-ingest --only insider`: pending zip quarters fill the filings EDGAR missed) and `zips_ingest_datasets` (`zips-ingest --only notes --only ftd`). `edgar_index` refreshes the local EDGAR filing index once, over the widest walk window, so no walk refreshes it again that night (`refreshed_today`); the insider walk is refreshed like the others, its `source_history=True` widening its window to the source start; the eight EDGAR walks (`insider_edgar`, `sec_13d`, `sec_13g`, `sec_8k_items`, `filing_text`, `sec_employees_text`, `sec_def14a_text`, `sec_fundamentals_facts`) wait for it in the `sec_api` pool, and `insider_edgar` also waits for `zips_ingest_insider`: it reads every indexed Form 3/4/5 of each ticker's lineage filed from the earlier of the day after the last zip quarter and the run date minus 7 days that is not yet stored from EDGAR, all into `insider_transactions`. The 13F walk, its new-ticker backfill and the 13F manager chain stay independent because they do not resolve issuer filing history; `thirteen-f-backfill` runs after `thirteen-f` so only one EDGAR walk runs at a time, and it does nothing on a night with no new ticker.
3. Run the two fundamentals producers independently: `sec_fundamentals_history` replays the facts the `sec_fundamentals_facts` walk stored, offline; `fundamentals_sharadar` waits only for `identity` and builds `sharadar_fundamentals_history` from Sharadar alone. `fundamentals_check` logs where the two histories disagree once both have finished (ALL_DONE) and writes nothing. Each LLM pass runs in `default_pool` after the walk storing its text: `sec_employees_text >> sec_employees_llm` and `[sec_def14a_text, sec_8k_items] >> sec_def14a_llm >> sec_8k_votes`; the LLM passes read no SEC document. The cube reads headcount from `fundamentals_employees`, which no fundamentals producer waits for. Each task can be repaired and rerun without replaying another. `StepExtractAllData` runs both text walks and the DEF 14A pass, not the employees pass (`sec-employees-llm`).
4. Run `extract-earnings-calls` as one task in the default pool. It reads the transcripts file's footer, then only the row groups whose latest call reaches the stored frontier minus the table's overlap (a new ticker's at any date), and the transcripts of new or re-issued calls only. The split, sentiment and embeddings run later in `build-text`.
5. Each fetcher saves what it read, re-reads failed SEC documents in up to three in-task rounds and exits 0; Airflow retries only a task that raises, up to three times. Per-ticker SEC API walks consume the whole SEC pool so their process-local rate limiters cannot exceed the shared request budget.
6. Run `extraction-status`. Its table inventory, cadence, and publication date column come from `freshness_tables()` in [schema.py](../../src/data_store/schema.py); cadence tolerances come from [constants.py](../../src/constants/constants.py). A table is RED when its newest date is older than its cadence tolerance, or, for the three prediction inputs (`prices`, `sharadar_fundamentals_facts`, `sharadar_fundamentals_history`), when the share of universe tickers whose own latest date is fresh falls below `prediction_fresh_share`. It logs one WARNING per RED table and exits 0. `earnings_call_sections` declares quarterly freshness on `as_of`. Google Trends is not scheduled and declares no freshness; there is no Wikipedia pageviews task.
7. Trigger [peer deduction](../modules/data-peers.md), then the [cube-build flow](./cube-build.md). Every aggregation task runs on `ALL_DONE`, so a failed part does not stop the later parts or the assembly.
8. Run the cube part-status report, then trigger [model prediction](./model-training-and-prediction.md). A red status fails its own task visibly; the insider source is red when the newest stored insider `filing_date` (markers included) is more than `source_freshness.insider_max_lag_days` (4) days behind the reference price date. On a night the insider walk did not run, quiet insider windows after that date stay NaN and the next build rewrites them. `modelling predict` raises `StaleInputsError` and scores nothing when a prediction input's per-ticker fresh share is below `prediction_fresh_share` or the cube ends before `prices`. The prediction DAG's own 06:00 schedule repeats the same guarded run.

## Failure modes

A source that fails does not stop the night: its fetcher keeps what it read and exits 0, and the next run lists the rest again, because every work list is derived from the stored rows. A filing that was read and holds no data for its key is stored once as an empty-filing marker, so it is not listed again; consumer reads never see markers. A failed `edgar_index` holds the eight EDGAR walks; a failed `zips_download` still lets `identity` run on the cached evidence. `extraction-status` reports stale tables and blocks nothing. The `predict` guard is the only hard stop: stale prices, Sharadar fundamentals or the Sharadar fundamentals history, or a cube that ends before `prices`. Aggregation remains sequential because the largest part determines peak memory; parallelizing part builds would add their working sets.

## Related

- [DAGs and infrastructure](../modules/dags-and-infrastructure.md)
- [Source availability](../concepts/source-availability.md)
- [Run the pipeline](../guides/run-the-pipeline.md)
