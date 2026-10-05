---
title: Wiki Log
description: Append-only audit trail of wiki generation and refresh runs.
---

# Wiki Log

Append-only audit trail. Add one dated entry per generation or refresh run, recording the profile, the `source_commit` it was anchored to, and the coverage. The codebase-wiki skill describes the entry shape.

## 2026-09-25: generate

- Profile: internal/standard
- source_commit: 054f0e1
- Coverage: overview; five architecture areas; fourteen application, infrastructure, and test modules; five end-to-end flows; seven core concepts; five task guides
- Pages: [Overview](./OVERVIEW.md)
- Architecture: [System overview](./architecture/system-overview.md), [Data platform](./architecture/data-platform.md), [Orchestration and runtime](./architecture/orchestration-and-runtime.md), [Feature-to-portfolio pipeline](./architecture/feature-to-portfolio.md), [Point-in-time and quality controls](./architecture/point-in-time-and-quality.md)
- Modules: [Runtime and shared utilities](./modules/runtime-and-shared-utils.md), [Constants and configuration](./modules/constants-and-configuration.md), [Data store](./modules/data-store.md), [Data extraction](./modules/data-extract.md), [GPT extraction](./modules/gpt-extract.md), [Peer deduction](./modules/data-peers.md), [Cube aggregation](./modules/data-aggregate.md), [Modelling](./modules/modelling.md), [Strategies](./modules/strategies.md), [Portfolio](./modules/portfolio.md), [Validation](./modules/validation.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [Application and scripts](./modules/application-and-scripts.md), [Tests](./modules/tests.md)
- Flows: [Nightly data refresh](./flows/nightly-data-refresh.md), [SEC and LLM extraction](./flows/sec-llm-extraction.md), [Cube build](./flows/cube-build.md), [Model training and daily prediction](./flows/model-training-and-prediction.md), [Portfolio to trade ledger](./flows/portfolio-to-trade-ledger.md)
- Concepts: [Step pattern](./concepts/step-pattern.md), [Table registry and store boundary](./concepts/table-registry-and-store-boundary.md), [Point-in-time data](./concepts/point-in-time-data.md), [Cube parts](./concepts/cube-parts.md), [Peer-relative features](./concepts/peer-relative-features.md), [Source availability](./concepts/source-availability.md), [Strategy sleeves](./concepts/strategy-sleeves.md)
- Guides: [Run the pipeline](./guides/run-the-pipeline.md), [Add a data source](./guides/add-a-data-source.md), [Add a cube feature](./guides/add-a-cube-feature.md), [Add a model or sleeve](./guides/add-a-model-or-sleeve.md), [Validate a change](./guides/validate-a-change.md)

## 2026-09-25: refresh

- Profile: internal/standard
- source_commit: 054f0e1 (unchanged)
- Coverage: fully read and consolidated all 12 legacy `docs/` pages; added dense reference, operations, coding, testing, data-access, and backlog pages; rewired every wiki documentation link
- Primary pages: [Overview](./OVERVIEW.md), [TODO](./TODO.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Live database](./reference/live-database.md), [Configuration](./reference/configuration.md), [Modelling and portfolio](./reference/modelling-and-portfolio.md)
- Guides: [Run the pipeline](./guides/run-the-pipeline.md), [Data access](./guides/data-access.md), [Coding standards](./guides/coding-standards.md), [Testing and validation](./guides/testing.md), [Add a data source](./guides/add-a-data-source.md), [Add a model or sleeve](./guides/add-a-model-or-sleeve.md), [Validate a change](./guides/validate-a-change.md)
- Rewired sections: [architecture](./architecture/system-overview.md), [modules](./modules/data-store.md), [flows](./flows/nightly-data-refresh.md), and [concepts](./concepts/source-availability.md)
- Agent entry points: [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md)
- Retirement state: legacy `docs/` remains present for human review; package metadata and documentation-sync tooling now select the wiki

## 2026-09-25: refresh — documentation retirement audit

- Profile: internal/standard
- source_commit: 054f0e1 (unchanged)
- Coverage: removed active legacy-path references; verified all twelve former documentation subjects against canonical wiki destinations; made the backlog directly discoverable; corrected validation guidance against the inspected CLI; added source-grounded recovery procedures
- Pages: [Overview](./OVERVIEW.md), [TODO](./TODO.md), [Documentation coverage](./reference/documentation-coverage.md), [Run the pipeline](./guides/run-the-pipeline.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md)
- Agent entry points: [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md)
- Audit boundary: generated caches, tool-managed local state, secrets, report artifacts, and external provider URLs are not repository-documentation dependencies

## 2026-09-26: refresh — institutionals refactor

- Profile: internal/standard
- source_commit: 0d4ebd0
- Coverage: refreshed the institutional cube-part architecture, ordered build flow, completeness-frontier source references, and the insider outlier diagnostic after extracting input loading and frontier resolution from the step orchestrator
- Architecture: the top-level staged pipeline and persistence boundaries are unchanged; the institutional sub-step now delegates I/O to [inputs.py](../src/data_aggregate/utils/institutionals/inputs.py) and completeness decisions to [frontiers.py](../src/data_aggregate/utils/institutionals/frontiers.py)
- Pages: [Cube aggregation](./modules/data-aggregate.md), [Cube build](./flows/cube-build.md), [Source availability](./concepts/source-availability.md), [Application and scripts](./modules/application-and-scripts.md)
- Public contracts retained: `StepCubeInstitutionals.run()`, `build_panel()`, the ordered seven-panel merge, the shared conditioning sink, exact price/share projections, completeness semantics, and `cube_part_institutionals` persistence

## 2026-09-26: refresh — Ruff and Pyright contract

- Profile: internal/standard
- source_commit: 783e6a5
- Coverage: aligned the repository coding guide with the enforced Ruff 0.16.9 and Pyright 1.1.414 contracts after the institutionals refactor; no architecture, persistence, aggregation, feature, or protocol contract changed
- Naming: lower snake case for functions, methods, arguments, and local variables; exception classes end in `Error`; module constants and classes retain capitals
- Runtime syntax: Python 3.13 unions use `A | B`, including the explicit project convention for `isinstance`; every `zip()` states its strictness
- Tooling: [pyrightconfig.json](../pyrightconfig.json) resolves the project-local `.venv`; matching `pandas-stubs` supports pandas boundaries; Airflow-only suppressions remain file-scoped because DAGs run in the separate Python 3.12 environment
- Page: [Coding standards](./guides/coding-standards.md)

## 2026-09-27: refresh — schema-driven extraction freshness

- Profile: internal/standard
- source_commit: bb65544 (was 783e6a5)
- Coverage: replaced the parallel extraction freshness registry with the canonical table metadata in `src/data_store/schema.py`; documented the three-retry hard gate between extraction and aggregation; removed retired Wikipedia and Google Trends tables from freshness checks
- Pages: [Overview](./OVERVIEW.md), [Data extraction](./modules/data-extract.md), [Nightly data refresh](./flows/nightly-data-refresh.md)

## 2026-09-30: refresh — SEC identity and employee extraction

- Profile: internal/standard
- source_commit: 62dc6d3 (was bb65544)
- Coverage: refreshed SEC registrant discovery, per-ticker identity-scope invalidation, truthful XBRL outcomes, standalone employee repair, Airflow ordering, and recovery guidance
- Pages: [Data extraction](./modules/data-extract.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [Nightly data refresh](./flows/nightly-data-refresh.md), [SEC and LLM extraction](./flows/sec-llm-extraction.md), [Run the pipeline](./guides/run-the-pipeline.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md), [Data sources](./reference/data-sources.md), [Table catalog](./reference/table-catalog.md)

## 2026-09-30: refresh — financial notes archive availability

- Profile: internal/standard
- source_commit: 91395e7 (was bb65544)
- Coverage: documented monthly SEC Financial Statement and Notes archive availability, the persisted `available_at` clock, monthly freshness, and fail-closed point-in-time projection from the later of filing and archive availability
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Overview](./OVERVIEW.md)
- Operational boundary: the code and schema declarations are current; the live database still requires the separately authorized column evolution, metadata repair, and full fundamentals-part rebuild

## 2026-09-30: refresh — historical notes availability audit

- Profile: internal/standard
- source_commit: b91495e
- Coverage: corrected the archive-availability contract after finding that current ZIP `Last-Modified` and local cache mtime do not establish original public release dates; historical repair and full rebuild are on hold for point-in-time use
- Pages: [Overview](./OVERVIEW.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md)
- Evidence: the SEC documents later archive corrections; local older ZIPs were acquired in 2026, and their internal build dates differ from SEC dataset update and acquisition dates

## 2026-09-30: refresh — notes estimated and observed date policy

- Profile: internal/standard
- source_commit: 40cba83 (was c679331)
- Coverage: historical notes archives through August 2026 use the next-month 12th, rolled to Monday on weekends; archives from September 2026 use the successful New York download date; existing dates can be corrected by a metadata-only CLI path
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md)
- Operational boundary: no live schema migration, notes metadata repair, or fundamentals-part rebuild was run; the historical 12th is an estimate and does not guarantee strict point-in-time provenance

## 2026-09-30: refresh — FTD ZIP availability and settlement lineage

- Profile: internal/standard
- source_commit: 29f81a3 (was b91495e)
- Coverage: added the ZIP-grain `sec_ftd_vintages` table and the estimated/observed availability policy; documented that source `period`, not settlement day, controls FTD publication, including day-15 rows in some `b` ZIPs
- Pages: [Overview](./OVERVIEW.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Live database](./reference/live-database.md), [Source availability](./concepts/source-availability.md), [Data extraction](./modules/data-extract.md), [Cube aggregation](./modules/data-aggregate.md)
- Measurement: 1,079,328 stored FTD rows, 413 vintage periods, 36,299 date-inferred-period mismatches, and zero settlement dates shared across stored ZIP periods on 2026-09-30

## 2026-09-30: refresh — institutionals merge and model-readiness backlog

- Profile: internal/standard
- source_commit: 11d8735+838064d (pre-merge parent tips)
- Coverage: reconciled FTD ZIP availability and SEC identity/extraction documentation with current `dev`; added the train/holdout label-boundary and historical-universe P0 gates plus institutionals validation follow-ups
- Pages: [Overview](./OVERVIEW.md), [TODO](./TODO.md), [Cube aggregation](./modules/data-aggregate.md), [Data sources](./reference/data-sources.md)
- Validation: the approved FTD ZIP-publication correction moved only `f_ic_ftd_to_adv20` and `f_ic_ftd_persistence_30d` in the frozen short-flow panel; the updated regression passes all 37 outputs and 1,749 hashed columns

## 2026-10-01: refresh — quarterly pension ZIP availability

- Profile: internal/standard
- source_commit: 2f8f8da (working-tree pension fix not yet committed; previous wiki stamp 11d8735+838064d)
- Coverage: SEC Financial Statement Data Sets now retain quarterly `pension_facts` vintages with an estimated historical quarter-end-plus-12-day clock, an observed future download clock, and a feature start at the later of filing and archive availability
- Pages: [Overview](./OVERVIEW.md), [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md)
- Measurement: the table was backed up, recreated, and replayed from 66 CRC-checked ZIPs; 12,742 rows across 66 quarters, no null clocks or duplicate keys, and zero historical date-rule mismatches. The existing fundamentals cube part was not rebuilt and remains stale until a separate full build and assembly.
- Limitation: historical +12 is an estimate, not a verified SEC first-publication date; the generic raw-table leakage validator abstains, while targeted availability/PIT tests and the BDX revision check pass.

## 2026-10-01: refresh — derive FTD ZIP availability during feature builds

- Profile: internal/standard
- source_commit: 7ce4ad7 (uncommitted `harness/ftd-on-the-fly` worktree)
- Coverage: removed the `sec_ftd_vintages` registry, bootstrap DDL, extraction writes, quality-script requirement, and cube read; the feature builder now derives each ZIP date from stored `period` plus 15 days and weekend roll. The globally latest eligible stored ZIP alone may use its local cache date when that New York date is today or yesterday.
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Source availability](./concepts/source-availability.md), [Data extraction](./modules/data-extract.md), [Cube aggregation](./modules/data-aggregate.md), [TODO](./TODO.md)
- Validation: 41 focused tests passed, including partial-universe latest-period selection; 37 aggregate regression outputs and 1,749 hashed columns matched the baseline; a read-only AAPL sample confirmed that the April 15 `202404b` row first publishes May 15.
- Operational boundary: the old physical table may still exist in PostgreSQL; this worktree did not drop it or rewrite persisted cube rows. The 15-day historical clock is an estimate, not verified SEC first-publication provenance.

## 2026-10-01: refresh — employee incremental resume

- Profile: internal/standard
- source_commit: 3e955eb (working-tree employee resume fix; wiki-wide freshness anchor remains 2f8f8da)
- Coverage: routine employee extraction skips filing dates already stored in `fundamentals_employees` even when the accession manifest is empty; `--full` remains the explicit replay
- Pages: [Data sources](./reference/data-sources.md), [Run the pipeline](./guides/run-the-pipeline.md)
- Evidence: the local table had 11,257 rows while its manifest had zero accession outcomes; a read-only plan skipped 31 AAPL and 30 CSCO stored dates, and 14 focused tests passed.

## 2026-10-01: refresh — modelling refactor

- Profile: internal/standard
- source_commit: f9ab52d (harness/modelling-transformers-refactor, merged into dev)
- Coverage: `src/modelling` split into `steps/` (StepLongShort), `transformers/` (model families, Monitor, Backtest) and `utils/`; `long_book` and `trend_cta` sleeves removed; new config keys `model.models_dir`, `model.backtest`, per-family `task`; artifact format `transformer-pickle-v1` (portable member pickles; metadata records library versions)
- Pages: [Modelling](./modules/modelling.md), [Strategies](./modules/strategies.md), [Modelling and portfolio](./reference/modelling-and-portfolio.md), [Configuration](./reference/configuration.md), [Table catalog](./reference/table-catalog.md), [Model training and daily prediction](./flows/model-training-and-prediction.md), [Add a model or sleeve](./guides/add-a-model-or-sleeve.md), [Coding standards](./guides/coding-standards.md), [Step pattern](./concepts/step-pattern.md), [Feature-to-portfolio pipeline](./architecture/feature-to-portfolio.md), [TODO](./TODO.md)
- Agent entry points: [AGENTS.md](../AGENTS.md)

## 2026-10-02: refresh — earnings-call extraction on defeatbeta

- Profile: internal/standard
- source_commit: 166a9f2 plus the uncommitted legacy-retirement phase (harness/earnings-call-extraction-gaps)
- Coverage: one transcript source (HuggingFace `defeatbeta/yahoo-finance-data`) replaces ROIC, Motley Fool and the kurry dataset; the Wikipedia pageviews fetcher is gone; `earnings_call_sections` is re-keyed to (ticker, quarter, paragraph) with the call date as `as_of`; CLI `extract-earnings-calls` and one default-pool DAG task; shared speaker-turn split and cleaning in `src/utils/earnings_call_split.py`; config `earnings_calls` in `configs/data.yml`; tone-drift item deferred to the TODO
- Pages: [Data sources](./reference/data-sources.md), [Table catalog](./reference/table-catalog.md), [Live database](./reference/live-database.md), [Configuration](./reference/configuration.md), [Data extraction](./modules/data-extract.md), [Cube aggregation](./modules/data-aggregate.md), [Run the pipeline](./guides/run-the-pipeline.md), [Data access](./guides/data-access.md), [Data platform](./architecture/data-platform.md), [Nightly data refresh](./flows/nightly-data-refresh.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-01-earnings-call-extraction-gaps/` (full load 33,591 calls in 18.6 min, exact parity with the source index, split `ok` 98.71 %)
- Operational boundary: full FinBERT scoring, the remaining embeddings and the universe `build-text -F` are pending; `cube_part_text` holds 3 sample tickers; the `*_legacy` tables await the user's drop confirmation.

## 2026-10-02: refresh — shared EDGAR extraction driver

- Profile: internal/standard
- source_commit: abd4b53 (branch `harness/edgar-extract-refactor`)
- Coverage: per-ticker EDGAR fetchers declared as `EdgarFetch` and walked by `run_edgar_fetch`; `FilingStamp`; `Identity.filing_scope`; shared `sec_atom.py`, `item_carve.py`, `read_zip_tables` / `pending_periods`; one 13F walk feeding `sec13f_hr` and `sec13f_manager_holdings` plus the per-CIK manager catch-up; one 13D/13G row builder; one insider contract for bulk and live; one Form 3/4/5 pass for both identity tables; superinvestor overrides moved to `configs/sec/`; `gpt.threads` sizes DEF 14A and vote workers; deleted Wikipedia pageview and Google Trends fetchers removed from the docs
- Pages: [Data sources](./reference/data-sources.md), [Data extraction](./modules/data-extract.md), [System overview](./architecture/system-overview.md), [Run the pipeline](./guides/run-the-pipeline.md), [Configuration](./reference/configuration.md), [TODO](./TODO.md)
- Table contracts unchanged; [Live database](./reference/live-database.md) not edited.

## 2026-10-03: refresh — EDGAR extraction validation fixes

- Profile: internal/standard
- source_commit: d86de08 (branch `harness/edgar-extract-refactor`)
- Coverage: `EdgarFetch` has no partial-success flag; `sec_atom.py` owns Atom paging and the entry filter; the 13F listing sorts amendments after same-day originals; the 13F manager catch-up decides per filing from stored (period, filing date) pairs and splits transient from deterministic read failures; review residuals and minor simplifications deferred to the TODO
- Pages: [Data sources](./reference/data-sources.md), [Data extraction](./modules/data-extract.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-01-edgar-extract-refactor/` (`06-final.md`)

## 2026-10-03: refresh — one insider transactions table

- Profile: internal/standard
- source_commit: 843a4eb (branch `harness/insider-transactions-merge`)
- Coverage: `insider_transactions` is the only Forms 3/4/5 table, keyed (`accession_number`, `security_type`, `row_sequence`) with `source`, `owner_ciks`, `n_reporting_owners`, `original_submission_date`, `footnote_ids`, `acceptance_datetime`, `fetched_at`; EDGAR is authoritative and a zip quarter fills only the filings EDGAR missed, with a per-quarter missing-from-EDGAR warning; identity rejects are logged, not stored; the frontier is the EDGAR run's manifest entry; the cube collapses repeat copies and supersedes Form 4/A cells point in time. Removed: the EDGAR staging, coverage and quarantine tables, the accession overlay, `validate insider-parity`, its `validate.yml` thresholds and the `data.yml` bulk cutover date. Full rebuild = drop the table, then `insider-transactions -F`. TODO: the zip/EDGAR encoding item is resolved (one shared encoding) and removed; the Atom 503 item stays.
- Pages: [Table catalog](./reference/table-catalog.md), [Data sources](./reference/data-sources.md), [Configuration](./reference/configuration.md), [Live database](./reference/live-database.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md), [Run the pipeline](./guides/run-the-pipeline.md), [Validation](./modules/validation.md), [Data extraction](./modules/data-extract.md), [Source availability](./concepts/source-availability.md), [Cube build](./flows/cube-build.md), [Application and scripts](./modules/application-and-scripts.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [Point-in-time and quality controls](./architecture/point-in-time-and-quality.md), [Nightly data refresh](./flows/nightly-data-refresh.md), [Documentation coverage](./reference/documentation-coverage.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-03-insider-transactions-merge/` (`03-implementation.md`)
- Operational boundary: the live database still holds the retired tables until the user drops them and runs the refill.

## 2026-10-04: refresh — extraction resume reads only the database

- Profile: internal/standard
- source_commit: 6080c74 (branch `feat/db-derived-resume`)
- Coverage: every extracted table declares a `Resume` contract and, where a read can find nothing, an empty-filing marker in `schema.py`; `resume.py` derives series windows, EDGAR document work lists (local filing index minus stored accessions) and archive work lists from the tables alone; consumer reads drop markers; the run manifest, the bulk sidecars and their config keys are retired; `sec_io` is the one SEC retry policy; fetchers save what they read, retry in rounds and exit 0; one `price-history` download writes prices, dividends and splits; short interest reads day-level holes, with `--repair-gaps` for per-key gaps; the 13F low watermark and the new-ticker backfill from the SEC 13F data sets; the insider EDGAR window starts at the earlier of the day after the last zip quarter and the run date minus 7 days; the insider, 13D and 13G frontiers are read from the tables; every cube part rewrites at least 7 sessions; DAG gates run on `ALL_DONE`; `extraction-status` reports and exits 0; `predict` raises `StaleInputsError`. TODO: the "Extraction resume reads only the database" item is closed and removed.
- Pages: [Data extraction](./modules/data-extract.md), [Nightly data refresh](./flows/nightly-data-refresh.md), [Configuration](./reference/configuration.md), [Run the pipeline](./guides/run-the-pipeline.md), [Large backfills and recovery](./guides/large-backfills-and-recovery.md), [Table catalog](./reference/table-catalog.md), [Data access](./guides/data-access.md), [Data sources](./reference/data-sources.md), [Cube build](./flows/cube-build.md), [Live database](./reference/live-database.md), [Source availability](./concepts/source-availability.md), [Model training and daily prediction](./flows/model-training-and-prediction.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md), [Coding standards](./guides/coding-standards.md), [Add a data source](./guides/add-a-data-source.md), [TODO](./TODO.md)
- Evidence: `reports/validate/2026-10-03-db-derived-extraction-resume/` (`02-plan.md`, `03-implementation-phase-*.md`)
- Operational boundary: the live database gets `sp500_tickers.added_on`, the markers and the cache builds only at the cutover.
